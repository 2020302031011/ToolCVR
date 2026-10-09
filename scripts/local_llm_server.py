#!/usr/bin/env python3
"""A minimal OpenAI-compatible chat endpoint backed by transformers.

This exists because the pipeline only needs three things from a language model
server -- ``POST /v1/chat/completions``, ``GET /v1/models`` and greedy JSON
output -- and neither vLLM nor sglang is installed in this environment.  It is
the default launcher for ``run_question.py``; ``--server-command`` overrides it
when a faster stack is available.

Thinking is off by default.  Qwen3 is a hybrid reasoning model, and the callers
in this repository parse the reply with ``parse_json_object``, which does not
strip a reasoning block -- leaving thinking on would turn every structured call
into a parse error, and on CPU it would cost minutes per call besides.
``--enable-thinking`` turns it back on for experiments, and a reasoning block
is stripped from the reply either way.

``--enable-thinking`` is honoured by the *chat template*, and Qwen3 checkpoints
do not all ship the same one.  The template in this repository pre-closes the
block when thinking is off and adds nothing when it is on, so the flag only
removes a constraint -- it cannot force the model to reason.  Replies are
decoded with ``skip_special_tokens=False`` and split on the token that closes a
block, which is what makes both spellings work.
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

_THINK_TERMINATORS = (
    "<｜end▁of▁thinking｜>",  # Qwen3; one special token, not literal text
    "</think>",  # DeepSeek-R1 style distills, in case this server is pointed at one
)
# Chat scaffolding tokens that would otherwise sit in front of the JSON.
_SPECIAL_TOKEN = re.compile(r"<[｜|][^｜|]*[｜|]>")
# The opening of a reasoning block, however a template spells it.
_THINK_OPENER = re.compile(r"^<[｜|]?think[｜|]?>")
# A prompt that ends inside a reasoning block the model still has to close.
_THINK_OPEN_AT_END = re.compile(r"<[｜|]?think[｜|]?>\s*$")


def _prompt_opens_thinking(prompt: str) -> bool:
    """True when the rendered prompt already opened a reasoning block.

    This is read off the prompt rather than off ``--enable-thinking``, because
    the two do not agree.  Qwen3 checkpoints ship different templates: the one
    in this repository injects a *pre-closed* ``<think>\\n\\n</think>\\n\\n``
    when thinking is off and nothing at all when it is on, so the flag never
    produces an opener -- and a reply that then arrives without a terminator is
    a plain answer, not a truncated one.  Asking the prompt what it did keeps
    the two cases separable no matter which template is behind it.
    """
    return bool(_THINK_OPEN_AT_END.search(prompt))


def _strip_thinking(text: str, *, thinking_opened: bool = False) -> str:
    """Drop a leading reasoning block and any leftover chat scaffolding.

    ``skip_special_tokens=True`` is deliberately *not* used when decoding: the
    marker that closes a Qwen3 thinking block is itself a special token, so
    dropping specials first would leave the reasoning text welded to the answer
    with no way to tell where it ends.

    ``thinking_opened`` says the prompt already opened a reasoning block.  Then
    a reply with no terminator is a reply that was cut off before it answered,
    and the only honest thing to return is nothing -- the alternative is handing
    a fragment of reasoning to a caller that will parse it as the answer.  With
    no opener the same reply is indistinguishable from a plain answer, so this
    flag is exactly what makes the two cases separable.
    """
    best_start = -1
    best_end = -1
    for terminator in _THINK_TERMINATORS:
        index = text.rfind(terminator)  # the last block wins, not the first
        if index > best_start:
            best_start = index
            best_end = index + len(terminator)
    if best_start != -1:
        text = text[best_end:]
    elif thinking_opened or _THINK_OPENER.match(text.lstrip()):
        # No terminator, so there is no answer in here -- only reasoning that
        # stopped early.  An empty string makes the caller fail loudly instead
        # of parsing a fragment of reasoning as the reply.
        return ""
    return _SPECIAL_TOKEN.sub("", text).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a local causal LM")
    parser.add_argument("--served-model-name", help="name reported by /v1/models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float16 or float32")
    parser.add_argument("--default-max-tokens", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser


class _Engine:
    """The loaded model, with a lock because transformers is not thread-safe."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.lock = threading.Lock()
        self.name = args.served_model_name or args.model.rstrip("/").split("/")[-1]
        self.enable_thinking = args.enable_thinking
        self.default_max_tokens = args.default_max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=args.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def _prompt(self, messages: list[dict[str, Any]]) -> str:
        template = getattr(self.tokenizer, "apply_chat_template", None)
        if template is None:
            raise RuntimeError("the tokenizer has no chat template")
        try:
            return template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Templates that do not know the flag take no thinking position.
            return template(messages, tokenize=False, add_generation_prompt=True)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        torch = self.torch
        prompt = self._prompt(messages)
        thinking_opened = _prompt_opens_thinking(prompt)
        with self.lock:
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            started = time.monotonic()
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max(1, max_tokens),
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else None,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )
            elapsed = time.monotonic() - started
        prompt_length = encoded["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_length:]
        # ``skip_special_tokens=False`` on purpose: the token that closes a
        # Qwen3 thinking block is a special token, and dropping specials here
        # would weld the reasoning onto the answer with nothing left to split
        # on.  ``_strip_thinking`` removes the scaffolding instead.
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return {
            "content": _strip_thinking(text, thinking_opened=thinking_opened),
            "completion_tokens": int(new_tokens.shape[-1]),
            "elapsed_seconds": elapsed,
            "finish_reason": "stop",
        }


class ChatRequest(BaseModel):
    """One ``POST /v1/chat/completions`` body.

    This has to sit at module scope.  ``from __future__ import annotations``
    turns every annotation into a string, so FastAPI resolves ``ChatRequest`` by
    name against this module's globals -- and a class defined inside ``main`` is
    not in them.  FastAPI does not complain when that lookup fails; it demotes
    the parameter to a *query* parameter, so the request arrives as a 422
    reading ``loc: ["query", "request"]``, which looks nothing like the cause.
    """

    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    stream: bool | None = False


def create_app(engine: _Engine) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": engine.name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, Any]:
        result = engine.complete(
            request.messages,
            max_tokens=request.max_tokens or engine.default_max_tokens,
            temperature=request.temperature or 0.0,
        )
        print(
            f"[server] {result['completion_tokens']} tokens in "
            f"{result['elapsed_seconds']:.1f}s",
            flush=True,
        )
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["completion_tokens"],
            },
        }

    return app


def main() -> int:
    args = _parser().parse_args()
    import uvicorn

    print(f"[server] loading {args.model} on {args.device}", flush=True)
    engine = _Engine(args)
    print(f"[server] ready as {engine.name!r}, thinking={engine.enable_thinking}", flush=True)

    uvicorn.run(
        create_app(engine), host=args.host, port=args.port, log_level="warning", workers=1
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
