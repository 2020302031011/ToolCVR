import json
from unittest.mock import patch

from toolcvr.query import ConservativeQueryRewriter, OpenAICompatibleQueryRewriter


def test_conservative_rewriter_uses_semantic_options_without_inventing() -> None:
    plan = ConservativeQueryRewriter().rewrite(
        "What distinguishes the moisture removal method in Video 3 from the others?",
        [
            "A. Uses a metal press",
            "B. Video 2",
            "C. Employs a sieve",
            "D. Skips moisture removal",
        ],
    )
    texts = [item.text for item in plan.phrases]
    assert "Uses a metal press" in texts
    assert "Employs a sieve" in texts
    assert not any("Video 2" == item for item in texts)
    assert not any("Skips moisture removal" == item for item in texts)
    assert any("Skips moisture removal" in item for item in plan.report_hints)


def test_conservative_rewriter_drops_numeric_count_options() -> None:
    plan = ConservativeQueryRewriter().rewrite(
        "How many red cars are parked by the roadside?",
        ["A. 1", "B. 2", "C. 3", "D. 4"],
    )
    assert len(plan.phrases) == 1
    assert "red cars" in plan.phrases[0].text


def test_qwen_api_rewriter_requests_nonthinking_json(monkeypatch) -> None:
    monkeypatch.setenv("TEST_QWEN_KEY", "secret")
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "retrieval_phrases": [
                                    {
                                        "text": "a metal press removing moisture",
                                        "source": "option_A",
                                        "weight": 1.0,
                                        "target_video_ids": [],
                                    }
                                ],
                                "report_hints": ["compare moisture removal methods"],
                            }
                        )
                    }
                }
            ]
        }
    ).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return response_body

    captured = {}

    def _urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    with patch("urllib.request.urlopen", _urlopen):
        plan = OpenAICompatibleQueryRewriter(
            base_url="https://example.test/v1",
            model="qwen3.7-plus",
            api_key_env="TEST_QWEN_KEY",
            enable_thinking=False,
            thinking_parameter="enable_thinking",
        ).rewrite("How is moisture removed?", ["A. Uses a metal press"])

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert [item.text for item in plan.phrases] == [
        "How is moisture removed?",
        "Uses a metal press",
        "a metal press removing moisture",
    ]
    assert [item.source for item in plan.phrases[:2]] == [
        "original_question",
        "original_option_A",
    ]


def test_deepseek_rewriter_uses_native_thinking_switch(monkeypatch) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret")
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "retrieval_phrases": [],
                                "report_hints": ["preserve temporal evidence"],
                            }
                        )
                    }
                }
            ]
        }
    ).encode("utf-8")
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return response_body

    def _urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    with patch("urllib.request.urlopen", _urlopen):
        OpenAICompatibleQueryRewriter(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-flash",
            api_key_env="TEST_DEEPSEEK_KEY",
            enable_thinking=False,
            thinking_parameter="thinking",
        ).rewrite("What happens?", [])

    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in captured["payload"]


def test_api_rewriter_moves_negative_phrase_to_report_hint(monkeypatch) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret")
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "retrieval_phrases": [
                                    {
                                        "text": "sieve",
                                        "source": "option_C",
                                        "weight": 1.0,
                                        "target_video_ids": [],
                                    },
                                    {
                                        "text": "no moisture removal",
                                        "source": "option_D",
                                        "weight": 1.0,
                                        "target_video_ids": [],
                                    },
                                ],
                                "report_hints": ["compare all methods"],
                            }
                        )
                    }
                }
            ]
        }
    ).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return response_body

    with patch("urllib.request.urlopen", lambda request, timeout: _Response()):
        plan = OpenAICompatibleQueryRewriter(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-flash",
            api_key_env="TEST_DEEPSEEK_KEY",
            enable_thinking=False,
            thinking_parameter="thinking",
        ).rewrite("How is moisture removed?", ["D. Skips moisture removal"])

    assert [item.text for item in plan.phrases] == ["How is moisture removed?", "sieve"]
    assert any("Skips moisture removal" in item for item in plan.report_hints)
    assert any("no moisture removal" in item for item in plan.report_hints)


def test_empty_api_rewrite_does_not_add_original_anchor(monkeypatch) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret")
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "retrieval_phrases": [],
                                "report_hints": ["no grounded visual anchor"],
                            }
                        )
                    }
                }
            ]
        }
    ).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return response_body

    with patch("urllib.request.urlopen", lambda request, timeout: _Response()):
        plan = OpenAICompatibleQueryRewriter(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-flash",
            api_key_env="TEST_DEEPSEEK_KEY",
            enable_thinking=False,
            thinking_parameter="thinking",
        ).rewrite("These clips must be ordered chronologically.", [])

    assert plan.phrases == ()
