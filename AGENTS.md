# ToolCVR engineering constraints

ToolCVR is an independent successor prototype. Do not import mutable runtime
state from, or write results into, `UniCVR-Agent`.

## Logical architecture

The completed framework may contain exactly two concrete logical agents:

- `ObserverAgent`: owns visual `SCAN`, `FOCUS`, and `VERIFY`; it may read raw
  visual inputs and must return source-preserving evidence reports.
- `ReasonerAgent`: owns `PLAN`, `ALIGN`, `ANSWER`, and `STOP`; it consumes
  evidence reports and must not receive raw video frames during normal
  execution.

There is no third comparison agent. Cross-video alignment, correspondence
matching, relation scoring, and paired visual verification are tools used by
the two agents above.

Frame readers, CLIP encoders, query compilers, caches, temporal selectors,
budget managers, evidence stores, and report validators are tools/modules, not
agents. Do not add an `Agent` suffix to them.

## Unified execution

All CrossVid inputs must use the same evidence schema and action protocol. A
task label may be logged but must never select a task-specific solver. Query
retrieval strength must be inferred from available text/visual anchors and the
similarity distribution, not from task names.

The high-level action protocol is `SCAN`, `PLAN`, `FOCUS`, `ALIGN`, `VERIFY`,
`ANSWER`, and `STOP`. Concrete capabilities beneath those actions are tools,
not additional agents.

## Warm-start invariants

- Low-resolution CLIP scanning is separate from expensive VLM observation.
- Every decoded candidate and selected frame retains `video_id`, actual PTS,
  source frame index, segment ID, and feature-cache identity.
- CLIP-filtered regions are not negative visual evidence.
- One VLM request contains at most 16 frames.
- Do not impose a per-video frame-count or candidate-span ceiling after CLIP
  filtering. Preserve every retained frame and split it into successive VLM
  requests of at most 16 frames.
- Discontinuous spans must be labeled explicitly; never imply motion across a
  temporal gap.
- A visual-model request may contain sparse, non-contiguous frames from one
  video, but they must remain in strict timestamp order and retain their
  original `segment_id`. Never mix videos or imply that an unobserved gap is
  continuous evidence.
- VLM reports must be structured, validated, and nonempty.
- Report text consumed by `ReasonerAgent` must be rendered deterministically
  from validated structured evidence; free-form unvalidated VLM text must not
  enter the reasoning state.

## Development

- Keep the mock path deterministic and CPU executable.
- Fail loudly on invalid model output or provenance.
- Do not introduce broad exception swallowing, placeholders, or task routing.
- Run `python -m pytest -q` when pytest is available; otherwise run the test
  functions explicitly and report that pytest was unavailable.
- Tests must assert that exactly two concrete logical agent classes exist.
