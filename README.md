# ToolCVR

ToolCVR is an isolated prototype for a query-conditioned sparse warm start. It
does not import from or write into `UniCVR-Agent`.

The framework has exactly two logical agents:

- `ObserverAgent` reads visual inputs through `SCAN`, `FOCUS`, and `VERIFY`
  tools and returns source-preserving evidence.
- `ReasonerAgent` consumes report text for `PLAN`, `ALIGN`, `ANSWER`, and
  `STOP`; it does not receive raw frames during normal execution.

Cross-video correspondence and verification are tools, not a third agent.
The current milestone implements one executable warm-start path and the
validated inner tool-loop boundary for both agents:

1. Rewrite the question and options into observable visual retrieval phrases.
2. Sample a low-resolution CLIP timeline every 2 seconds (4 seconds for long
   videos) and cache the embeddings by video identity.
3. Rank candidates by the strongest applicable CLIP similarity and retain a
   configurable fraction. When no useful retrieval phrase exists, retain
   semantic change points plus bounded temporal coverage rather than either
   keeping the entire sparse timeline or filtering it with an unrelated query.
4. Preserve every retained candidate span without a per-video frame or span
   ceiling.
5. For inputs explicitly marked as synchronized views, apply the union of all
   selected source-frame positions back to every view while retaining each
   view's independent provenance.
6. Send no more than 16 timestamped frames in one VLM request.
7. Validate structured `event + timestep + evidence_frame_ids` reports while
   recording question relevance in a separate `query_assessment` object.
8. Deterministically render validated events into the only report-text format
   accepted by `ReasonerAgent`; unvalidated free-form VLM text never enters
   the reasoning state.

## Direct warm-evidence ablation

`run_warm_memory_baseline.py` measures whether the validated warm reports
already contain enough information to answer a question. It sends the complete
video-indexed, timestamped event memory directly to the same Reasoner and asks
only for an answer plus optional source event IDs. Answer outcome and citation
compliance are evaluated independently: missing or unknown citations are
recorded as audit metadata without fabricating evidence or discarding a valid
answer, while insufficient evidence is represented explicitly as `ABSTAIN`.
It does not construct an initial
state, issue requirements, call FOCUS, or perform evidence reprojection. The
ground truth is written beside the result for evaluation but is never included
in the model request. This path is an ablation of the shared evidence
representation, not a task-specific solver and not a replacement agent.

Visible events and question relevance are deliberately separate: an irrelevant
but visible action remains an event, while `query_assessment` describes only
that batch's local contribution. `insufficient` means “no validated local
match,” never “the complete multi-video sample lacks evidence.” Free-form local
needs are retained in the report but are not passed into initial state as
global unresolved issues.

Minor JSON shape drift such as a single `unresolved_needs` string is
canonicalized deterministically. If a batch still has no valid event after its
finite repair budget, the report stores an explicit provenance-preserving
`observation_gap` and continues with the remaining batches. A gap is displayed
as “not visual evidence” and cannot be cited as an event; it prevents one bad
batch from silently disappearing or invalidating every valid batch in a sample.

Both agents own a private tool set and choose repeatedly inside their own
bounded loop. The shared execution runtime only validates ownership and
arguments, executes the selected callable, and records a trace; it does not
rank or rewrite calls and is not a third agent. `run_question.py` remains a
warm-start/legacy-planner diagnostic. Multi-round inference is provided by
`run_real_reasoning_cycle.py` and its parallel launchers so legacy artifacts
are never silently mixed with the semantic-zoom path.

## Multi-round semantic zoom

Warm `SCAN` events initialize the current evidence memory as addressable coarse
slots. Intervals not observed by the warm VLM are represented separately as
`coverage` slots: these carry no positive or negative claim, but allow the
Reasoner to recover a region omitted by CLIP. A planning-stage Reasoner call
selects short integer row indices from the current `memory_map`, adds an
observable evidence goal and one mode (`refine` for an observed coarse/fine
slot or `recover` for a coverage slot), and may reserve a complementary next
probe. A deterministic compiler resolves those indices to full provenance
slot IDs. The model therefore never has to reproduce revision-sensitive slot
identifiers, video IDs, or timestamps.

The Observer resolves each selected slot to its complete source region,
samples it at 1 fps, and partitions it into successive calls of at most 16
frames. Only after every child succeeds is the parent slot atomically replaced
by current fine slots. A fine slot remains addressable for a later boundary or
transition refinement. The append-only evidence ledger retains the history,
while the Reasoner sees only the current replace-in-place memory, including
event uncertainty, explicit observation gaps, and unresolved local needs.

The Focus Observer receives a local observation question containing only the
selected video, interval, and evidence need. It does not receive the original
cross-video question, so a local visual report cannot legitimately conclude
facts about another view or an unobserved part of the video.

After each successful zoom, the previous reasoning state is discarded and a
new answer-free WorkingState is projected from the question, options, and
current memory. The projection also emits a small contrastive frontier: each
entry names one open requirement, a locally observable probe, its video scope,
and the distinction that probe could change. It never emits an answer,
confidence, action, or free-form completion summary. Mandatory exploration
uses a planning-only output schema with no `ANSWER` or requirement-status
update. Once the configured focus stages are complete, the same Reasoner enters
its decision schema and may continue planning or produce an evidence-cited
answer.

Requirement state is not patched by language-model deltas in the API path.
The Reasoner chooses an action; current evidence owns state. After every visual
acquisition, reprojection rebuilds the complete WorkingState from the latest
bounded memory snapshot before another policy action is allowed. `ALIGN` is a
separate optional Reasoner action for useful explicit comparison or
correspondence synthesis; it is neither the reprojection mechanism nor a
mandatory pre-answer gate. Invalid schemas, stale provenance, or failed visual
execution are execution failures and remain retryable by the outer runner;
they are never converted into a semantic `STOP`. Consequently a recorded
`STOP` means the Reasoner actually selected `STOP` (or a deterministic runtime
budget/stagnation condition was reached), rather than that JSON validation
failed.

## Multi-round SFT trajectory collection

Each real reasoning sample writes an append-only `trajectory.jsonl` beside
`cycle_result.json`. It is flushed and fsynced after every record so an
interrupted sample still retains all completed calls. A new invocation appends
a new session ID instead of overwriting an earlier failed attempt. Records
include:

- `run_start`: original question/options, evaluation ground truth (never sent
  to a model), inference protocol, source-artifact hashes, and run settings;
- `model_call`: role, model, exact text request, raw response, token usage,
  latency, retry history, and error when present;
- `cycle_progress`: the normalized executable action plus the complete current
  requirements and evidence references, explicitly marked as
  `decision_validated` or `evidence_committed` so a trainer does not confuse a
  post-Focus state snapshot with a second policy action;
- `run_end` or `run_failure`: terminal outcome and artifact/error metadata.

Embedded JPEG base64 is not duplicated in JSONL. Each model-visible visual
message retains its `video/frame ID`, logical timestamp, segment ID, and image
digest. The separate provenance manifest retains the source timestamp and
source frame index needed to reconstruct the visual SFT example, but those
source coordinates are never included in the model prompt. This distinction
prevents a clip cut from a longer recording (notably PSS) from leaking its
original position or answer order. Both accepted generations and
rejected/repair generations remain visible, allowing later filtering or
construction of preference data. API keys are never included.

The Reasoner context contains a complete compact temporal `memory_map` plus a
bounded evidence-rich page. Thus every current region remains addressable even
when all descriptions cannot fit verbatim. Reprojection uses the same design:
a complete compact index and a bounded page that prioritizes newly revised
fine evidence while retaining cross-video/time coverage. It may cite either a
displayed current slot ID or one of its displayed source evidence IDs.

Dataset storage semantics are compiled once into an `inference_protocol` and
passed to every Reasoner call without selecting a task-specific solver. The
protocol defines normalized video identities, answer format, shared versus
independent clocks, synchronized-view alias identity, physical-event counting,
and the meaning of one reasoning cycle. For synchronized UAV inputs, equal
timestamps are explicitly comparable and aliases with the same numeric suffix
(for example `A1`/`B1`) denote the same annotated physical track.

The answer format is also carried as a structured `answer_contract`, rather
than left to prompt interpretation: choice answers are labels, sequence answers
contain every declared input video ID once, interval answers are ordered
numeric pairs, and open answers are text. PSS slot/evidence IDs therefore
cannot leak into the final ordering. FSA explicitly binds the returned interval
to the search video rather than the reference clip. The UAV adapter exposes a
bounded reference-track sidecar containing only aliases already drawn in the
proxy videos, including their source-frame visibility ranges; it never exposes
unlabelled count targets.

Every compact `memory_map` row includes its coarse/fine/coverage level and its
estimated visual-call cost. A task-agnostic plan compiler normalizes
`refine`/`recover` against those levels and explicitly reduces an oversized
multi-slot PLAN to the largest in-order subset that fits the current stage.
The reduction is written into the PLAN reason and the next evidence
reprojection decides what still needs inspection; an over-broad model action
therefore no longer discards the complete sample.

To reuse existing reports and initial states while writing fresh diagnostic
cycle results:

```bash
PYTHONPATH=src /media/data2/xuejj/anaconda3/envs/qwen3/bin/python \
scripts/run_parallel_cycles_with_server.py \
  --llm-gpu 0 --vlm-gpus 2,3 \
  --llm-model /media/data6/xuejj/Qwen3-14B \
  --vlm-model /media/data6/xuejj/Qwen-VL \
  --sample-ids CC-474,MOC-0,MSR-85,FSA-303,PI-35,PI-63 \
  --reports-dir outputs/mechanism-v8-reproject-broad27/reports \
  --state-dir outputs/mechanism-v8-reproject-broad27/state \
  --evidence-dir outputs/mechanism-v8-reproject-broad27/evidence \
  --crossvid-root /media/data6/xuejj/CrossVid \
  --visual-call-budget 8 --minimum-focus-stages 2 \
  --output-root outputs/slot-focus-diagnostic6
```

## FOCUS locator interface

`FocusTool` uses the same `ObserverAgent` and event-report schema as warm
observation. It accepts either a single center timestamp or an explicit time
span. A timestamp is automatically expanded by the configured radius; both
forms are sampled into one dense micro-clip of at most 16 frames. There is no
task-specific routing and no separate report dialect:

```python
FocusRequest(
    video_id="v1",
    center_timestamp_seconds=42.0,
    evidence_need="inspect the action immediately before and after this moment",
)

FocusRequest(
    video_id="v1",
    time_span=(38.0, 46.0),
    evidence_need="verify the object state transition",
)
```

`FocusRequest` names exactly one video, because one VLM request may never hold
frames from two videos. A round that needs several videos is therefore a
*bundle*: one look per video, one `FocusTool` call per look, and the evidence
merged only afterwards, when the state is updated.

The videos in a round are the issue's own `scope`, narrowed by the videos the
chosen `candidate_request` named when it named any of them. Order is
least-cited first, ties on scope order, so the first look is the video the state
has accounted for least. Every video in that bundle is looked at; there is no
"most promising one" to pick, and the class order
(`uncited` before `starved` before `isolated` before `uninterpreted`) decides
*which issue* is served, not which video.

### A window always contains its anchors

A look's window is inherited from the events the state already cites for that
video. `maximum_span_seconds` is a *selection* budget, not a ceiling on the
window: it bounds the unpadded union of the anchors the window may hold, so a
window is at most the budget plus two margins — unless a single observation is
longer than the budget, in which case the window grows to contain it and that
look records `window_budget_exceeded: true`. Truncating the window instead
would return a window that does not contain the evidence it is about, which is
the one thing this function must never do (real MOC-0 did exactly that before
the invariant was enforced: a 92-second anchor in a 60-second window). Dropping
the oversized anchor was rejected as the alternative: on that sample it is the
only anchor, so dropping it would turn a plannable sample into a refusal whose
reason is our own configuration rather than anything about the evidence.

Because a look's window is sampled into at most 16 frames regardless of its
width, a widened window costs sampling density, not VLM calls.

A window is bounded on the other side by the video it points at, and the two
bounds are not symmetric. "The window contains the anchors it names" and "the
window is inside its file" are absolute; the margin and the minimum width are
preferences of our own configuration. `_place_window` is written in that order
— every clamp moves one edge toward the anchor union and never past it, so
containment holds by construction rather than by repair — which is what makes
the answer to "a 6-second clip and a 30-second minimum width" a property of the
code rather than a case someone thought of. Where the configuration and the
medium disagree the medium wins, the minimum is taken from the right first and
what the right cannot give from the left, and the look records
`window_clipped_to_medium: true`. Where an *anchor* reaches past the medium the
report is citing frames the file does not have, and the sample is refused
rather than clipped around: clamping the anchors to the file first would
"fix" that case by moving the anchors out of their own window, trading one
absolute clause against the other.

The planner does not decode anything to learn a file's length: durations are an
input, read from the same report the event catalogue comes from, so planning
stays media-free.

Both absolute clauses are checked as properties rather than as examples —
random anchor unions, durations across four orders of magnitude, paddings wider
than their clip and minimum widths longer than it, asserted against a
re-derivation of the contract
(`test_the_placement_contract_holds_on_random_media_and_configurations`). The
three window defects this corpus produced were each found by one sample that
happened to reach them, each repaired where it appeared; the property test is
what makes the next one fail there instead.

The executor resolves a span against the last decoded frame, not the
container's declared length, so a clipped window and its executed span can
differ by up to one frame (real CC-474: 0.04s at 25fps). That is a definition
difference — how long the file is versus when its last frame is — rather than a
plan that went unheeded: on the real 27-sample corpus, 75 of 76 executed spans
match their planned window exactly, and the one exception is that frame.

### Partial bundles

A video whose reports hold no event at all can inherit no window. It is named in
`unanchorable_videos` and the rest of the bundle is still looked at; the issue
stays unresolved, and that field is what distinguishes "this round did not need
that video" from "that video has nothing to show". Only a bundle in which every
video is unanchorable is refused. `uncovered_request_targets` is the third and
last case: videos a candidate request named that the issue's scope does not
cover, which no round will look at until some issue covers them.

### On disk

```json
{
  "plan": {
    "issue_id": "U01",
    "issue_class": "isolated",
    "evidence_need": "Compare the observed states across the cited videos.",
    "evidence_need_source": "candidate_request",
    "evidence_need_note": "",
    "looks": [
      {
        "video_id": "v1",
        "request_kind": "time_span",
        "time_window": [38.0, 44.0],
        "anchor_event_ids": ["v1-b0001-e1"],
        "anchor_source": "state_citation",
        "dropped_anchor_event_ids": [],
        "window_budget_exceeded": false,
        "window_clipped_to_medium": false
      },
      {
        "video_id": "v2",
        "request_kind": "time_span",
        "time_window": [0.0, 10.008],
        "anchor_event_ids": ["v2-b0001-e1"],
        "anchor_source": "state_citation",
        "dropped_anchor_event_ids": [],
        "window_budget_exceeded": false,
        "window_clipped_to_medium": false
      }
    ],
    "bundle_videos": ["v1", "v2"],
    "unanchorable_videos": [],
    "partial": false,
    "request_target_videos": ["v1", "v2"],
    "uncovered_request_targets": []
  },
  "refusals": [],
  "considered": ["U01"],
  "issue_classes": {"U01": "isolated"},
  "deferred": [],
  "contains_answer": false,
  "run": {
    "planner_identity": "focus-planner-v5-window-placement-contract",
    "state_source": "...", "state_sha256": "...",
    "report_source": "...", "report_sha256": "...",
    "video_durations": {"v1": 347.08, "v2": 210.51},
    "...": "..."
  }
}
```

The one sentence the model writes is `evidence_need`, and it belongs to the
whole round: it is appended to every call's prompt unchanged, so a comparison
("compare the chef in video 1 with the chef in video 2") is visible to both
looks without being regenerated per video. `evidence_need_source` says where the
sentence came from — `candidate_request` when state formation already wrote and
validated it, `generated` when no request names this issue or the request's
wording fails the guard that every sentence must pass. The builder is the
fallback, not the default: a candidate request is a better *source*, not a
licence to skip the check, and `evidence_need_note` records which of the two
happened. `refusals` lists every issue
that was considered and could not be planned, `deferred` the ones sorted behind
the winner, so `considered == refusals + deferred + the winner` and a planner
that could only serve part of the state cannot be mistaken for one that had
nothing to do. `run.planner_identity` is checked on `--resume`: a plan written
by older planning rules is regenerated rather than silently reused, because the
state digest alone cannot see a change in the rules.

### Executing a plan

`execute_focus_plan` (`src/toolcvr/focus_execution.py`) is the bridge between
the planner and `FocusTool`: one call per look, in plan order, one video each,
returning a `FocusEvidenceBundle`. It keeps the plan's sentence, video order,
segment IDs, and the `unanchorable_videos` record; a look whose video has no
`VideoInput` fails the round rather than coming back short, because a bundle
missing one of its videos is not the evidence the plan asked for. Reporters
number their events inside one batch (`e1`, `e2`, …), so two looks can return
the same id for different events; `repeated_event_ids` records that instead of
renaming, since report text embeds the ids verbatim and the rename belongs to
the boundary that lifts a round into the sample-wide namespace.

`scripts/execute_focus_plans.py` is that function as a stage: it reads the
plans, resolves every video through the same CrossVid adapter the reports were
built with, and writes `TASK/SAMPLE/focus_evidence.json`. It is the only place
in the repository that turns a planned window into frames.

```bash
cd /media/data6/xuejj/ToolCVR
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
/media/data2/xuejj/anaconda3/envs/qwen3/bin/python \
scripts/execute_focus_plans.py \
  --plan-dir outputs/mechanism-v2-broad27/plan \
  --reports-dir outputs/mechanism-v2-broad27/reports \
  --output-dir outputs/mechanism-v2-broad27/evidence \
  --vlm-model /media/data6/xuejj/Qwen-VL \
  --vlm-device cuda:0 \
  --resume
```

`--mock-reporter` replaces the model with the deterministic CPU reporter, which
exercises the whole path — decoding, windowing, provenance, the 16-frame limit
— without saying anything about what is in the frames. `--resume` reuses a
sample only while its plan digest and reporter identity both still match, so a
re-planned sample is re-executed and a sample observed with a different model
is not silently mixed in.

Each result's run block also records the reporter identity the original report
used (`observation_reporter_identity`), so a round can be checked for having
looked through the same eyes as the report it was planned from rather than
assumed to have.

### Provenance

Every artifact records the digest of every input it consumed, and every
consumer checks it before using the artifact. The records are one chain, and
each link is checked by the stage that consumes the earlier one:

| artifact | records | checked by |
| --- | --- | --- |
| `initial_state.json` | `report_source`, `report_sha256` | `plan_focus.py` |
| `focus_plan.json` | `state_source`, `state_sha256`, `report_source`, `report_sha256`, `planner_identity`, `builder_identity` | `plan_focus.py` on `--resume`, `execute_focus_plans.py` before executing |
| `focus_evidence.json` | `plan_source`, `plan_sha256`, `report_source`, `report_sha256`, `reporter_identity` | `execute_focus_plans.py` on `--resume` |

Writing a record that nothing reads verifies nothing, and the failure mode is
not hypothetical: the digests were recorded for every run this repository has
ever done and compared by no one, so an executor pointed at a plan built from
one observation and a report from another would have produced a bundle whose
every field was well formed and whose provenance was a lie. The comparison is
on content rather than location, so an outputs tree moved intact still
verifies, while a file edited in place — or a run pointed at another sample's
file — is refused by name with both digests in the message. Provenance is
checked before the per-sample `try`, as one wiring mistake to fix rather than
an error line per sample to read. The rule lives in
`src/toolcvr/provenance.py` and is shared by both entry points rather than
re-implemented in each.

`--resume` is the same question asked about a finished artifact, and it adds
the producer to the answer: an artifact is a function of its inputs *and* of
the rules that made it, so a changed `planner_identity` or `reporter_identity`
regenerates a file whose every digest still matches.

A producer is not always one field. A focus plan has two — `planner_identity`
for the rules that placed its windows, `builder_identity` for the model that
wrote its `evidence_need` sentence — and comparing only the first is a check
that always passes: a mock plan and a model-planned plan name the same rules,
so the artifact a `--mock --resume` overwrote and the one it left behind were
indistinguishable in every field on disk. `reuse_decision` therefore takes the
identities as a mapping and checks all of them, both for staleness and for the
refusal below.

Rebuilding has a direction, and only one of them loses something. Rebuilding
because the rules changed is what `--resume` is for. Rebuilding a *model's*
artifact *with a mock's* is not: the mock's output is well formed, passes every
check the model's did, and leaves nothing on disk saying the corpus stopped
being an observation. That is not hypothetical here — a `--mock --resume` over
a corpus whose 27 states a local Qwen3-14B had built rebuilt all of them as
mock, and the files overwritten were the only copy; the run log recorded the
real commands and was not read first. So `reuse_decision` refuses that
direction by name, before anything is written, naming the file, both builders,
and the flag that means the operator meant it:

```
$ scripts/build_initial_state.py --mock --resume ...      # over LLM-built states
ProvenanceError: /.../CC-474/initial_state.json was built by
'openai-compatible-v3-anchored-local-summary:Qwen3-14B' (builder_identity) and
this run would rebuild it with 'deterministic-mock-v2-local-assessment', which
produces synthetic evidence rather than a model's: the corpus would stop being
an observation and nothing on disk would say so. Pass --allow-builder-downgrade
if that is what you mean.
```

The message names the recorded field, because with more than one producer the
question a reader has is which of them disagreed.

The upgrade direction is not blocked, and neither is mock over mock: the
corpus runs a mock reporter against reports a visual model wrote, and that is
the normal case. `--allow-builder-downgrade` exists on all three entry points.

A reporter identity names its model as `<path>@<16 hex>`, the digest of that
directory's file list with sizes and modification times — not of the bytes,
which for a multi-gigabyte model would mean re-reading it every run, but
enough to tell the same weights from re-downloaded or quantized ones. The
recipe lives in `toolcvr.provenance.local_qwen3vl_identity_suffix` and is
built once for both stages, because its only use is a comparison: an evidence
bundle records `reporter_identity` and `observation_reporter_identity` side by
side so a reader can ask whether the focus round looked through the same eyes
as the report it was planned from. Two copies of that string-building had
already stopped agreeing — one used compact JSON separators and the other did
not, so the same directory was named `36ed3aa315d8884d` in every report and
`4f2ef03c6daa3b90` in every bundle beside it, and one copy named no overlay
setting at all. Neither disagreement raised anything. They now name the same
reporter in the same words; whether a run *should* use the same reporter is a
separate question, and for the mock path — a deterministic CPU reporter
executing plans planned for a visual model — the answer is deliberately no.

Review, state-delta generation, and the merge itself are stage 5 and are not
implemented. The merge point already exists — `state_update.apply_state_delta`
accepts an `AddInformation` naming several videos — so what a round owes it is
exactly this bundle.

## Generate report text from a frozen selection

Selection and observation are separate stages. The report command below does
not load CLIP and does not run the rewrite API again. It reads the exact
`frame_manifest` from each selection result, re-decodes those source frames,
and asks `ObserverAgent` for strict event JSON in batches of at most 16 frames.
Selected frames from one video are packed in strict timestamp order. They may
be sparse and retain their original segment IDs; prompts explicitly prohibit
inventing events inside unobserved gaps.
Validated events are then compiled deterministically into report text; there
is no second free-form summary call. Every report records selected/candidate
coverage and explicitly states that omitted regions are unknown rather than
negative evidence.

Use a new output directory. No existing selection result is modified:

```bash
cd /media/data6/xuejj/ToolCVR
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
/media/data2/xuejj/anaconda3/envs/qwen3/bin/python \
scripts/generate_reports_from_selection.py \
  --selection-dir outputs/crossvid-9task-clip-ratio35 \
  --output-dir outputs/crossvid-9task-observer-reports-v1 \
  --crossvid-root /media/data6/xuejj/CrossVid \
  --tasks PSS,CC,FSA,NC,PI,MSR,MOC,PEA,CCQA \
  --vlm-model /media/data6/xuejj/Qwen-VL \
  --vlm-device cuda:0 \
  --resume
```

`--vlm-adapter` is optional. It is deliberately omitted above because the new
ToolCVR observation protocol should first be tested with the base model and
will later use a newly distilled adapter rather than an old UniCVR adapter.
For a CPU execution-path check without semantic observation, replace the VLM
arguments with `--mock-reporter`. During an incomplete sample, a checkpoint is
written after every visual batch under `TASK/SAMPLE/batches/`; `--resume`
reuses only checkpoints whose selection digest and reporter identity still
match exactly. After the final `report.json` is committed, those intermediate
files are deleted automatically. Pass `--keep-batch-checkpoints` only when
they are needed for debugging.

For a remote multimodal reporter, independent warm-observation batches can be
kept in flight concurrently. `--worker-count` is a **global API request
limit**, not a number of static sample shards: one process owns the pool,
batch checkpoints remain independent, and completed results are merged back
in `(video_id, batch_index)` order. This avoids a long sample leaving the
other API slots idle while preserving deterministic provenance:

```bash
PYTHONPATH=src /media/data2/xuejj/anaconda3/envs/qwen3/bin/python \
scripts/run_parallel_vlm_stage.py reports \
  --worker-count 8 \
  --api-reporter \
  --vlm-api-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --vlm-api-model qwen3.8-flash \
  --vlm-api-key-env LLM_API_KEY \
  --vlm-api-thinking \
  --sample-ids PSS-62,CC-196 \
  --selection-dir outputs/selection \
  --output-dir outputs/api-reports \
  --crossvid-root /media/data6/xuejj/CrossVid
```

All remote model roles use the same OpenAI-compatible transport in
`src/toolcvr/openai_compatible.py`. That module alone owns endpoint
normalization, authentication, response-content extraction, transient HTTP
retry, bounded JSON repair, usage telemetry, and credential redaction. Query
rewriting, visual reporting, initial-state formation, evidence reprojection,
and Reasoner actions retain their own prompts and domain schemas, but they do
not implement separate HTTP clients. This boundary lets the reasoning and
memory mechanisms evolve without silently changing the API wire protocol.

## Initial state formation

This stage replaces the old answer-first `FORM` step. The old step emitted
`best_answer` as its first field, so every later field (`state_factors`,
`alternatives`, `ambiguities`) was derived from a decision the model had
already committed to, and a wrong early answer could not be detected, let
alone corrected.

The replacement consumes the frozen observer reports and produces a state that
structurally cannot contain a decision:

- `observed_information` — what the reports actually show, each item citing
  existing `event_id`s.
- `working_relations` — question-relevant interpreted statements over one or
  more videos, each citing supporting events.
- `unresolved_issues` — what is still unknown, scoped only to the videos that
  need more evidence and phrased so that it is answer-independent.
- `candidate_requests` — a pool of requests for later stages. They are never
  executed here.

Three properties are enforced by the module rather than asked of the model:

- **No answer, no completion.** The serialized state carries
  `contains_answer: false`, `contains_completion_decision: false`, and
  `model_reported_confidence: false`. There is no `best_answer`, no
  `alternatives`, no `DONE`, and no confidence field to fill in.
- **Derived evidence status.** `working_relations[].status` is computed from
  citation structure, not reported by the model: `evidence_complete` when every
  video in scope contributed a cited event, `evidence_partial` when only some
  did, and `unresolved` when an open issue covers the same pair. The derived
  value is a statement about *evidence coverage only* — it never asserts that
  the relation is semantically true, because complete citation coverage does
  not make "v1 happens before v2" correct. A model that declares
  `"status": "evidence_complete"` on a one-sided relation is overruled.
- **Question-driven scope.** The module does not enumerate all `N(N-1)/2`
  video pairs. The reasoner may create a single-video gap, a selected pairwise
  relation, or a multi-video issue according to the evidence need. Mechanical
  pair completion was removed because it manufactured work that the agent had
  not selected and overwhelmed the state on many-video samples.

Mechanical validation drops any item that references an unknown event, video,
or issue and records why. Duplicate event IDs across reports and a video count
below two fail loudly. `render_initial_state` produces the deterministic text
form accepted by `ReasonerAgent`.

```bash
cd /media/data6/xuejj/ToolCVR
PYTHONPATH=src \
/media/data2/xuejj/anaconda3/envs/qwen3/bin/python \
scripts/build_initial_state.py \
  --reports-dir outputs/crossvid-9task-observer-reports-v1 \
  --output-dir outputs/crossvid-9task-initial-state-v1 \
  --mock --resume
```

Drop `--mock` and pass `--api-base-url`/`--api-model` (or set `LLM_BASE_URL` and
`LLM_MODEL`, optionally through `--api-env-file`) to use a language model
instead. The script never decodes video, never loads CLIP, and never generates
an answer. It validates every citation and provenance scope but does not add
relations or issues solely to make an all-pairs table complete.

An explicitly stated limitation of this stage: it does not execute
`candidate_requests`, does not call a second visual model, does not judge
`READY`/`DONE`, and does not route on the task label.

## Install

The existing `qwen3` environment already contains the required runtime
packages. An editable install is convenient but optional:

```bash
cd /media/data6/xuejj/ToolCVR
/media/data2/xuejj/anaconda3/envs/qwen3/bin/pip install -e .
```

Without installing, prefix commands with `PYTHONPATH=src`.

## CPU-only smoke test

The mock encoders and reporter exercise video decoding, cache creation,
  selection, 16-frame windowing, timestamp propagation, schema validation, and report
merging without loading a foundation model:

```bash
cd /media/data6/xuejj/ToolCVR
PYTHONPATH=src /media/data2/xuejj/anaconda3/envs/qwen3/bin/python -m toolcvr.cli warm-start \
  --mock-models \
  --video v1=/path/to/video.mp4 \
  --question "Which tool is used to process the potatoes?" \
  --option "A. A metal press" \
  --option "B. A sieve" \
  --output outputs/smoke.json
```

## Real CLIP with mock event reporter

This isolates and times the new retrieval mechanism before spending VLM
compute. `--clip-model` can be a Hugging Face model ID or a local directory.

```bash
cd /media/data6/xuejj/ToolCVR
PYTHONPATH=src /media/data2/xuejj/anaconda3/envs/qwen3/bin/python -m toolcvr.cli warm-start \
  --clip-model openai/clip-vit-large-patch14-336 \
  --clip-device cuda:0 \
  --mock-reporter \
  --video v1=/media/data6/xuejj/CrossVid/videos/cook/PQxzLMP99BI.mp4 \
  --video v2=/media/data6/xuejj/CrossVid/videos/cook/xiiu9ua4MPE.mp4 \
  --video v3=/media/data6/xuejj/CrossVid/videos/cook/_AZ5jo3y7V4.mp4 \
  --video v4=/media/data6/xuejj/CrossVid/videos/cook/xS78EktJZZA.mp4 \
  --question "Which video uses a different kitchen tool for processing potatoes compared to the others?" \
  --output outputs/cc0-selection.json
```

The first run builds `.cache/clip/*.npz`; later questions over the same videos
reuse those embeddings.

## Real local Qwen3-VL report

```bash
cd /media/data6/xuejj/ToolCVR
PYTHONPATH=src /media/data2/xuejj/anaconda3/envs/qwen3/bin/python -m toolcvr.cli warm-start \
  --clip-model openai/clip-vit-large-patch14-336 \
  --clip-device cuda:0 \
  --vlm-model /media/data6/xuejj/Qwen-VL \
  --vlm-device cuda:0 \
  --video v1=/path/to/video.mp4 \
  --question "How is moisture removed from the grated potatoes?" \
  --output outputs/report.json
```

An optional PEFT adapter can be supplied with `--vlm-adapter`. Each request
header supplies one `video_id` and one `segment_id`; every image is paired
with only its compact frame ID and actual decoded PTS. Full source-frame
provenance remains in the machine-readable manifest instead of bloating the
VLM prompt.

## Query rewriting

The default rewriter is deterministic and conservative: it never invents an
unmentioned object or action. For a stronger rewrite, point the CLI at an
OpenAI-compatible text endpoint:

```bash
--rewrite-base-url http://127.0.0.1:8000/v1 \
--rewrite-model Qwen3-8B \
--rewrite-api-key-env LLM_API_KEY
```

The rewriter may only emit positively observable visual phrases. Counts,
negation, chronology, correctness, and causal judgments remain report hints
for the VLM and are not used as hard CLIP filters.

### CrossVid real-sample test with the DeepSeek Flash API

`scripts/test_real_warm_start.py` reads a CrossVid QA item directly and uses
the `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` settings for query rewriting.
The command below reuses the previous DeepSeek credential file while explicitly
selecting the current `deepseek-flash` model. It prints the compiled retrieval
phrases before loading either visual model. `--video-index` is one-based and
may be repeated; omit it to run every video in the sample.

```bash
cd /media/data6/xuejj/ToolCVR
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=src \
/media/data2/xuejj/anaconda3/envs/qwen3/bin/python scripts/test_real_warm_start.py \
  --task CC \
  --sample-id 1 \
  --video-index 3 \
  --api-env-file /media/data6/xuejj/UniCVR-Agent/.env.bak-deepseek \
  --api-model deepseek-flash \
  --vlm-adapter /media/data6/xuejj/UniCVR-Agent/outputs/sft-qwen3vl-event-interface-v3-r16-e2/adapter \
  --output outputs/cc-1-v3-real-warm-start.json
```

Add `--rewrite-only` to inspect the Qwen-generated phrases without loading
CLIP or Qwen3-VL.

### Nine-task CrossVid CLIP false-negative audit

The batch audit supports `PSS,CC,FSA,NC,PI,MSR,MOC,PEA,CCQA`. Task-specific
storage details are normalized only in the data adapter; every normalized
sample then uses the same rewrite, CLIP selector, cache, and fallback rule.
An empty rewrite or unusable query signal switches to semantic-novelty
coverage: CLIP feature changes and a maximum temporal gap choose anchors
without pretending that irrelevant text is a valid filter. The audit exports
both selected-only sheets and full timeline sheets with green `SELECTED` and
gray `DROPPED` frames.

```bash
cd /media/data6/xuejj/ToolCVR
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  /media/data2/xuejj/anaconda3/envs/qwen3/bin/python \
  scripts/audit_crossvid_clip_filter.py \
  --tasks PSS,CC,FSA,NC,PI,MSR,MOC,PEA,CCQA \
  --samples-per-task 8 \
  --crossvid-root /media/data6/xuejj/CrossVid \
  --api-env-file /media/data6/xuejj/UniCVR-Agent/.env.bak-deepseek \
  --api-model deepseek-flash \
  --clip-model openai/clip-vit-large-patch14-336 \
  --clip-device cuda:0 \
  --clip-batch-size 64 \
  --output-dir outputs/crossvid-9task-clip-audit-v1 \
  --proxy-dir .cache/crossvid-audit-proxies \
  --resume
```

`summary.json` records total and per-video retention. Each sample directory
contains its frozen rewrite, selected sheets, complete timeline sheets, and
machine-readable result. MOC/MSR proxies keep every source frame and overlay
only the five question aliases; interval tasks retain original source PTS.

To compare high-recall ranked-retention operating points, pass
`--selection-profiles speed,balanced,recall`. These retain 50%, 65%, or 80% of
each video's candidates ranked by the strongest raw CLIP cosine. An empty
query plan uses semantic-novelty coverage instead. The profiles run in
one process, reuse one frozen rewrite per sample, and are written below the
corresponding `speed/`, `balanced/`, and `recall/` subdirectories:

```bash
--selection-profiles speed,balanced,recall \
--output-dir outputs/crossvid-clip-profile-sweep
```

The singular `--selection-profile` and optional `--query-plan-dir` remain
available for backward-compatible one-profile runs.

For an arbitrary operating point, use a numeric fraction instead of a named
profile, for example `--retention-ratio 0.60`. It is mutually exclusive with
`--selection-profile` and `--selection-profiles`.

## Tests

```bash
cd /media/data6/xuejj/ToolCVR
PYTHONPATH=src /media/data2/xuejj/anaconda3/envs/qwen3/bin/pytest -q
```
