from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from toolcvr.schemas import VideoInput


SUPPORTED_CROSSVID_TASKS = ("PSS", "CC", "FSA", "NC", "PI", "MSR", "MOC", "PEA", "CCQA")
AnswerKind = Literal["choice", "sequence", "interval", "free_text"]


def crossvid_sample_id(
    task: str,
    row: dict[str, object],
    source_index: int,
) -> str:
    """Identify a row without decoding or validating its media payload."""

    normalized_task = task.strip().upper()
    if normalized_task not in SUPPORTED_CROSSVID_TASKS:
        raise ValueError(f"unsupported CrossVid task: {normalized_task}")
    return f"{normalized_task}-{row.get('id', source_index)}"


@dataclass(frozen=True)
class CrossVidAuditSample:
    sample_id: str
    task: str
    question: str
    options: tuple[str, ...]
    answer: object
    videos: tuple[VideoInput, ...]
    adaptation_notes: tuple[str, ...] = ()
    answer_kind: AnswerKind = "free_text"
    required_answer_ids: tuple[str, ...] = ()


class CrossVidAuditAdapter:
    """Normalize all CrossVid formats before the shared selector sees them.

    Branches in this class only decode dataset storage conventions.  The
    returned samples all follow the same VideoInput/question/options protocol.
    """

    def __init__(self, root: Path, *, proxy_directory: Path) -> None:
        self.root = root.expanduser().resolve()
        self.video_root = self.root / "videos"
        self.uav_root = self.root / "uav"
        self.uav_materializer = UavAuditMaterializer(proxy_directory)

    def normalize(self, task: str, row: dict[str, object], source_index: int) -> CrossVidAuditSample:
        normalized_task = task.upper()
        sample_id = crossvid_sample_id(normalized_task, row, source_index)
        adaptation_notes: list[str] = []
        if normalized_task in {"CC", "NC"}:
            answer_kind: AnswerKind = "choice"
            question = _required_string(row, "question")
            options = _string_tuple(row.get("options"), "options")
            videos = self._plain_videos(_string_tuple(row.get("videos"), "videos"))
        elif normalized_task == "CCQA":
            answer_kind = "free_text"
            question = _required_string(row, "question")
            options = ()
            videos = self._plain_videos(
                (_required_string(row, "video A"), _required_string(row, "video B"))
            )
            adaptation_notes.extend((
                "v1 is Video A and v2 is Video B; preserve this attribution in the comparative answer.",
                "The final answer is free text, not an option label.",
            ))
        elif normalized_task == "PEA":
            answer_kind = "choice"
            question = _required_string(row, "question")
            options = _string_tuple(row.get("options"), "options")
            relative_paths = _string_tuple(row.get("videos"), "videos")
            begins = _number_tuple(row.get("begin"), "begin")
            ends = _number_tuple(row.get("end"), "end")
            if not (len(relative_paths) == len(begins) == len(ends)):
                raise ValueError("PEA videos/begin/end lengths differ")
            pea_inputs: list[VideoInput] = []
            for position, (relative, begin, end) in enumerate(
                zip(relative_paths, begins, ends), start=1
            ):
                path, note = self._resolve_annotated_interval_video(
                    relative,
                    begin=begin,
                    end=end,
                )
                if note:
                    adaptation_notes.append(f"v{position}: {note}")
                pea_inputs.append(
                    VideoInput(
                        video_id=f"v{position}",
                        path=path,
                        time_spans=((begin, end),),
                    )
                )
            videos = tuple(pea_inputs)
        elif normalized_task == "FSA":
            answer_kind = "interval"
            question = (
                "Video 1 is a reference cooking step. Find the functionally equivalent step in "
                "Video 2. Return only its Video 2 interval as [start_seconds, end_seconds]."
            )
            options = ()
            reference = _interval(row.get("ref_segment"), "ref_segment")
            videos = (
                VideoInput("v1", self._video_path(_required_string(row, "video A")), (reference,)),
                VideoInput("v2", self._video_path(_required_string(row, "video B"))),
            )
            adaptation_notes.extend((
                "v1 is the fixed reference segment and v2 is the search video; their clocks are independent.",
                "Match functional step and state transition before refining boundaries; local visual resemblance alone is insufficient.",
                "The final answer is exactly one ordered [start_seconds, end_seconds] interval in v2 local seconds.",
            ))
        elif normalized_task == "PI":
            answer_kind = "choice"
            question = (
                "Video 1 shows the beginning and Video 2 shows the ending of one story. "
                "Which option best explains what happened in the missing interval?"
            )
            options = _string_tuple(row.get("options"), "options")
            source = self._video_path(_required_string(row, "video"))
            videos = (
                VideoInput("v1", source, (_interval(row.get("beginning"), "beginning"),)),
                VideoInput("v2", source, (_interval(row.get("ending"), "ending"),)),
            )
            adaptation_notes.append(
                "v1 is the beginning and v2 is the ending of one story; infer the missing event from visible before/after state changes rather than treating them as synchronized views."
            )
        elif normalized_task == "PSS":
            answer_kind = "sequence"
            question = (
                "These numbered clips are shuffled segments from one cooking video. Determine "
                "their chronological order. Return only clip numbers joined by ->."
            )
            options = ()
            source = self._video_path(_required_string(row, "video"))
            raw_segments = row.get("segments")
            if not isinstance(raw_segments, dict) or len(raw_segments) < 2:
                raise ValueError("PSS segments must contain at least two numbered clips")
            inputs: list[VideoInput] = []
            for key in sorted(raw_segments, key=lambda value: int(value)):
                raw_intervals = raw_segments[key]
                if isinstance(raw_intervals, dict):
                    raw_intervals = raw_intervals.get("interval")
                if not isinstance(raw_intervals, list) or not raw_intervals:
                    raise ValueError(f"PSS segment {key} has no intervals")
                intervals = tuple(_interval(value, f"segments.{key}") for value in raw_intervals)
                # PSS clips are logical inputs, not windows on a clock the
                # model is allowed to inspect.  Decode through the original
                # source intervals, but expose a clip-local clock beginning
                # at zero.  Otherwise absolute source offsets solve the
                # benchmark without visual temporal reasoning.
                inputs.append(VideoInput(
                    f"v{key}", source, intervals, model_clock="local",
                ))
            videos = tuple(inputs)
            adaptation_notes.extend((
                "The inputs are shuffled segments from one source video; input numbering and rendered slot order are not chronological evidence.",
                "Infer order from visible state transitions. A visual call reports local states only and must not decide cross-segment chronology.",
                "The final sequence contains each input video ID exactly once; slot IDs and evidence IDs are never answer items.",
            ))
        elif normalized_task in {"MOC", "MSR"}:
            answer_kind = "choice"
            question = _required_string(row, "question")
            options = _string_tuple(row.get("options"), "options")
            scene_id = row.get("vid")
            objects = row.get("objects")
            if not isinstance(scene_id, int):
                raise ValueError("UAV vid must be an integer")
            if not isinstance(objects, list) or not all(isinstance(item, dict) for item in objects):
                raise ValueError("UAV objects must be an object list")
            paths = self.uav_materializer.materialize(
                uav_root=self.uav_root,
                scene_id=scene_id,
                objects=objects,
                cache_key=sample_id,
            )
            videos = tuple(
                VideoInput(
                    f"v{index}",
                    path,
                    scan_stride_seconds=2.0,
                    synchronization_group=f"uav-scene-{scene_id}",
                )
                for index, path in enumerate(paths, 1)
            )
            adaptation_notes.extend((
                "The synchronized UAV views use one source-frame clock; compare equal timestamps across views.",
                "Overlay aliases with the same numeric suffix (for example A1 and B1) denote the same annotated physical track in different views.",
                "A transition seen in both synchronized views is one physical transition, not two countable events.",
                "In this UAV dataset vocabulary, cycles means two-wheeled road vehicles such as bicycles, e-bikes, scooters, motorcycles, and mopeds; count vehicles rather than riders.",
                "A conditional trigger such as when/after/once establishes the synchronized target timestamp; evaluate the requested relation at that time rather than applying the trigger predicate to counted targets.",
            ))
            adaptation_notes.extend(self.uav_materializer.reference_track_notes(
                uav_root=self.uav_root,
                scene_id=scene_id,
                objects=objects,
            ))
        else:
            raise ValueError(f"unsupported CrossVid task: {normalized_task}")
        return CrossVidAuditSample(
            sample_id=sample_id,
            task=normalized_task,
            question=question,
            options=options,
            answer=row.get("answer"),
            videos=videos,
            adaptation_notes=tuple(adaptation_notes),
            answer_kind=answer_kind,
            required_answer_ids=(
                tuple(video.video_id for video in videos)
                if answer_kind == "sequence"
                else ()
            ),
        )

    def _plain_videos(self, relative_paths: Sequence[str]) -> tuple[VideoInput, ...]:
        if not relative_paths:
            raise ValueError("sample has no videos")
        return tuple(
            VideoInput(f"v{position}", self._video_path(relative))
            for position, relative in enumerate(relative_paths, 1)
        )

    def _video_path(self, relative: str) -> Path:
        path = (self.video_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CrossVid video does not exist: {path}")
        return path

    def _resolve_annotated_interval_video(
        self,
        relative: str,
        *,
        begin: float,
        end: float,
    ) -> tuple[Path, str | None]:
        original = self._video_path(relative)
        duration = _video_duration(original)
        if begin < duration and end <= duration + 0.1:
            return original, None

        # CrossVid PEA contains a small known class of rows whose interval was
        # authored on the RGB view but whose path names a truncated HMC view
        # from the same Assembly101 recording. Repair only an exact recording
        # prefix with one unique RGB file that covers the complete interval.
        marker = "_HMC_"
        if marker in original.name:
            recording_prefix = original.name.split(marker, 1)[0]
            covering = [
                path
                for path in original.parent.glob(f"{recording_prefix}_*_rgb.mp4")
                if _video_duration(path) + 0.1 >= end
            ]
            if len(covering) == 1:
                replacement = covering[0].resolve()
                note = (
                    "replaced interval-incompatible HMC asset "
                    f"({duration:.3f}s for [{begin:.3f},{end:.3f}]s) with the unique "
                    f"same-recording RGB asset ({_video_duration(replacement):.3f}s)"
                )
                return replacement, note
        raise ValueError(
            f"annotated interval [{begin:.3f},{end:.3f}]s lies outside {original} "
            f"(duration={duration:.3f}s), and no unique same-recording asset covers it"
        )


class UavAuditMaterializer:
    """Render both UAV views while preserving the source temporal scale.

    The UAV asset is an ordered image sequence.  It is encoded at its source
    frame rate, rather than at an artificial audit rate: otherwise a
    time-based sparse sampler would mistakenly inspect every source image and
    report timestamps that are tens of times too large.
    """

    def __init__(self, directory: Path, *, long_edge: int = 960, source_fps: float = 30.0) -> None:
        if long_edge <= 0 or source_fps <= 0:
            raise ValueError("UAV proxy long edge and source fps must be positive")
        self.directory = directory.expanduser().resolve()
        self.long_edge = long_edge
        self.source_fps = source_fps

    def materialize(
        self,
        *,
        uav_root: Path,
        scene_id: int,
        objects: Sequence[dict[str, object]],
        cache_key: str,
    ) -> tuple[Path, Path]:
        first = self._view(uav_root, scene_id, 1, objects, cache_key)
        second = self._view(uav_root, scene_id, 2, objects, cache_key)
        return first, second

    def reference_track_notes(
        self,
        *,
        uav_root: Path,
        scene_id: int,
        objects: Sequence[dict[str, object]],
    ) -> tuple[str, ...]:
        """Render the bounded reference sidecar used by UniCVR's UAV plugins.

        Only the tracks explicitly listed in the question row are exposed.
        This cannot leak unlabelled count targets, but it makes alias identity
        and trigger-time visibility deterministic for both synchronized views.
        """
        sources: dict[int, tuple[int, dict[int, dict[int, tuple[int, int, int, int]]]]] = {}
        for view in (1, 2):
            frame_directory = uav_root / "frames" / str(view) / f"{scene_id}-{view}"
            bbox_path = uav_root / "bbox" / str(view) / f"{scene_id}.json"
            image_count = len([
                path for path in frame_directory.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])
            if image_count <= 0:
                raise ValueError(f"UAV frame directory is empty: {frame_directory}")
            sources[view] = (image_count, _load_uav_boxes(bbox_path))

        notes = [
            "The following reference-track sidecar covers only annotated aliases already drawn in the proxy videos; it is not a list of countable targets."
        ]
        for position, item in enumerate(objects, start=1):
            object_id = item.get("id")
            if not isinstance(object_id, int):
                continue
            pieces = [
                f"A{position}/B{position} is source_track={object_id}",
                f"declared={item.get('label', 'object')}",
            ]
            for view, prefix in ((1, "A"), (2, "B")):
                frame_count, boxes = sources[view]
                ranges = _contiguous_ranges(boxes.get(object_id, {}))
                rendered = ",".join(
                    f"{start}-{end} ({start / self.source_fps:.3f}-{(end + 1) / self.source_fps:.3f}s)"
                    for start, end in ranges[:3]
                ) or "none"
                pieces.append(f"{prefix}{position}_visible={rendered}/{frame_count}_frames")
            notes.append("; ".join(pieces) + ".")
        notes.append(
            "Reference visible ranges are authoritative for alias presence and trigger timing; they do not determine unlabelled target counts."
        )
        return tuple(notes)

    def _view(
        self,
        uav_root: Path,
        scene_id: int,
        view: int,
        objects: Sequence[dict[str, object]],
        cache_key: str,
    ) -> Path:
        frame_directory = uav_root / "frames" / str(view) / f"{scene_id}-{view}"
        bbox_path = uav_root / "bbox" / str(view) / f"{scene_id}.json"
        if not frame_directory.is_dir() or not bbox_path.is_file():
            raise FileNotFoundError(f"missing UAV assets for scene={scene_id}, view={view}")
        image_paths = sorted(
            path for path in frame_directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_paths:
            raise ValueError(f"UAV frame directory is empty: {frame_directory}")
        identity = {
            "scene_id": scene_id,
            "view": view,
            "objects": objects,
            "frame_count": len(image_paths),
            "first": image_paths[0].name,
            "last": image_paths[-1].name,
            "bbox_size": bbox_path.stat().st_size,
            "bbox_mtime_ns": bbox_path.stat().st_mtime_ns,
            "fps": self.source_fps,
            "long_edge": self.long_edge,
            "cache_key": cache_key,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output = self.directory / digest[:2] / f"{digest}.mp4"
        sidecar = output.with_suffix(".provenance.json")
        if output.is_file() and sidecar.is_file():
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        boxes = _load_uav_boxes(bbox_path)
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise RuntimeError(f"failed to read UAV frame: {image_paths[0]}")
        first = _resize_long_edge(first, self.long_edge)
        height, width = first.shape[:2]
        temporary = output.with_name(f"{output.stem}.{os.getpid()}.tmp.mp4")
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter.fourcc(*"mp4v"), self.source_fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot create UAV proxy: {temporary}")
        prefix = "A" if view == 1 else "B"
        try:
            for frame_index, image_path in enumerate(image_paths):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"failed to read UAV frame: {image_path}")
                _draw_aliases(image, frame_index, objects, boxes, prefix)
                image = _resize_long_edge(image, self.long_edge)
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    image,
                    f"source_frame={image_path.name}",
                    (12, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(image)
        finally:
            writer.release()
        if not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"UAV proxy encoding failed: {output}")
        temporary.replace(output)
        sidecar.write_text(
            json.dumps(
                {
                    "scene_id": scene_id,
                    "view": view,
                    "proxy_fps": self.source_fps,
                    "source_frames": [path.name for path in image_paths],
                    "objects": list(objects),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output
def _load_uav_boxes(path: Path) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"UAV bbox JSON must contain a list: {path}")
    parsed: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for entity in raw:
        if not isinstance(entity, dict) or not isinstance(entity.get("id"), int):
            continue
        frames: dict[int, tuple[int, int, int, int]] = {}
        raw_boxes = entity.get("bbox")
        if isinstance(raw_boxes, dict):
            for key, value in raw_boxes.items():
                if not isinstance(value, dict) or value.get("outside") is True:
                    continue
                coordinates = [value.get(name) for name in ("xtl", "ytl", "xbr", "ybr")]
                if all(isinstance(item, (int, float)) for item in coordinates):
                    frames[int(key)] = (
                        round(float(coordinates[0])),
                        round(float(coordinates[1])),
                        round(float(coordinates[2])),
                        round(float(coordinates[3])),
                    )
        parsed[entity["id"]] = frames
    return parsed


def _contiguous_ranges(
    boxes: dict[int, tuple[int, int, int, int]],
) -> tuple[tuple[int, int], ...]:
    """Collapse visible source-frame indices into inclusive ranges."""
    indices = sorted(boxes)
    if not indices:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = indices[0]
    for frame_index in indices[1:]:
        if frame_index == previous + 1:
            previous = frame_index
            continue
        ranges.append((start, previous))
        start = previous = frame_index
    ranges.append((start, previous))
    return tuple(ranges)


def _draw_aliases(
    image: np.ndarray,
    frame_index: int,
    objects: Sequence[dict[str, object]],
    boxes: dict[int, dict[int, tuple[int, int, int, int]]],
    prefix: str,
) -> None:
    colors = ((0, 255, 255), (255, 128, 0), (0, 255, 0), (255, 0, 255), (0, 128, 255))
    height, width = image.shape[:2]
    for position, item in enumerate(objects, 1):
        object_id = item.get("id")
        if not isinstance(object_id, int):
            continue
        box = boxes.get(object_id, {}).get(frame_index)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        color = colors[(position - 1) % len(colors)]
        label = f"{prefix}{position} {item.get('label', 'object')}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(image, (x1, max(0, y1 - 28)), (min(width - 1, x1 + 180), y1), color, -1)
        cv2.putText(image, label, (x1 + 4, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def _resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= long_edge:
        return image
    scale = long_edge / max(height, width)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"video has invalid fps/frame count: {path}")
    return frame_count / fps


def _required_string(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(value)


def _number_tuple(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field} must be a number list")
    return tuple(float(item) for item in value)


def _interval(value: object, field: str) -> tuple[float, float]:
    numbers = _number_tuple(value, field)
    if len(numbers) != 2 or numbers[0] < 0 or numbers[1] <= numbers[0]:
        raise ValueError(f"{field} must be an ordered [start, end] interval")
    return numbers[0], numbers[1]
