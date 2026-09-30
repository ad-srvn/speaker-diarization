"""RTTM I/O and dependency-light diarization scoring utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Segment:
    recording_id: str
    start: float
    end: float
    speaker: str

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid segment: {self}")


@dataclass
class DiarizationScore:
    reference_speaker_time: float = 0.0
    missed_speech: float = 0.0
    false_alarm: float = 0.0
    speaker_confusion: float = 0.0

    @property
    def error_time(self) -> float:
        return self.missed_speech + self.false_alarm + self.speaker_confusion

    @property
    def der(self) -> float:
        if self.reference_speaker_time == 0:
            return float("nan")
        return self.error_time / self.reference_speaker_time

    def add(self, other: "DiarizationScore") -> None:
        self.reference_speaker_time += other.reference_speaker_time
        self.missed_speech += other.missed_speech
        self.false_alarm += other.false_alarm
        self.speaker_confusion += other.speaker_confusion

    def to_dict(self) -> dict:
        result = asdict(self)
        result["error_time"] = self.error_time
        result["der"] = self.der
        return result


def read_rttm(path: str | Path) -> list[Segment]:
    segments = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 9 or fields[0].upper() != "SPEAKER":
                raise ValueError(f"{path}:{line_number}: invalid RTTM SPEAKER line")
            start = float(fields[3])
            duration = float(fields[4])
            segments.append(Segment(fields[1], start, start + duration, fields[7]))
    return segments


def write_rttm(segments: Iterable[Segment], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for segment in sorted(segments, key=lambda item: (item.recording_id, item.start, item.end)):
            duration = segment.end - segment.start
            handle.write(
                f"SPEAKER {segment.recording_id} 1 {segment.start:.3f} {duration:.3f} "
                f"<NA> <NA> {segment.speaker} <NA> <NA>\n"
            )


def _active_speakers(segments: list[Segment], start: float, end: float) -> set[str]:
    return {segment.speaker for segment in segments if segment.start < end and segment.end > start}


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def score_diarization(
    reference: list[Segment],
    hypothesis: list[Segment],
    evaluation_end: float,
    collar: float = 0.25,
    ignore_overlap: bool = False,
) -> DiarizationScore:
    """Compute DER with an optimal per-recording speaker mapping.

    The implementation uses exact segment boundaries rather than sampled frames.
    A collar is excluded on both sides of every reference boundary. Overlapping
    reference speech is scored by default, matching the VoxConverse protocol.
    """
    if evaluation_end <= 0:
        raise ValueError("evaluation_end must be positive")
    if collar < 0:
        raise ValueError("collar must be non-negative")

    recording_ids = {segment.recording_id for segment in reference + hypothesis}
    if len(recording_ids) > 1:
        raise ValueError("score_diarization accepts one recording at a time")

    excluded = _merge_intervals([
        (max(0.0, boundary - collar), min(evaluation_end, boundary + collar))
        for segment in reference
        for boundary in (segment.start, segment.end)
    ])
    boundaries = {0.0, float(evaluation_end)}
    for segment in reference + hypothesis:
        boundaries.add(max(0.0, min(evaluation_end, segment.start)))
        boundaries.add(max(0.0, min(evaluation_end, segment.end)))
    for start, end in excluded:
        boundaries.update((start, end))
    boundaries = sorted(boundaries)

    intervals = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        if any(ex_start <= midpoint < ex_end for ex_start, ex_end in excluded):
            continue
        ref_active = _active_speakers(reference, start, end)
        hyp_active = _active_speakers(hypothesis, start, end)
        if ignore_overlap and len(ref_active) > 1:
            continue
        intervals.append((start, end, ref_active, hyp_active))

    ref_speakers = sorted({speaker for _, _, active, _ in intervals for speaker in active})
    hyp_speakers = sorted({speaker for _, _, _, active in intervals for speaker in active})
    overlap = np.zeros((len(ref_speakers), len(hyp_speakers)), dtype=np.float64)
    ref_index = {speaker: index for index, speaker in enumerate(ref_speakers)}
    hyp_index = {speaker: index for index, speaker in enumerate(hyp_speakers)}
    for start, end, ref_active, hyp_active in intervals:
        duration = end - start
        for ref_speaker in ref_active:
            for hyp_speaker in hyp_active:
                overlap[ref_index[ref_speaker], hyp_index[hyp_speaker]] += duration

    mapping = {}
    if overlap.size:
        ref_rows, hyp_columns = linear_sum_assignment(-overlap)
        mapping = {hyp_speakers[column]: ref_speakers[row] for row, column in zip(ref_rows, hyp_columns)}

    score = DiarizationScore()
    for start, end, ref_active, hyp_active in intervals:
        duration = end - start
        ref_count = len(ref_active)
        hyp_count = len(hyp_active)
        correct = sum(1 for hyp_speaker in hyp_active if mapping.get(hyp_speaker) in ref_active)
        score.reference_speaker_time += duration * ref_count
        score.missed_speech += duration * max(ref_count - hyp_count, 0)
        score.false_alarm += duration * max(hyp_count - ref_count, 0)
        score.speaker_confusion += duration * (min(ref_count, hyp_count) - correct)
    return score
