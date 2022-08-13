from __future__ import annotations
import argparse
from dataclasses import dataclass
import sys
import json
from typing import Iterable, TypeVar, Iterator, Tuple, Dict, TypedDict, List
from itertools import chain

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

T = TypeVar("T")


def chunk(inp: Iterator[str]) -> Iterable[Tuple[str, str]]:
    try:
        first = next(inp)
    except StopIteration:
        return
    while True:
        try:
            second = next(inp)
        except StopIteration:
            # yield first
            return
        if second.startswith("\\"):
            yield (first, second)
            try:
                first = next(inp)
            except StopIteration:
                return
        else:
            yield (first, "\\\t0.9\t0.0\n")
            first = second


class Sample(TypedDict):
    labels: List[str]
    data: List[float]


@dataclass
class Label:
    start: float
    end: float
    label: str
    min_freq: float
    max_freq: float

    @staticmethod
    def from_two_lines(first_line: str, second_line: str) -> Label:
        start, end, label, sep, min_freq, max_freq = list(
            chain(first_line.split("\t"), second_line.split("\t"))
        )
        assert sep == "\\"
        return Label(
            float(start), float(end), label.strip(), float(min_freq), float(max_freq)
        )

    def to_lines(self) -> str:
        return f"{self.start}\t{self.end}\t{self.label}\n\\\t{self.min_freq}\t{self.max_freq}"


def pa() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sound", type=str, required=True)
    parser.add_argument("--input-labels", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = pa()
    labels = []
    with open(args.input_labels) as f:
        for first_line, second_line in chunk(f):
            labels.append(Label.from_two_lines(first_line, second_line))
    print(labels)
    data, sr = librosa.load(args.input_sound)
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram)
    seconds = data.shape[0] / sr
    frames_per_second = spectrogram.shape[1] / seconds
    print(seconds // 60, seconds - 60 * (seconds // 60), frames_per_second)
    samples: Dict[int, Sample] = {}

    def time_to_frame(time: float) -> int:
        return int(time * frames_per_second)

    for label in labels:
        start_frame = time_to_frame(label.start)
        end_frame = time_to_frame(label.end) + 1
        print(label.label, label.start, label.end, start_frame, end_frame)
        for frame in range(start_frame, end_frame):
            if frame not in samples:
                samples[frame] = {"labels": [], "data": []}
            if label.label not in samples[frame]["labels"]:
                samples[frame]["labels"].append(label.label)
            samples[frame]["data"] = [float(x) for x in spectrogram[:, frame]]
    with open(args.output_path, "w") as f:
        print(json.dumps(samples, indent=4), file=f)


if __name__ == "__main__":
    main()
