from __future__ import annotations
import argparse
import json
from collections import Counter
import pickle
from typing import TypeVar, Tuple, List, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from tqdm import tqdm

import numpy as np
import librosa

T = TypeVar("T")


def chunk(inp: Iterator[T]) -> Iterable[Tuple[T, T]]:
    while True:
        try:
            yield (next(inp), next(inp))
        except StopIteration:
            return


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
        return Label(float(start), float(end), label.strip(), float(min_freq), float(max_freq))

    def to_lines(self) -> str:
        return f"{self.start}\t{self.end}\t{self.label}\n\\\t{self.min_freq}\t{self.max_freq}"


def main(model_file: str, file: str, output_version: str) -> None:
    print()
    print(file, model_file, output_version)

    with open(model_file, "rb") as fb:
        models = pickle.load(fb)
    data, sr = librosa.load(f"{file}")
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram)
    seconds = data.shape[0] / sr
    frames_per_second: float = spectrogram.shape[1] / seconds
    print(seconds // 60, seconds - 60 * (seconds // 60), frames_per_second)
    transposed = spectrogram.T

    def frame_to_time(frame: int) -> float:
        return frame / frames_per_second

    for t_label, model in models.items():
        labels = []
        predictions = model.predict(transposed)
        coverage = 0.0
        silent_minute_coverage = 0.0
        count = 0
        for frame_start, frame_end in chunk(
            iter(np.nonzero(np.ediff1d(predictions.astype(dtype=np.int64), to_begin=0, to_end=1))[0],)
        ):
            start = frame_to_time(frame_start)
            end = frame_to_time(frame_end)
            if end - start < 0.1:
                continue
            labels.append(Label(start, end, t_label, 0.0, 0.0))
        prev_end = -1000
        filtered_labels: List[Label] = []
        for label in labels:
            if label.start - prev_end < 0.5:
                filtered_labels[-1].end = label.end
            else:
                filtered_labels.append(label)
        labels = filtered_labels
        prev = 0.0
        for label in labels:
            count += 1
            coverage += label.end - label.start
            diff = (label.start - prev)
            if diff >= 60:
                silent_minute_coverage += label.start - prev
            prev = label.end
        print(f"{t_label}: {coverage * 100 / seconds:.2f}%, {silent_minute_coverage * 100 / seconds:.2f}% {count}, {count * 60 / seconds:.2f} b/m, {seconds / 3600:.2f}h")
        base = file.rsplit(".", maxsplit=1)[0]
        with open(f"{base}.predicted.{output_version}.{label.label}.txt", "w") as f:
            for label in labels:
                print(label.to_lines(), file=f)


def pa() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--files", nargs="+", default=["2021-05-29"], required=True)
    parser.add_argument("--output-version", default="default")
    return parser.parse_args()


if __name__ == "__main__":
    args = pa()
    print(args)
    for file in args.files:
        main(args.model_file, file, args.output_version)
