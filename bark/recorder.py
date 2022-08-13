import argparse
import contextlib
from collections import Counter
import time
import pickle
from typing import (
    Iterable,
    NewType,
    List,
    Tuple,
    Optional,
    Generic,
    TypeVar,
    Sequence,
    Dict,
    Any,
    Type,
    Counter as TCounter,
    ContextManager,
    cast,
)
from dataclasses import dataclass

import librosa
import numpy as np
import pyaudio
import wave

from bark.predictor import chunk

Time = NewType("Time", float)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Timed(Generic[T]):
    start: Time
    duration: Time
    payload: T

    @property
    def end(self) -> Time:
        return Time(self.start + self.duration)


@contextlib.contextmanager
def record_stream(
    channels: int, fs: int, chunk: int
) -> ContextManager[Tuple[pyaudio.PyAudio, pyaudio.Stream, Time]]:
    # Code to acquire resource, e.g.:
    p = pyaudio.PyAudio()
    current_time = time.time_ns() / 1_000_000_000
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=fs,
        frames_per_buffer=chunk,
        input=True,
    )
    try:
        yield (p, stream, Time(current_time))
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def timed_stream(
    start: float, stream: pyaudio.Stream, chunk_size: int, fs: int
) -> Iterable[Timed[bytes]]:
    i = 0
    duration = Time(chunk_size // fs)
    while True:
        # for _ in range(100):
        time_shift = i * chunk_size * 1_000_000_000
        current_time = Time(start + (time_shift / fs) / 1_000_000_000)
        data = stream.read(chunk_size)
        yield Timed(current_time, duration, data)
        if time_shift % fs == 0:
            start = Time(start + (time_shift // fs) / 1_000_000_000)
            i = 0
        i += 1


def batch_by_time(length: Time, stream: Iterable[Timed[T]]) -> Iterable[List[Timed[T]]]:
    # TODO: add overlaps
    out = []
    time_in = 0.0
    for data in stream:
        time_in += data.duration
        out.append(data)
        if time_in >= length:
            yield out
            time_in = 0
            out = []
    if out:
        yield out


def mel_spectrogram(block: Sequence[Timed[bytes]], fs: int) -> np.ndarray:
    data = np.concatenate([np.frombuffer(x.payload, dtype=np.float32) for x in block])
    spectrogram = librosa.power_to_db(
        librosa.feature.melspectrogram(y=data, sr=fs, center=True)
    ).T
    return cast(np.ndarray, spectrogram)


def load_model(model_version: str) -> Dict[str, Any]:
    with open(f"models/2021-01-29.lr.{model_version}.pickle", "rb") as fb:
        models = pickle.load(fb)
    return models  # type: ignore


def raw_predict(
    models: Dict[str, Any], spectrogram: np.ndarray, duration: Time
) -> Dict[str, List[Timed[str]]]:
    out = {}
    ns_per_frame: float = duration / (spectrogram.shape[0] - 1)

    def frame_to_seconds(frame: int) -> float:
        return ns_per_frame * frame

    for t_label, model in models.items():
        predictions = model.predict(spectrogram)
        preds = []
        for frame_start, frame_end in chunk(
            iter(
                np.nonzero(
                    np.ediff1d(predictions.astype(dtype=np.int32), to_begin=0, to_end=1)
                )[0]
            )
        ):
            start = frame_to_seconds(frame_start)
            end = frame_to_seconds(frame_end)
            if end - start < 0.1:
                continue
            preds.append(Timed(Time(start), Time(end - start), t_label))
        filtered_preds: List[Timed[str]] = []
        for label in preds:
            if filtered_preds and label.start - filtered_preds[-1].end < 0.5:
                filtered_preds[-1].duration = Time(
                    label.start + label.duration - filtered_preds[-1].start
                )
            else:
                filtered_preds.append(label)
        out[t_label] = filtered_preds
    return out


@dataclass
class WithPredictions:
    data: bytes
    predictions: Dict[str, List[Timed[str]]]
    noise_levels: List[Timed[float]]


def slice_by_duration(
    being_sliced: Iterable[Timed[T]], slice_by: Iterable[Timed[R]]
) -> Iterable[List[Timed[T]]]:
    pass


# TODO
#
# Return to normal stream
# Write predictions into some DB -- greppable
# Generate notification if dog barks -- ideally
# Save samples around predictions and high noise levels (with some buffers in seconds)
# Migrate training scripts and stuff into this


def run_predictions(
    models: Dict[str, Any], fs: int, stream: Iterable[List[Timed[bytes]]]
) -> Iterable[List[Timed[WithPredictions]]]:

    noise_level_counter: TCounter[float] = Counter()
    for x in stream:
        duration = Time(sum((xx.duration for xx in x), 0.0))
        ns_per_frame: float = duration / (spectrogram.shape[0] - 1)

        def frame_to_seconds(frame: int) -> float:
            return ns_per_frame * frame

        spectrogram = mel_spectrogram(x, fs)
        predictions = raw_predict(models, spectrogram, duration)
        have_prediction = False
        for p, preds in predictions.items():
            if preds:
                print(p, preds)
                have_prediction = True

        noise_level = 10 * np.log10(np.sum(np.power(10, spectrogram / 10), axis=1))
        if not have_prediction:
            noise_level_counter.update(np.round(noise_level, 1))
        total = sum(noise_level_counter.values())
        tot = 0
        res: Optional[float] = None
        for k, v in sorted(noise_level_counter.items()):
            tot += v
            if tot >= total * 0.95:
                res = k
                break

        noises: List[Timed[float]] = []
        for i, noise in enumerate(noise_level[:-1]):
            noises.append(
                Timed(Time(frame_to_seconds(i)), Time(ns_per_frame), float(noise))
            )

        print(
            spectrogram.shape,
            noise_level.shape,
            np.mean(noise_level),
            np.median(noise_level),
            res,
            sum(noise_level > res),
            len(noise_level),
        )


def main() -> None:
    chunk = 1024 * 4  # Record in chunks of 1024 samples
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 3
    filename = "output.wav"
    models = load_model("v3")
    with record_stream(channels=channels, fs=fs, chunk=chunk) as (
        p,
        stream,
        start_time,
    ):
        sample_size = p.get_sample_size(pyaudio.paFloat32)
        run_predictions(
            models,
            fs,
            batch_by_time(
                Time(1_000_000_000), timed_stream(start_time, stream, chunk, fs)
            ),
        )
        # for d in x:
        #    # print(t / 1_000_000_000, type(d), len(d), (time.time_ns() - t)/1_000_000_000)
        #    np_data = np.(d.payload, dtype=np.float32)
        #    spectrogram_raw = librosa.feature.melspectrogram(y=np_data, sr=fs, center=True)
        #    spectrogram = librosa.power_to_db(spectrogram_raw)
        #    print(d.start, d.duration, np_data.shape, spectrogram_raw.shape, spectrogram.shape)
        #    frames.append(d.payload)

    # wf = wave.open(filename, "wb")
    # wf.setnchannels(channels)
    # wf.setsampwidth(sample_size)
    # wf.setframerate(fs)
    # wf.writeframes(b"".join(frames))
    # wf.close()


if __name__ == "__main__":
    main()
