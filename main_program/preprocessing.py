from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import correlate, find_peaks
import os
import shutil


""" Переводим файл в формат wav """


def convert(file):
    file_wav = str(file).rsplit(".", 1)[0] + ".wav"

    audio = AudioSegment.from_file(file)
    audio.export(file_wav, format="wav")

    return file_wav


""" Изображение спектрограммы """


def spectrogram(wav):

    plt.figure(figsize=(12, 5))
    plt.plot(wav)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return 0


""" Извлечение отдельных фрагментов с речью """


class Splitter:
    def __init__(self, audio, tomograph="tomograph_2.wav", segments="segments", correlation_threshold=0.4, min_segment_duration=0.5):

        self.audio = audio
        self.tomograph = tomograph
        self.fragments = segments
        self.correlation_threshold = correlation_threshold
        self.min_segment_duration = min_segment_duration

        self.wav, self.sr = librosa.load(audio, sr=None)
        self.wav_t, self.sr_t = librosa.load(tomograph, sr=self.sr)

        self.timestamps = []
        self.correlations = []
        self.segments = []

        self.i = 0

    def find(self):

        corr = correlate(self.wav, self.wav_t, mode="valid")
        corr = corr / np.max(corr)

        peaks, properties = find_peaks(corr, height=self.correlation_threshold, distance=len(self.wav_t) * 0.5)

        self.timestamps = peaks / self.sr
        self.correlations = properties["peak_heights"] if len(peaks) > 0 else []

        return self.timestamps

    def __new_folder(self):
        if os.path.exists(self.fragments):
            shutil.rmtree(self.fragments)

        os.makedirs(self.fragments, exist_ok=True)

        return 0

    def cut(self):
        duration_t = len(self.wav_t) / self.sr_t

        timestamps_sorted = sorted(self.timestamps)

        intervals = list()

        if len(timestamps_sorted) > 0 and timestamps_sorted[0] > 0:
            intervals.append({"start": 0, "end": timestamps_sorted[0], "type": "before_first"})

        for i in range(len(timestamps_sorted) - 1):
            current_end = timestamps_sorted[i] + duration_t
            next_start = timestamps_sorted[i + 1]

            if next_start > current_end:
                intervals.append({"start": current_end, "end": next_start, "type": "between", "between_indices": (i, i + 1)})

        if len(timestamps_sorted) > 0:
            last_end = timestamps_sorted[-1] + duration_t
            total_duration = len(self.wav) / self.sr
            if last_end < total_duration:
                intervals.append({"start": last_end, "end": total_duration, "type": "after_last"})

        self.segments = []
        segment_count = 0

        for interval in intervals:
            start_time = interval["start"]
            end_time = interval["end"]
            duration = end_time - start_time

            if duration < self.min_segment_duration:
                continue

            segment_count += 1

            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            segment = self.wav[start_sample:end_sample]

            self.i += 1

            segment_name = f"segment_{self.i}.wav"

            segment_path = os.path.join(self.fragments, segment_name)
            sf.write(segment_path, segment, self.sr)

            self.segments.append({"num": segment_count, "type": interval["type"], "path": segment_path, "filename": segment_name, "start_sec": start_time, "end_sec": end_time, "duration": duration})

        return 0

    def process(self):
        timestamps = self.find()

        if len(timestamps) == 0:
            self.__new_folder()
            self.i += 1
            segment_name = f"segment_{self.i}.wav"
            segment_path = os.path.join(self.fragments, segment_name)
            sf.write(segment_path, self.wav, self.sr)
            self.segments.append({"num": 1, "type": "full_file", "path": segment_path, "filename": segment_name, "start_sec": 0, "end_sec": len(self.wav) / self.sr, "duration": len(self.wav) / self.sr})
            return {"timestamps": timestamps, "correlations": [], "segments": self.segments, "total_signals": 0, "total_segments": 1, "output_folder": self.fragments}

        self.__new_folder()
        self.cut()

        return {"timestamps": self.timestamps, "correlations": self.correlations, "segments": self.segments, "total_signals": len(self.timestamps), "total_segments": len(self.segments), "output_folder": self.fragments}
