import pandas as pd
from werpy import wer
import numpy as np
import onnx_asr
import soundfile as sf
from scipy import signal
import noisereduce as nr
import librosa
import re


''' Класс для работы с распознаванием '''
class Recognition:
    def __init__(self):
        self.gigaam = onnx_asr.load_model('gigaam-v3-e2e-rnnt')

    def clean(self, file):
        audio, sr = sf.read(file)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        for i in [50, 100]:
            if i < sr / 2:
                b, a = signal.iirnotch(i / (sr / 2), 35)
                audio = signal.lfilter(b, a, audio)

        audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.8)
        audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9

        return audio.astype(np.float32), sr

    def transcription(self, audio):
        audio_clean, sr = self.clean(audio)

        hypothesis = self.gigaam.recognize(audio, sample_rate=16000)
        words = hypothesis.split()

        return hypothesis

    def result(self, audio):

        line = self.transcription(audio)
        line = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', line).lower().strip()

        return line





''' Класс для работы с классификацией '''
class Analysis:
    def __init__(self, words):
        self.words = words