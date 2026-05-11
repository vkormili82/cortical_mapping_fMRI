import pandas as pd
from werpy import wer

import onnx_asr
import whisper
import vosk

import soundfile as sf


''' Класс для работы с распознаванием '''
class Recognition:
    def __init__(self):
        self.gigaam = onnx_asr.load_model('gigaam-v3-e2e-rnnt')
        #self.vosk = vosk.Model('vosk-model-ru-0.42')
        #self.whisper = whisper.load_model("small")
        #self.nvidia = onnx_asr.load_model('nvidia/stt_ru_fastconformer_hybrid_large_pc')


    def transcription(self, audio):

        hypothesis = self.gigaam.recognize(audio, sample_rate=16000)
        words = hypothesis.split()
        print(hypothesis)

        line = {
            'word1': words[0] if len(words) > 0 else None,
            'word2': words[1] if len(words) > 1 else None,
            'word3': words[2] if len(words) > 2 else None,
            'word4': words[3] if len(words) > 3 else None,
            'total time': None,
            'start time <>': None,
            'finish time <>': None
        }

        return line




''' Класс для работы с классификацией '''
class Analysis:
    def __init__(self, words):
        self.words = words