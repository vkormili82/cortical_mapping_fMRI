import os

from preprocessing import convert, spectrogram, Splitter
from speech import Recognition, Analysis
import pandas as pd


if __name__ == "__main__":
    file_wav = convert('4_PYu.m4a')
    tomograph = 'tomograph_2.wav'

    #wav, sr = librosa.load(file_wav)
    #spectrogram(wav)

    Splitter.process(Splitter(file_wav))
    df = pd.DataFrame(
        columns=['number',
                 'word1',
                 'word2',
                 'word3',
                 'word4',
                 'total time',
                 'start time <>',  # Не знаю, как это назвать, кроме как "ответ на стимул"
                 'finish time <>'])
    i = 1
    rec = Recognition()

    while os.path.isfile(f'C:/Users/Admin/PycharmProjects/corticalmapping_fMRI/segments/segment_{i}.wav'):
        line = rec.result(
            f'C:/Users/Admin/PycharmProjects/corticalmapping_fMRI/segments/segment_{i}.wav')
        print(line)
        #df = pd.concat([df, pd.DataFrame([line])], ignore_index=True)