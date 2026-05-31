import os
import pandas as pd
import librosa

from preprocessing import convert, Splitter
from speech import Recognition, Analysis


def trim_silence_and_get_duration(file_path, top_db=20):
    y, sr = librosa.load(file_path, sr=None)
    return len(librosa.effects.trim(y, top_db=top_db)) / sr


if __name__ == '__main__':
    audio = '4_PYu.m4a'
    tomograph = 'tomograph_2.wav'

    examples = pd.read_csv('example.csv', sep=';', encoding='cp1251')

    rater = 'program'
    file_wav = convert(audio)

    splitter = Splitter(file_wav, tomograph)
    result = splitter.process()
    segments = 'segments'

    rec = Recognition()
    rows = []

    i = 1
    while os.path.isfile(f'{segments}/segment_{i}.wav'):
        path_tosegment = f'{segments}/segment_{i}.wav'

        if i >= len(examples):
            print(f'Warning: No more rows in CSV for segment {i}')
            break

        pattern = examples['sentence'].iloc[i]
        line = rec.result(path_tosegment)

        words = line.split()
        sentence = ' '.join(words[:3]) if len(words) >= 3 else line
        response = words[3] if len(words) >= 4 else ''

        analyser = Analysis(line, pattern, type='real')
        res = analyser.process()
        if res is not None:
            flag, flag_phon, flag_gram, flag_sem, cos_sim = res
        else:
            flag = flag_phon = flag_gram = flag_sem = False

        accuracy = (flag_phon and flag_gram and flag_sem)

        errortypes = []

        if not flag_phon: errortypes.append('phon')
        if not flag_gram: errortypes.append('gram')
        if not flag_sem: errortypes.append('sem')
        errortype = ','.join(errortypes) if errortypes else ''

        duration = trim_silence_and_get_duration(path_tosegment)
        time_ok = duration <= 7.0
        patient_id = os.path.splitext(os.path.basename(audio))[0]

        row = {
            'Block': examples['Block'].iloc[i],
            'Trial': examples['Trial'].iloc[i],
            'cond': examples['cond'].iloc[i],
            'sentence': sentence,
            'response': response,
            'accuracy': accuracy,
            'errortype': errortype,
            'gram': flag_gram,
            'sem': flag_sem,
            'phon': flag_phon,
            'time': time_ok,
            'comment': '',
            'rater': rater,
            'patient_ID': patient_id,
        }
        
        rows.append(row)
        i += 1


    df = pd.DataFrame(rows)
    df.to_csv('result.csv', index=False, encoding='utf-8')
    print('См. результат в таблице result.csv')