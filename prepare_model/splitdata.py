import os, shutil
import numpy as np, pandas as pd
import librosa, soundfile as sf
from scipy.signal import correlate, find_peaks
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from tqdm import tqdm
import random
from preprocessing import convert


def augmentation(audio):
    sr = 16000  # фиксированная частота дискретизации
    augmentations = [
        lambda: audio + np.random.normal(0, random.uniform(0.002, 0.01)),
        lambda: librosa.effects.time_stretch(audio, rate=random.uniform(0.85, 1.15)),
        lambda: librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(0.5, 1.5)),
        lambda: librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(-1.5, -0.5)),
        lambda: librosa.effects.time_stretch(audio, rate=random.uniform(1.05, 1.15)),
        lambda: librosa.effects.time_stretch(audio, rate=random.uniform(0.85, 0.95)),
    ]
    return random.choice(augmentations)()


class Splitter:
    def __init__(self, audio_path, tomograph_path='tomograph_2.wav',
                 correlation_threshold=0.4, min_segment_duration=0.5):

        self.audio_path = audio_path
        self.tomograph_path = tomograph_path
        self.fragments = 'segments'
        self.correlation_threshold = correlation_threshold
        self.min_segment_duration = min_segment_duration

        try:
            self.wav, self.sr = librosa.load(audio_path, sr=None)
            self.wav_t, self.sr_t = librosa.load(tomograph_path, sr=self.sr)
        except Exception as e:
            print(f"Ошибка загрузки аудио {audio_path}: {e}")
            self.wav = None
            self.sr = None
            self.wav_t = None
            self.sr_t = None

        self.timestamps = []
        self.correlations = []
        self.segments = []
        self.i = 0

    def find(self):
        if self.wav is None or self.wav_t is None:
            return []
        corr = correlate(self.wav, self.wav_t, mode='valid')
        if len(corr) == 0:
            return []
        corr = corr / np.max(corr)
        peaks, properties = find_peaks(corr, height=self.correlation_threshold,
                                       distance=len(self.wav_t) * 0.5)
        self.timestamps = peaks / self.sr
        self.correlations = properties['peak_heights'] if len(peaks) > 0 else []
        return self.timestamps

    def _new_folder(self):
        if os.path.exists(self.fragments):
            shutil.rmtree(self.fragments)
        os.makedirs(self.fragments, exist_ok=True)

    def cut(self):
        if len(self.timestamps) == 0:
            return

        duration_t = len(self.wav_t) / self.sr_t
        timestamps_sorted = sorted(self.timestamps)

        intervals = []

        if timestamps_sorted[0] > 0:
            intervals.append({
                'start': 0,
                'end': timestamps_sorted[0],
                'type': 'before_first'
            })

        for i in range(len(timestamps_sorted) - 1):
            current_end = timestamps_sorted[i] + duration_t
            next_start = timestamps_sorted[i + 1]
            if next_start > current_end:
                intervals.append({
                    'start': current_end,
                    'end': next_start,
                    'type': 'between',
                    'between_indices': (i, i + 1)
                })

        last_end = timestamps_sorted[-1] + duration_t
        total_duration = len(self.wav) / self.sr
        if last_end < total_duration:
            intervals.append({
                'start': last_end,
                'end': total_duration,
                'type': 'after_last'
            })

        self.segments = []
        self.i = 0

        for interval in intervals:
            start_time = interval['start']
            end_time = interval['end']
            duration = end_time - start_time

            if duration < self.min_segment_duration:
                continue

            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            segment = self.wav[start_sample:end_sample]

            self.i += 1
            segment_name = f'segment_{self.i}.wav'
            segment_path = os.path.join(self.fragments, segment_name)
            sf.write(segment_path, segment, self.sr)

            self.segments.append({
                'num': self.i,
                'type': interval['type'],
                'path': segment_path,
                'filename': segment_name,
                'start_sec': start_time,
                'end_sec': end_time,
                'duration': duration
            })

    def process(self):
        self.find()
        if len(self.timestamps) == 0:
            return None

        self._new_folder()
        self.cut()

        return {
            'timestamps': self.timestamps,
            'correlations': self.correlations,
            'segments': self.segments,
            'total_signals': len(self.timestamps),
            'total_segments': len(self.segments),
            'output_folder': self.fragments
        }


class AudioDatasetBuilder:
    def __init__(self, audio_folder, tomograph_path='tomograph_2.wav'):
        self.audio_folder = Path(audio_folder)
        self.tomograph_path = tomograph_path
        self.all_segments = []
        self.files_info = []

    def process_all_files(self):
        audio_files = []
        for ext in ['*.m4a', '*.wav', '*.mp3']:
            audio_files.extend(list(self.audio_folder.glob(ext)))

        for audio_file in tqdm(audio_files, desc="Обработка файлов"):
            try:
                wav_path = convert(str(audio_file))
                if wav_path is None:
                    continue

                splitter = Splitter(wav_path, self.tomograph_path)
                result = splitter.process()

                if result and result['segments']:
                    source_name = Path(audio_file).stem
                    source_name = source_name.replace('_диктофон', '')

                    for seg in result['segments']:
                        seg['source_file'] = source_name

                    self.all_segments.extend(result['segments'])
                    self.files_info.append({
                        'file': audio_file.name,
                        'segments_count': len(result['segments']),
                        'status': 'success'
                    })
                else:
                    self.files_info.append({
                        'file': audio_file.name,
                        'segments_count': 0,
                        'status': 'no_segments'
                    })
            except Exception as e:
                self.files_info.append({
                    'file': audio_file.name,
                    'segments_count': 0,
                    'status': 'failed',
                    'error': str(e)
                })

        return self._create_dataframe()

    def _create_dataframe(self):
        if not self.all_segments:
            return pd.DataFrame()

        df = pd.DataFrame(self.all_segments)
        df['segment_id'] = df.apply(
            lambda x: f"{x['source_file']}_seg{x['num']:03d}", axis=1
        )
        df = df.rename(columns={
            'path': 'audio_path',
            'filename': 'audio_file',
            'duration': 'duration_sec'
        })
        return df


def load_fmri_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['patient_ID'] = df['patient_ID'].astype(str).str.strip()
    df = df[df['patient_ID'] != 'nan']
    return df


def match_segments_with_transcriptions(
        segments_df: pd.DataFrame,
        fmri_df: pd.DataFrame
) -> pd.DataFrame:
    fmri_df['patient_ID_clean'] = fmri_df['patient_ID'].astype(str).str.strip()
    transcriptions_by_patient = {}

    for idx, row in fmri_df.iterrows():
        patient_id = row['patient_ID_clean']
        if not patient_id or patient_id == 'nan':
            continue

        sentence = str(row.get('sentence', '')).strip()
        response = str(row.get('response', '')).strip()

        if sentence and response and response != 'nan':
            full_text = f"{sentence} {response}"
        elif sentence and sentence != 'nan':
            full_text = sentence
        elif response and response != 'nan':
            full_text = response
        else:
            continue

        full_text = full_text.replace('...', '').strip()
        full_text = full_text.lower()

        if patient_id not in transcriptions_by_patient:
            transcriptions_by_patient[patient_id] = []

        transcriptions_by_patient[patient_id].append({
            'text': full_text,
            'trial': row.get('Trial', idx),
            'block': row.get('Block', ''),
            'cond': row.get('cond', ''),
            'accuracy': row.get('accuracy', ''),
            'sentence': sentence,
            'response': response,
            'original_row': idx
        })

    alt_id_map = {}
    for pid in transcriptions_by_patient.keys():
        pid_str = str(pid)
        alt_id_map[pid_str.replace('_', '')] = pid_str
        if len(pid_str) <= 4 and '_' not in pid_str:
            alt_id_map[f"{pid_str}_"] = pid_str

    matched_segments = []
    unmatched_patients = set()

    for idx, segment in segments_df.iterrows():
        source_file = str(segment.get('source_file', ''))
        matched_id = None

        if source_file in transcriptions_by_patient:
            matched_id = source_file
        elif source_file in alt_id_map:
            matched_id = alt_id_map[source_file]
        else:
            for pid in transcriptions_by_patient.keys():
                pid_str = str(pid)
                if source_file in pid_str or pid_str in source_file:
                    matched_id = pid_str
                    break

        if matched_id:
            patient_transcripts = transcriptions_by_patient[matched_id]
            seg_num = segment.get('num', 1)

            if seg_num - 1 < len(patient_transcripts):
                transcript_info = patient_transcripts[seg_num - 1]

                matched_segment = segment.to_dict()
                matched_segment['text'] = transcript_info['text']
                matched_segment['sentence'] = transcript_info['sentence']
                matched_segment['response'] = transcript_info['response']
                matched_segment['trial'] = transcript_info['trial']
                matched_segment['block'] = transcript_info['block']
                matched_segment['cond'] = transcript_info['cond']
                matched_segment['accuracy'] = transcript_info['accuracy']
                matched_segment['matched_patient_id'] = matched_id

                matched_segments.append(matched_segment)
        else:
            unmatched_patients.add(source_file)

    if matched_segments:
        result_df = pd.DataFrame(matched_segments)
        return result_df
    else:
        return pd.DataFrame()


def split_by_patients(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    unique_patients = df['source_file'].unique().tolist()

    if len(unique_patients) < 3:
        train_patients = unique_patients[:1] if len(unique_patients) > 0 else []
        val_patients = unique_patients[1:2] if len(unique_patients) > 1 else unique_patients[:1]
        test_patients = unique_patients[2:3] if len(unique_patients) > 2 else unique_patients[:1]
    else:
        train_patients, temp_patients = train_test_split(
            unique_patients,
            test_size=test_size + val_size,
            random_state=random_state
        )

        if len(temp_patients) > 1:
            val_relative_size = val_size / (test_size + val_size)
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=test_size / (test_size + val_size),
                random_state=random_state
            )
        else:
            val_patients = temp_patients
            test_patients = []

    train_df = df[df['source_file'].isin(train_patients)].copy()
    val_df = df[df['source_file'].isin(val_patients)].copy()
    test_df = df[df['source_file'].isin(test_patients)].copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    return train_df, val_df, test_df, {
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients
    }


def save_datasets(train_df, val_df, test_df, full_df, output_dir="dataset_split"):
    os.makedirs(output_dir, exist_ok=True)
    full_df.to_csv(f"{output_dir}/all_segments.csv", index=False, encoding='utf-8')
    train_df.to_csv(f"{output_dir}/train_dataset.csv", index=False, encoding='utf-8')
    val_df.to_csv(f"{output_dir}/val_dataset.csv", index=False, encoding='utf-8')
    test_df.to_csv(f"{output_dir}/test_dataset.csv", index=False, encoding='utf-8')


def augment_dataset(df: pd.DataFrame,
                    augmentations_per_segment: int = 2) -> pd.DataFrame:
    segments_augmented = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Аугментация"):
        original_path = row['audio_path']
        original_text = row['text']

        segments_augmented.append(row.to_dict())

        audio, sr = librosa.load(original_path, sr=16000)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        for aug_num in range(augmentations_per_segment):
            augmented_audio = augmentation(audio)
            augmented_audio = augmented_audio / (np.max(np.abs(augmented_audio)) + 1e-6)

            seg_id = row.get('segment_id', f"seg_{idx}")
            aug_path = f"segments_augmented/{seg_id}_aug{aug_num + 1}.wav"

            os.makedirs("segments_augmented", exist_ok=True)
            sf.write(aug_path, augmented_audio, 16000)

            new_row = row.to_dict()
            new_row['segment_id'] = f"{seg_id}_aug{aug_num + 1}"
            new_row['audio_path'] = aug_path
            new_row['is_augmented'] = True
            new_row['augmentation_id'] = aug_num

            segments_augmented.append(new_row)

    result_df = pd.DataFrame(segments_augmented)

    return result_df


def main():
    fmri_df = load_fmri_data('DATA_FMRI.csv')

    builder = AudioDatasetBuilder('audiodataset', 'tomograph_2.wav')
    full_df = builder.process_all_files()

    if full_df.empty:
        return

    matched_df = match_segments_with_transcriptions(full_df, fmri_df)

    if matched_df.empty:
        return

    train_df, val_df, test_df, split_info = split_by_patients(
        matched_df, test_size=0.2, val_size=0.1
    )

    save_datasets(train_df, val_df, test_df, matched_df, output_dir="segments_original")

    with open("dataset_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_augmented_df = augment_dataset(train_df, augmentations_per_segment=2)

    val_df['is_augmented'] = False
    test_df['is_augmented'] = False

    full_augmented_df = pd.concat([train_augmented_df, val_df, test_df], ignore_index=True)

    os.makedirs("dataset_split_augmented", exist_ok=True)

    train_augmented_only = full_augmented_df[full_augmented_df['split'] == 'train']
    val_df.to_csv("dataset_split_augmented/val_dataset.csv", index=False, encoding='utf-8')
    test_df.to_csv("dataset_split_augmented/test_dataset.csv", index=False, encoding='utf-8')
    train_augmented_only.to_csv("dataset_split_augmented/train_dataset.csv", index=False, encoding='utf-8')
    full_augmented_df.to_csv("dataset_split_augmented/all_segments.csv", index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
