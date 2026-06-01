import numpy as np
import onnx_asr
import soundfile as sf
from scipy import signal
import noisereduce as nr
import librosa
import re
import csv
from transformers import RobertaTokenizer, RobertaModel
import torch
from autocorrect import Speller
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc, MorphVocab
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import warnings

warnings.filterwarnings('ignore')


tokenizer_roberta = RobertaTokenizer.from_pretrained('ai-forever/ru-en-RoSBERTa')
model_roberta = RobertaModel.from_pretrained('ai-forever/ru-en-RoSBERTa')

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
spell = Speller(lang='ru')


def govprobing(path='GovProbing'):
    p = {}
    with open(f'{path}/RUS-verb.tsv', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if len(row) < 9:
                continue
            for v in (row[2].strip(), row[3].strip()):
                if not v:
                    continue
                cases = {c[:3] for c in re.findall(r'Case:(\w+)', row[6] + row[7] + row[8])}
                if cases:
                    p[v] = list(cases)
    return p


''' Класс для работы с распознаванием '''


class Recognition:
    def __init__(self):
        self.gigaam = onnx_asr.load_model('gigaam-v3-ctc')

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
        if self.gigaam is None:
            return ''
        audio_clean, sr = self.clean(audio)
        if audio_clean is None:
            return ''
        hypothesis = self.gigaam.recognize(audio_clean, sample_rate=16000)
        return hypothesis

    def result(self, audio):
        line = self.transcription(audio)
        if not line:
            return ''
        line = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', line).lower().strip()
        return line


''' Класс для работы с классификацией '''


class Analysis:
    def __init__(self, line: str, pattern: str, type: str):
        self.type = type
        self.line = re.sub(r'(\*+)', r'\1', line) if line else ''
        self.pattern = pattern
        self.flag = False
        self.flag_phon = False
        self.flag_gram = False
        self.flag_sem = False
        self.cosine_similarity_score = 0.0

        self.govprobing = govprobing()

        self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        self.model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')

        self.mask_tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        self.mask_model = AutoModel.from_pretrained('cointegrated/rubert-tiny2')
        self.mask_model.eval()

        self.verb = ''
        self.noun = ''

    def control(self):
        if not self.line or not self.pattern:
            return False, False
        self.line = re.sub(' ', '', self.line)
        if self.line[:-2] == self.pattern:
            self.flag = True
        if len(self.line) >= 3 and len(self.pattern) >= 3:
            if self.line[-3:] == self.pattern[-3:]:
                self.result = True
        return self.flag, getattr(self, 'result', False)

    def _agreement(self, wordA, wordB):
        if not wordA or not wordB:
            return False

        docA = Doc(wordA.lower())
        docA.segment(segmenter)
        if docA.tokens:
            docA.tag_morph(morph_tagger)

        docB = Doc(wordB.lower())
        docB.segment(segmenter)
        if docB.tokens:
            docB.tag_morph(morph_tagger)

        if not docA.tokens or not docB.tokens:
            return False

        t1, t2 = docA.tokens[0], docB.tokens[0]

        if t1.pos == 'ADJ':
            f1, f2 = dict(t1.feats or {}), dict(t2.feats or {})
            return f1.get('Gender') == f2.get('Gender') and f1.get('Number') == f2.get('Number') and f1.get('Case') == f2.get('Case')

        elif t1.pos == 'VERB':
            t1.lemmatize(morph_vocab)
            lemma = t1.lemma
            f2 = dict(t2.feats or {})
            actual_case = f2.get('Case')
            expected_cases = self.govprobing.get(lemma, [])
            return actual_case in expected_cases

        return False

    def __phonetic(self):
        if not self.line:
            return False
        clean = spell(self.line)
        if clean == self.line:
            self.flag_phon = True
        else:
            self.line = clean
        return self.flag_phon

    def __grammar(self):
        if not self.line:
            return False

        words = self.line.split()
        if len(words) < 4:
            return False

        self.noun = words[-1]

        doc = Doc(words[2].lower())
        doc.segment(segmenter)
        if not doc.tokens:
            return False

        doc.tag_morph(morph_tagger)

        if doc.tokens[0].pos == 'ADJ':
            if len(words) > 3:
                self.flag_gram = self._agreement(words[1], words[3]) and self._agreement(words[2], words[3])
                self.verb = words[2]
        else:
            if len(words) > 3:
                self.verb = words[3]

        if self.flag_gram and self.verb:
            self.flag_gram = self._agreement(words[2], words[3])

        return self.flag_gram

    def __cosine_similarity(self, text1: str, text2: str):
        if not text1 or not text2:
            return 0.0

        inputs1 = tokenizer_roberta(text1, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs1 = model_roberta(**inputs1)
            embeddings1 = outputs1.last_hidden_state.mean(dim=1).numpy()

        inputs2 = tokenizer_roberta(text2, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs2 = model_roberta(**inputs2)
            embeddings2 = outputs2.last_hidden_state.mean(dim=1).numpy()

        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        return float(similarity)

    def __embedding(self, text: str):
        inputs = self.mask_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.mask_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        return embedding

    def __semantic_bert(self, words: list) -> float:
        if len(words) < 2:
            return 0.0

        prefix_words = words[:3] if len(words) >= 3 else words[:-1]
        prefix = ' '.join(prefix_words)

        sentence_with_mask = f'{prefix} {self.mask_tokenizer.mask_token}'
        original_sentence = ' '.join(words)

        emb_with_mask = self.__embedding(sentence_with_mask)
        emb_original = self.__embedding(original_sentence)

        similarity = cosine_similarity(emb_with_mask, emb_original)[0][0]

        words = self.line.split()
        if len(words) >= 4:
            bert_similarity = float(similarity)
            return bert_similarity > 0.5
        return False

    def __semantic_roles(self) -> bool:
        words = self.line.split()
        if len(words) < 4:
            return False

        score2 = self.__cosine_similarity(f'{self.noun} {self.verb}', f'{self.noun} {self.verb}')
        score1 = self.__cosine_similarity(f'{self.verb} {self.noun}, f'{self.verb} {self.noun})

        return not score1 > 0.6 or score2 > 0.5

    def __semantic_proximity(self) -> bool:
        if not self.noun:
            return False
        sims = []
        if self.verb:
            sims.append(self.__cosine_similarity(self.noun, self.verb))
        if not sims:
            return False
        return max(sims) > 0.1

    def _semantic(self):
        self.flag_sem = self.__semantic_proximity() and self.__semantic_roles()
        return self.flag_sem

    def experiment(self):
        if not self.line or not self.pattern:
            return self.flag, False, False, False, 0.0

        if self.line == self.pattern:
            self.flag = True

        words = self.line.split()
        if len(words) != 4:
            return self.flag, False, False, False, 0.0

        if len(words) >= 4:
            verb = words[1]
            noun = words[3]
            self.verb = verb
            self.noun = noun

            self.cosine_similarity_score = self.__cosine_similarity(self.line, self.pattern)

            bert_similarity = self.__semantic_bert(words)
            self.flag_sem = bert_similarity > 0.9

        return self.flag, self.flag_phon, self.flag_gram, self.flag_sem, self.cosine_similarity_score

    def process(self):
        if self.type == 'real':
            self.__phonetic()
            self.__grammar()
            result = self.experiment()

            if not result[3]:
                self._semantic()

            return result
        else:
            return self.control()

    def get_result(self) -> dict:
        return {'text': self.line,
                'response': self.noun,
                'phon': self.flag_phon,
                'gram': self.flag_gram,
                'sem': self.flag_sem}


# Создаем экземпляр класса с тестовой строкой
analysis = Analysis(
    line='водитель громко торопит комод',
    pattern='водитель громко торопит',
    type='real'
)

# Получаем эмбеддинги через методы класса
words = analysis.line.split()  # ['водитель', 'громко', 'торопит', 'аллигатора']

# Вычисляем семантическую близость через маскирование
bert_similarity = analysis._Analysis__semantic_bert(words)
print(f'BERT similarity (masking): {bert_similarity:.4f}')

# Сравниваем с эталоном
cosine_with_pattern = analysis._Analysis__cosine_similarity(
    analysis.line,
    analysis.pattern
)
print(f'Cosine with pattern: {cosine_with_pattern:.4f}')

# Вывод
if bert_similarity > 0.9:
    print('✓ Конструкция естественна (как в коде)')
else:
    print('✗ Конструкция неестественна (как в коде)')