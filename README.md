# Картирование речи методом фМРТ
_Автоматический анализатор, созданный в рамках курсовой работы на 3 курсе программы Фундаментальная и Компьютерная лингвистика, НИУ ВШЭ_

## Описание

Программа, направленная на распознавание речи, записанной при картировании методом функциональной магнитно-резонансной томофграфии, и классификации по корректности употребления с точки зрения современного русского языка.

### Структура репозитория
* В папке main_programm содержатся файлы:
  * main.py (получает на вход аудиозапись и возвращает csv с заполненной согласно принятой разметке строкой)
  * preprocessing.py (предобрабатывает аудиозаписи, возвращает транскрипцию)
  * speech.py (возврщает решения о корректности реакции)
  * tomograph.wav (запись звука томографа для кросс-валидации)
* В папке prepare models лежат файлы с кодом и данными, использовавшимися для дообучения модели распознавания GigaAM-CTC-v3 и проведения сравнительного анализа полученных результатов
   * modeltraining.py (аугментирует файлы)
   * metricsc.py (код для проведения сопоставления результаты работы моделей)
   * modelevaluation (папка, где хранятся результаты работы metrics.py)
   * папка audiodaset (хранилище аудиозаписей для обучения)
   * modelevaluation (папка, где хранятся результаты
   * data (папка, содержащая в себе код и данные для сопоставления аудио и расшифровок, а также аугментирования)
      * DATA-FMRI.csv (изначальный csv с расшифровками)
      * splitdata.py (код для подготовки датасета)
      * segments_original (папка, где лежат выделенные, вырезанные сегменты)
      * segments_augmented (папка, где лежат аугментированные варианты сегментов)
      * model_datasets (папка, где лежат разделенные на тренировочный, тестовый и валидационный датасеты)
  
    


## Установка и использование
Потребуется версия python = 3.10

### Как использовать

скачайте папку main_program, запустите файл main.py
при необходимости загрузите файл с аудиозаписью в рабочую папку программы и переименуйте название анализируемого файла в коде main.py

## Ресурсы
https://wonderscribe.pro/blog/format-audio-dlya-transkribacii-wav-mp3 

https://pypi.org/project/natasha/

https://github.com/revitaai/govprobing

https://huggingface.co/docs/transformers/model_doc/bert

Elin K., Malyutina S., Bronov O., Stupina E., Marinets A., Zhuravleva A., Dragoy O. A New Functional Magnetic Resonance Imaging Localizer for Preoperative Language Mapping Using a Sentence Completion Task: Validity, Choice of Baseline Condition, and Test–Retest Reliability // Frontiers in Human Neuroscience vol. 16, article 791577, 2022.

## Контакт для [обратной] связи
git: @vkormili82; tg: @vkormili; mail: vkormili@gmail.com
