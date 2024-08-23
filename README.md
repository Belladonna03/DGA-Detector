# 🚀 Модель обнаружения DGA 🕵️‍♂️

Этот проект содержит модель машинного обучения для обнаружения алгоритмов генерации доменов (DGA). Давайте защитим интернет вместе!

## 📝 Описание

Модель обучается с использованием классификатора RandomForestClassifier и использует различные признаки, такие как длина домена, энтропия и частоты n-грамм, для предсказания, является ли домен сгенерированным DGA или нет.

## 📂 Файлы данных

- `data/all_legit.txt`: Список легальных доменов.
- `data/words.txt`: Список слов для анализа n-грамм.
- `data/val.csv`: Данные для валидации.
- `data/test.csv`: Данные для тестирования.

## 📜 Скрипты

- `scripts/train.py`: Скрипт для обучения модели.
- `scripts/validate.py`: Скрипт для валидации модели.
- `scripts/test.py`: Скрипт для тестирования модели.

## 🚀 Как использовать

1. Клонируйте репозиторий.
2. Установите необходимые зависимости.
3. Запустите скрипт `scripts/train.py` для обучения модели.
4. Запустите скрипт `scripts/validate.py` для валидации модели.
5. Запустите скрипт `scripts/test.py` для предсказаний на тестовых данных.

## 📊 Результаты

После выполнения скриптов `validate.py` и `test.py`, результаты будут сохранены в следующих файлах:

- `results/prediction.csv`: Файл с предсказаниями модели на тестовых данных.
- `results/validation_1.txt`: Файл с результатами валидации модели, включая метрики точности, полноты и F1-score.