import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Загрузка модели из файла
rf_model = joblib.load('rf_model.pkl')

# Загрузка образца валидации
val_data = pd.read_csv('val.csv')

# Применение инженерных признаков к данным валидации
val_data['length'] = val_data['domain'].apply(len)

def entropy(s):
    from collections import Counter
    import math
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

val_data['entropy'] = val_data['domain'].apply(entropy)

alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(pd.read_csv('all_legit.txt', names=['uri'], header=None, encoding='utf-8')['uri'].apply(lambda x: x.split('.')[0].strip().lower()))
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
val_data['alexa_grams'] = alexa_counts * alexa_vc.transform(val_data['domain']).T

word_dataframe = pd.read_csv('data/words.txt', names=['word'], header=None, dtype={'word': str}, encoding='utf-8')
word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

dict_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-5, max_df=1.0)
counts_matrix = dict_vc.fit_transform(word_dataframe['word'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())
val_data['word_grams'] = dict_counts * dict_vc.transform(val_data['domain']).T

val_data['diff'] = val_data['alexa_grams'] - val_data['word_grams']

X_val = val_data[['length', 'entropy', 'alexa_grams', 'word_grams', 'diff']].values
y_val = val_data['is_dga'].values

# Создание предсказаний на наборе валидации
y_val_pred = rf_model.predict(X_val)

# Матрица ошибок
cm = confusion_matrix(y_val, y_val_pred)

# Извлечение TP, FP, FN, TN
TN, FP, FN, TP = cm.ravel()

# Расчет метрик
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

# Запись результатов в validation.txt
with open('results/validation.txt', 'w') as f:
    f.write(f"True positive: {TP}\n")
    f.write(f"False positive: {FP}\n")
    f.write(f"False negative: {FN}\n")
    f.write(f"True negative: {TN}\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1: {f1:.3f}\n")