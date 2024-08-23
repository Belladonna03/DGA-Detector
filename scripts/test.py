import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Загрузка модели из файла
rf_model = joblib.load('rf_model.pkl')
# Загрузка тестового образца
test_data = pd.read_csv('data/test.csv')
# Применение инженерных признаков к тестовым данным
test_data['length'] = test_data['domain'].apply(len)

def entropy(s):
    from collections import Counter
    import math
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

test_data['entropy'] = test_data['domain'].apply(entropy)

alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(pd.read_csv('data/all_legit.txt', names=['uri'], header=None, encoding='utf-8')['uri'].apply(lambda x: x.split('.')[0].strip().lower()))
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
test_data['alexa_grams'] = alexa_counts * alexa_vc.transform(test_data['domain']).T

word_dataframe = pd.read_csv('words.txt', names=['word'], header=None, dtype={'word': str}, encoding='utf-8')
word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

dict_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-5, max_df=1.0)
counts_matrix = dict_vc.fit_transform(word_dataframe['word'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())
test_data['word_grams'] = dict_counts * dict_vc.transform(test_data['domain']).T

test_data['diff'] = test_data['alexa_grams'] - test_data['word_grams']

X_test = test_data[['length', 'entropy', 'alexa_grams', 'word_grams', 'diff']].values

# Создание предсказаний на тестовом наборе
y_test_pred = rf_model.predict(X_test)
# Создание DataFrame с предсказаниями
predictions = pd.DataFrame({'domain': test_data['domain'], 'is_dga': y_test_pred})
# Сохранение предсказаний в prediction.csv
predictions.to_csv('results/prediction.csv', index=False)