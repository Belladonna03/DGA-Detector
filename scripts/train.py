import math
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Словарь для хранения датафреймов для каждого типа домена
dataframe_dict = {'alexa': [], 'conficker': [], 'cryptolocker': [], 'zeus': [], 'pushdo': [], 'rovnix': [], 'tinba': [],
                  'matsnu': [], 'ramdo': []}

# Загрузка и предварительная обработка списка слов
word_dataframe = pd.read_csv('data/words.txt', names=['word'], header=None, dtype={'word': str}, encoding='utf-8')
word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

# Загрузка и предварительная обработка данных доменов
for i, v in dataframe_dict.items():
    if i == 'alexa':
        v = pd.read_csv('data/all_legit.txt', names=['uri'], header=None, encoding='utf-8')
        v['domain'] = v['uri'].apply(lambda x: x.split('.')[0].strip().lower())
        del v['uri']
        v['is_dga'] = 0
        dataframe_dict[i] = v
    else:
        v = pd.read_csv('dga_wordlists/' + i + '.txt', names=['uri'], header=None, encoding='utf-8')
        v['domain'] = v['uri'].apply(lambda x: x.split('.')[0].strip().lower())
        del v['uri']
        v['is_dga'] = 1
        dataframe_dict[i] = v

# Объединение всех данных доменов в один датафрейм
all_domains = pd.concat([dataframe_dict['alexa'], dataframe_dict['conficker'], dataframe_dict['cryptolocker'],
                         dataframe_dict['zeus'], dataframe_dict['pushdo'], dataframe_dict['rovnix'],
                         dataframe_dict['tinba'], dataframe_dict['matsnu'], dataframe_dict['ramdo']],
                        ignore_index=True)

val_data = pd.read_csv('data/val.csv')
test_data = pd.read_csv('data/test.csv')

# Объединение столбцов "domain" из данных валидации и тестирования в один DataFrame
combined_df = pd.concat([val_data['domain'], test_data['domain']], ignore_index=True).drop_duplicates()

# Создание маски для идентификации строк в исходном датафрейме, которые имеют совпадения в combined_df на основе столбца "domain"
mask = all_domains['domain'].isin(combined_df)
all_domains = all_domains[~mask]

# Признак: Длина домена
all_domains['length'] = all_domains['domain'].apply(len)

# Признак: Энтропия домена
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())
all_domains['entropy'] = all_domains['domain'].apply(entropy)

# Признак: N-граммы Alexa
alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(dataframe_dict['alexa']['domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
all_domains['alexa_grams'] = alexa_counts * alexa_vc.transform(all_domains['domain']).T

# Признак: N-граммы словаря
dict_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-5, max_df=1.0)
counts_matrix = dict_vc.fit_transform(word_dataframe['word'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())
all_domains['word_grams'] = dict_counts * dict_vc.transform(all_domains['domain']).T

# Признак: Разница между N-граммами Alexa и словаря
all_domains['diff'] = all_domains['alexa_grams'] - all_domains['word_grams']

# Подготовка данных для моделирования
X_train = all_domains[['length', 'entropy', 'alexa_grams', 'word_grams', 'diff']].values
y_train = all_domains['is_dga'].values

# Определение классификатора
clf = RandomForestClassifier(n_estimators=1500, random_state=1, n_jobs=-1)

# Обучение классификатора
clf.fit(X_train, y_train)

# Сохранение модели в файл
joblib.dump(clf, 'rf_model.pkl')