import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib 
import os

# Descargamos stopwords
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

data_training = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
data_testing = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)


data_training['genres'] = data_training['genres'].apply(ast.literal_eval)
total_generos = sorted(set(g for generos in data_training['genres'] for g in generos))
for genero_m in total_generos:
    data_training[genero_m] = data_training['genres'].apply(lambda x: 1 if genero_m in x else 0)


X = data_training[['title', 'plot', 'year']]
y = data_training[total_generos]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words)

    def fit(self, X, y=None):
        self.vectorizer.fit(X[self.column])
        return self

    def transform(self, X):
        return self.vectorizer.transform(X[self.column])

# Creamos pipelines para cada tipo de transformación
transformador_titulo = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words=stop_words))
])

transformador_plot = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words=stop_words))
])

transformador_anio = Pipeline(steps=[
    ('scaler', StandardScaler())
])


preprocesador = ColumnTransformer(
    transformers=[
        ('title', transformador_titulo, 'title'),
        ('plot', transformador_plot, 'plot'),
        ('year', transformador_anio, ['year'])
    ])


X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)


X_train_transformado = preprocesador.fit_transform(X_train)
X_val_transformado = preprocesador.transform(X_val)


modelo = LogisticRegression(solver='liblinear')


param_grid = {'estimator__C': [0.1, 1, 10, 100]}
cv = KFold(n_splits=5)
grid_search = GridSearchCV(OneVsRestClassifier(modelo), param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train_transformado, y_train)


best_model = grid_search.best_estimator_



modelo_final = {
    'preprocessor': preprocesador,
    'model': best_model,
    'all_genres': total_generos
}



ruta_modelos = "D:/LOS ANDES/MACHINE LEARNING Y NLP/PROYECTO API PELICULAS/"

joblib.dump(modelo_final, os.path.join(ruta_modelos, 'model_data.pkl'), compress=3)
joblib.dump(modelo_final, os.path.join(ruta_modelos, 'model_data.joblib'))

