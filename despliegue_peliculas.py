#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_genres(df_test, fila):
    # Cargamos el modelo de predicción de géneros y el transformador
    model_data = joblib.load(os.path.join(os.path.dirname(__file__), 'model_data.pkl'))
    clf = model_data['model']
    preprocessor = model_data['preprocessor']
    all_genres = model_data['all_genres']
    

    if fila not in df_test.index:
        raise ValueError(f'El índice {fila} no existe en el conjunto de testing.')
    
    # Seleccionamos la fila que queremos predecir
    dato = df_test.loc[[fila], ['title', 'plot', 'year']]
    
    # Aplicamos la transformación al dato de entrada usando el preprocesador
    transformacion = preprocessor.transform(dato)
    
    # Hacemos la predicción
    y_pred_proba = clf.predict_proba(transformacion)
    
    
    prediccion = pd.DataFrame(y_pred_proba, columns=all_genres)
    
    # Determinamos los tres géneros más probables
    top_genres = sorted(
        [(genre, round(prob, 3)) for genre, prob in prediccion.iloc[0].items()],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return top_genres, prediccion

if __name__ == "__main__":

    df_test = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)

    if len(sys.argv) == 1:
        print('Por favor ingrese un número de ID para evaluar en el conjunto de TEST')
    else:
        fila = int(sys.argv[1])
        try:
            top_genres, pred_probas = predict_genres(df_test, fila)
            print(f'Los 3 géneros predichos con más alta probabilidad para la película del ID: {fila} del conjunto de prueba son: {top_genres}')
            print("------------------------------------------------------------------------------------------")
            print("------------------------------------------------------------------------------------------")
            print(f'Datos de probabilidad por género para la película del ID {fila} del conjunto de prueba:\n\n{pred_probas.T}')
        except ValueError as e:
            print(e)
