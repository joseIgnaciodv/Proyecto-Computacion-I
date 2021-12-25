import streamlit as st
import string
from nltk.tokenize import word_tokenize
import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
import plotly.express as px

def tokenizar_texto(texto):
    lista_tokens = word_tokenize(str(texto.read(), encoding="utf8").lower(), language="spanish")
    return lista_tokens


def limpiar_texto(lista_tokens: list):
    palabras = []
    fichero_parada = open("Lista_Stop_Words.txt", "r", encoding="utf8")
    lista_parada = fichero_parada.read().split("\n")
    puntuacion = list(string.punctuation)
    lista_parada += puntuacion
    for palabra in lista_tokens:
        if palabra not in lista_parada:
            palabras.append(palabra)
    return palabras

def stemming(lista_palabras: list):
    texto = ""
    stemmer = Stemmer.Stemmer('spanish')
    for palabra in lista_palabras:
        s = stemmer.stemWord(palabra)
        texto = texto + " " + s
    return texto

def generar_coleccion(lista_textos: list):
    coleccion = []
    for texto in lista_textos:
        tokens = tokenizar_texto(texto)
        lista_limpia = limpiar_texto(tokens)
        texto = stemming(lista_limpia)
        coleccion.append(texto)
    return coleccion

def predecir_clases(modelo, coleccion_noticias: list):
    tf = TfidfVectorizer(vocabulary=joblib.load('vocabulario.bin'))
    vectores_noticias = tf.fit_transform(coleccion_noticias).toarray()
    matriz_idf = pd.DataFrame(vectores_noticias, columns=tf.get_feature_names_out())
    predicciones = modelo.predict(matriz_idf)
    return predicciones

def main():
    unlabeled = st.file_uploader("Unlabeled", accept_multiple_files=True, type='txt')
    modelo = st.file_uploader("Modelo")
    modelo = joblib.load(modelo)
    coleccion_unlabeled = generar_coleccion(unlabeled)
    predicciones = predecir_clases(modelo, coleccion_unlabeled)
    
    noticias = []
    for n in unlabeled:
        noticias.append(n.name)

    l = []
    for p in predicciones:
        if p == 'Odio':
            l.append('Si')
        else:
            l.append('No')

    resultados = {'Noticia': noticias, 'Odio': l}
    tabla_resultados = pd.DataFrame.from_dict(resultados)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.dataframe(tabla_resultados, width=500)
    with col4:
        guradar_resultados = st.radio("Guardar Resultados: ", ['CSV', 'Excel', 'Txt'])
    if guradar_resultados == 'CSV':
        tabla_resultados.to_csv('resultados.csv',encoding="utf8")
    elif guradar_resultados == 'Excel':
        tabla_resultados.to_excel('resultados.xlsx') 
    elif guradar_resultados == 'Txt':
        tabla_resultados.to_csv('resultados.txt',encoding="utf8")


main()