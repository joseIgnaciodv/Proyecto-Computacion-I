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

@st.cache
def generar_coleccion(lista_textos: list):
    coleccion = []
    for texto in lista_textos:
        tokens = tokenizar_texto(texto)
        lista_limpia = limpiar_texto(tokens)
        texto = stemming(lista_limpia)
        coleccion.append(texto)
    return coleccion

@st.cache
def predecir_clases(modelo, coleccion_noticias: list):
    tf = TfidfVectorizer(vocabulary=joblib.load('vocabulario.bin'))
    vectores_noticias = tf.fit_transform(coleccion_noticias).toarray()
    matriz_idf = pd.DataFrame(vectores_noticias, columns=tf.get_feature_names_out())
    predicciones = modelo.predict(matriz_idf)
    return predicciones

@st.cache
def grafica_resultados(lista_ficheros: pd.DataFrame, cont_odio: int, cont_no: int):
    for clase in lista_ficheros['Odio']:
        if clase == 'Si':
            cont_odio += 1
        else:
            cont_no += 1
    return {'Odio': cont_odio, 'No Odio': cont_no}
        