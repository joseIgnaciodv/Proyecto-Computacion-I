import streamlit as st
import os
import subprocess
import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.stem.snowball import SpanishStemmer
import pandas as pd

nltk.download('punkt')

def contar_ficheros(lista_ficheros: list):
    return len(lista_ficheros)

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
    stemmer = SpanishStemmer()
    for palabra in lista_palabras:
        s = stemmer.stem(palabra)
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

def seleccionar_algoritmo(algoritmo: str):
    if algoritmo == "Gradient Boosted Tree":
        cv = cross_validate(GradientBoostingClassifier, )
    elif algoritmo == "Support Vector Machine":
        cv = cross_validate(SVC, )
    elif algoritmo == "Arbol Decision":
        cv = cross_validate(DecisionTreeClassifier,)

def main():
    lista_odio = st.file_uploader('Noticias Odio: ', accept_multiple_files=True, type='txt')
    lista_no_odio = st.file_uploader('Noticias No Odio: ', accept_multiple_files=True, type='txt')

    algoritmo = st.selectbox("Seleccionar Algoritmo: ", ["Gradient Boosted Tree", "Support Vector Machine", "Arbol Decision"])

    num_odio = contar_ficheros(lista_odio)
    num_no_odio = contar_ficheros(lista_no_odio)

    prueba = {'Odio': num_odio, 'No Odio': num_no_odio}
    preview = st.text_area("Vista Previa", "Ejemplares 'Odio': " + "\t" + str(num_odio) + "\nEjemplares 'No Odio': " + "\t" + str(num_no_odio) + "\nTotal: " + "\t" + str(num_odio + num_no_odio) + "\nAlgoritmo Seleccionado: " + "\t" + algoritmo, height=130)

    df = pd.DataFrame.from_dict(prueba, orient='index')

    st.bar_chart(df)
    col1, col2, col3 = st.columns(3)
    with col2:
        ejecutar = st.button("Ejecutar")

main()


