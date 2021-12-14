import streamlit as st
import os
import subprocess
import nltk
from sklearn.model_selection import cross_validate

nltk.download('punkt')

def contar_ficheros(lista_ficheros: list):
    return len(lista_ficheros)

def gradient_boosted_tree():
    return ""

def SVM():
    return ""

def seleccionar_algoritmo(algoritmo: str):
    if algoritmo == "Gradient Boosted Tree":
        gradient_boosted_tree()
    elif algoritmo == "Support Vector Machine":
        SVM()

def main():
    lista_odio = st.file_uploader('Noticias Odio: ', accept_multiple_files=True, type='txt')
    lista_no_odio = st.file_uploader('Noticias No Odio: ', accept_multiple_files=True, type='txt')

    algoritmo = st.selectbox("Seleccionar Algoritmo: ", ["Gradient Boosted Tree", "Support Vector Machine"])

    num_odio = contar_ficheros(lista_odio)
    num_no_odio = contar_ficheros(lista_no_odio)

    preview = st.text_area("Vista Previa", "Ejemplares 'Odio': " + "\t" + str(num_odio) + "\nEjemplares 'No Odio': " + "\t" + str(num_no_odio) + "\nTotal: " + "\t" + str(num_odio + num_no_odio) + "\nAlgoritmo Seleccionado: " + "\t" + algoritmo)

    col1, col2, col3 = st.columns(3)
    with col2:
        ejecutar = st.button("Ejecutar")

main()
