import streamlit as st
import os
import subprocess

def contar_ficheros(ruta_carpeta: str):
    carpeta = list(os.scandir(ruta_carpeta))
    return len(carpeta)

def seleccionar_carpeta(ruta):
    fichero = subprocess.Popen(r'explorer /select,')
    print(fichero.stdout.read())


    

col1, col2 = st.columns(2)

with col1:
    odio = st.text_input("Noticias Odio", value="")
with col2:
    st.write(".")
    abrir_fichero = st.button("Abrir Odio")

col1, col2 = st.columns(2)

with col1:
    odio = st.text_input("Noticias No Odio", value="")
with col2:
    st.write(".")
    abrir_fichero_no = st.button("Abrir No Odio")

algoritmo = st.selectbox("Seleccionar Algoritmo: ", ["Gradient Boosted Tree", "Support Vector Machine"])



if abrir_fichero:
    seleccionar_carpeta(r"C:\Users\delva\Downloads\noticias_odio_pais\noticias_odio_pais")
