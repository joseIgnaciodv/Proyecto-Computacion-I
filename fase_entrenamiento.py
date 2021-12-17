from pandas.core.algorithms import mode
import streamlit as st
import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from nltk.stem.snowball import SpanishStemmer
import pandas as pd
import plotly.express as px

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
    model = ""
    if algoritmo == "Gradient Boosted Tree":
        model = GradientBoostingClassifier()
    elif algoritmo == "Support Vector Machine":
        model = SVC()
    elif algoritmo == "Arbol Decision":
        model = DecisionTreeClassifier()
    return model

def asociar_clase(odio: list, no_odio: list):
    clase_odio = []
    clase_no_odio = []
    for texto_odio in odio:
        clase_odio.append(1)
    for texto_no in no_odio:
        clase_no_odio.append(0)
    return clase_odio + clase_no_odio

def entrenar_modelo(algoritmo: str, coleccion_documentos: list, clases: list):
    tf = TfidfVectorizer()
    matriz_idf = tf.fit_transform(coleccion_documentos).toarray()
    df = pd.DataFrame(matriz_idf , columns=tf.get_feature_names_out())
    clf = seleccionar_algoritmo(algoritmo)
    modelo = clf.fit(df, clases)

def puntuacion_cross_validation(algoritmo: str, coleccion_documentos: list, clases: list):
    modelo = seleccionar_algoritmo(algoritmo)
    clf = make_pipeline(TfidfVectorizer(), modelo)
    y_pred = cross_val_predict(clf, coleccion_documentos, clases, cv=10)
    cm = confusion_matrix(clases, y_pred)
    return cm


def visualizacion_previa(odio, no_odio, algoritmo):
    num_odio = contar_ficheros(odio)
    num_no_odio = contar_ficheros(no_odio)

    ejemplares = {'Odio': num_odio, 'No Odio': num_no_odio}
    st.text_area("Vista Previa", "Ejemplares 'Odio': " + "\t" + str(num_odio) + "\nEjemplares 'No Odio': " + "\t" + str(num_no_odio) + "\nTotal: " + "\t" + str(num_odio + num_no_odio) + "\nAlgoritmo Seleccionado: " + "\t" + algoritmo, height=130)

    df = pd.DataFrame.from_dict(ejemplares, orient='index', columns=['Ejemplares'])
    df = df.rename_axis('Clase')
    figura = px.bar(df, y="Ejemplares", color="Ejemplares")
    st.plotly_chart(figura)

def main():
    lista_odio = st.file_uploader('Noticias Odio: ', accept_multiple_files=True, type='txt')
    lista_no_odio = st.file_uploader('Noticias No Odio: ', accept_multiple_files=True, type='txt')

    algoritmo = st.selectbox("Seleccionar Algoritmo: ", ["Gradient Boosted Tree", "Support Vector Machine", "Arbol Decision"])

    visualizacion_previa(lista_odio, lista_no_odio, algoritmo)
    col1, col2, col3 = st.columns(3)
    with col2:
        ejecutar = st.button("Ejecutar")
    
    odio = generar_coleccion(lista_odio)
    no_odio = generar_coleccion(lista_no_odio)

    coleccion = odio + no_odio
    clases = asociar_clase(odio, no_odio)

    if ejecutar:
        resultados_entrenamiento = puntuacion_cross_validation(algoritmo, coleccion, clases)
        st.write(resultados_entrenamiento)
main()