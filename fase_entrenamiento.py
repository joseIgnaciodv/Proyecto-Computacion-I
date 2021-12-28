import streamlit as st
#import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import Stemmer                              # Descargar PyStemmer, pip install PyStemmer
import pandas as pd
import plotly.express as px
import plotly.figure_factory as plt
import joblib
# tokenizador que utiliza nltk
# nltk.download('punkt')                # Solo descargar una vez

# Funcion que cuenta el numero de fichero que incluye una lista
def contar_ficheros(lista_ficheros: list):
    return len(lista_ficheros)

# Funcion que convierte a minusculas el texto pasado como argumento y lo tokeniza/separa las palabras del texto
def tokenizar_texto(texto):
    lista_tokens = word_tokenize(str(texto.read(), encoding="utf8").lower(), language="spanish")
    return lista_tokens

# Funcion en la que aplicamos un lista de parada, es decir, le pasamos un fichero con las palabras o tokens que nos
# interesa eliminar del texto procesado y las quitamos
def limpiar_texto(lista_tokens: list):
    palabras = []
    # Abrimos nuestra lista de parada
    fichero_parada = open("Lista_Stop_Words.txt", "r", encoding="utf8")
    # Hemos leido el fichero como un string y necesitamos una lista de strings con las palabras a eliminar,
    # para ello aplicamos la funcion split para separar el texto por los espacios
    lista_parada = fichero_parada.read().split("\n")
    # Creamos una lista con los diferentes caracteres(!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~) y los añadimos a nuestra
    # lista de parada, pues no nos son relevantes tampoco a la hora de aplicar nuestro algoritmo de aprendizaje
    puntuacion = list(string.punctuation)
    lista_parada += puntuacion
    # Aplicamos la lista de parada y nos quedamos solo con las palabras/tokens que nos interesan
    for palabra in lista_tokens:
        if palabra not in lista_parada:
            palabras.append(palabra)
    return palabras

# Funcion con la que aplicaremos el stemming (esta técnica nos ayudará a simplificar o reducir los tokens o palabras al stem) 
def stemming(lista_palabras: list):
    texto = ""
    # Cogemos el stemmer de snowball en español
    stemmer = Stemmer.Stemmer('spanish')
    for palabra in lista_palabras:
        # Cogemos el stem de cada palabra
        s = stemmer.stemWord(palabra)
        texto = texto + " " + s
    return texto

# Funcion que genera una coleccion con los textos procesados, es decir, aplicando la tokenizacion, la transformacion a
# minusculas, eliminar las palabras que no se quieren a partir de una lista de parada y aplicacion de un stemmer(Snowball)
def generar_coleccion(lista_textos: list):
    coleccion = []
    for texto in lista_textos:
        # Aplicamos tokenizacion del texto
        tokens = tokenizar_texto(texto)
        # Aplicamos lista de parada
        lista_limpia = limpiar_texto(tokens)
        # Aplicamos stemmer
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

# Funcion que asigna el tipo de noticia (odio o no odio) a la que pertenece el texto
def asociar_clase(odio: list, no_odio: list):
    clase_odio = []
    clase_no_odio = []
    for texto_odio in odio:
        clase_odio.append('Odio')
    for texto_no in no_odio:
        clase_no_odio.append('No Odio')
    return clase_odio + clase_no_odio

def precision(tp: int, fp: int):
    return tp/(tp + fp)

def recall(tp: int, fn: int):
    return tp/(tp + fn)

def entrenar_modelo(algoritmo: str, coleccion_documentos: list, clases: list):
    tf = TfidfVectorizer()
    matriz_idf = tf.fit_transform(coleccion_documentos).toarray()
    df = pd.DataFrame(matriz_idf , columns=tf.get_feature_names_out())
    clf = seleccionar_algoritmo(algoritmo)
    X_train, X_test, Y_train, Y_test = train_test_split(df, clases, test_size=0.3)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    modelo = clf.fit(X_train, Y_train)
    y_pred = modelo.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    cm = [[tn, fp], [fn, tp]]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision: ", "{:.2f}".format((precision(tp, fp) * 100)) + "%", delta="{:.2f}".format((precision(tp, fp) * 100) - 95) + "%")
    with col2:
        st.metric("Accuracy: ", "{:.2f}".format(modelo.score(X_test, Y_test) * 100) + "%", delta="{:.2f}".format((modelo.score(X_test, Y_test) * 100) - 95) + "%")
    with col3:
        st.metric("Recall: ", "{:.2f}".format((recall(tp, fn) * 100)) + "%", delta="{:.2f}".format((recall(tp, fn) * 100) - 95) + "%")
    st.write("Resultados Entrenamiento: ")
    figura = plt.create_annotated_heatmap(cm, colorscale='Viridis')
    st.plotly_chart(figura)
    joblib.dump(tf.vocabulary_, 'vocabulario.bin')
    return modelo

def visualizacion_previa(odio, no_odio, algoritmo):
    num_odio = contar_ficheros(odio)
    num_no_odio = contar_ficheros(no_odio)

    ejemplares = {'Odio': num_odio, 'No Odio': num_no_odio}
    st.text_area("Vista Previa", "Ejemplares 'Odio': " + "\t" + str(num_odio) + "\nEjemplares 'No Odio': " + "\t" + str(num_no_odio) + "\nTotal: " + "\t" + str(num_odio + num_no_odio) + "\nAlgoritmo Seleccionado: " + "\t" + algoritmo, height=130)

    df = pd.DataFrame.from_dict(ejemplares, orient='index', columns=['Ejemplares'])
    df = df.rename_axis('Clase')
    figura = px.bar(df, y="Ejemplares", color="Ejemplares")
<<<<<<< HEAD
    st.plotly_chart(figura)
=======
    st.plotly_chart(figura)

def main():

    st.title("Fase de entrenamiento")

    # Permitimos subir varios archivos mediante el componente "file_uploader" de Streamlit
    lista_odio = st.file_uploader('Noticias Odio: ', accept_multiple_files=True, type='txt')
    lista_no_odio = st.file_uploader('Noticias No Odio: ', accept_multiple_files=True, type='txt')

    # Habilitamos una caja que nos permita elegir el algoritmo que queremos utilizar
    algoritmo = st.selectbox("Seleccionar Algoritmo: ", ["Gradient Boosted Tree", "Support Vector Machine", "Arbol Decision"])

    # Llamamos a nuestra funcion previamente creada que nos permite visualizar el numero de ejemplares de odio, 
    # de no odio, los totales y el algoritmo selccionado. También nos muestra un gráfico de barras de con el numero de 
    # ejemplares de delitos de odio vs los de no odio
    visualizacion_previa(lista_odio, lista_no_odio, algoritmo)

    # Generamos la coleccion total de textos de odio
    odio = generar_coleccion(lista_odio)
    # Generamos la coleccion total de textos de no odio
    no_odio = generar_coleccion(lista_no_odio)
    # Concatenamos ambas colecciones
    coleccion = odio + no_odio

    # Indicamos el tipo de noticia
    clases = asociar_clase(odio, no_odio)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
        guardar = st.button("Guardar Modelo")
    with col3:
        st.write("")
    
    if guardar:
        st.title("Resultados")
        st.write("")
        fecha = datetime.now()
        st.write("Fecha realizacion entrenamiento: " + fecha.strftime("%d-%m-%Y %H:%M"))
        st.write("")
        modelo_entrenado = entrenar_modelo(algoritmo, coleccion, clases)
        joblib.dump(modelo_entrenado, 'modelo.bin')
        st.success("Modelo Guardado correctamente")
>>>>>>> ad2bc6a63b5299ac85e101f90aa4f8b8cb17dee5
