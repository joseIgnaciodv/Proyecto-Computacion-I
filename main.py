import streamlit as st
import os
from tkinter import filedialog, Tk
from fase_entrenamiento import *
from fase_testeo import *
import joblib
import pandas as pd
import plotly.express as px

paginas = {
"pagina_1": "Fase Entrenamiento",
"pagina_2": "Fase Testeo",
}

selected_page = st.sidebar.radio("Selecciona la página", paginas.values())

root = Tk()
root.attributes('-topmost',True)
root.withdraw()

if selected_page == paginas["pagina_1"]: 
    st.title("Fase de entrenamiento")

    # Permitimos subir varios archivos mediante el componente "file_uploader" de Streamlit
    lista_odio = st.file_uploader('Noticias Odio: ', accept_multiple_files=True, type='txt')
    lista_no_odio = st.file_uploader('Noticias No Odio: ', accept_multiple_files=True, type='txt')

    # Habilitamos una caja que nos permita elegir el algoritmo que queremos utilizar
    algoritmo = st.selectbox("Seleccionar Algoritmo: ", ["Gradient Boosted Tree", "Support Vector Machine", "Arbol Decision"])

    # Llamamos a nuestra funcion previamente creada que nos permite visualizar el numero de ejemplares de odio, 
    # de no odio, los totales y el algoritmo seleccionado. También nos muestra un gráfico de barras de con el numero de 
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
        dir_act = os.getcwd()
        nombre_fichero = filedialog.asksaveasfile('w', parent=root, initialdir=dir_act, title='Seleccione una carpeta', defaultextension=".bin", filetypes=(("Binario", "*.bin"),))

        st.title("Resultados")
        st.write("")
        fecha = datetime.now()
        st.write("Fecha realizacion entrenamiento: " + fecha.strftime("%d-%m-%Y %H:%M"))
        st.write("")
        modelo_entrenado = entrenar_modelo(algoritmo, coleccion, clases)
        joblib.dump(modelo_entrenado, nombre_fichero.name)
        st.success("Modelo Guardado correctamente")
elif selected_page == paginas["pagina_2"]:
    unlabeled = st.file_uploader("Unlabeled", accept_multiple_files=True, type='txt')
    modelo = st.file_uploader("Modelo")
    modelo = joblib.load(modelo)
    coleccion_unlabeled = generar_coleccion(unlabeled)
    predicciones = predecir_clases(modelo, coleccion_unlabeled)
    
    noticias = []
    for n in unlabeled:
        noticias.append(n.name)

    lista_odio = []
    for p in predicciones:
        if p == 'Odio':
            lista_odio.append('Si')
        else:
            lista_odio.append('No')

    resultados = {'Noticia': noticias, 'Odio': lista_odio}
    tabla_resultados = pd.DataFrame.from_dict(resultados)

    st.dataframe(tabla_resultados, width=500)
    st.write("")
    res_grafica = grafica_resultados(tabla_resultados, 0, 0)
    fig = px.pie(values=res_grafica.values(), names=res_grafica.keys(), title="<b><i>Resultados Clasificacion</b></i>")
    fig.update_traces(hovertemplate = "Clase: %{label} <br>Clasificados: %{value}</br>")
    st.plotly_chart(fig)
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")
    with col2:
        guardar = st.button("Guardar Resultados")
    with col3:
        guradar_resultados = st.radio("Guardar Resultados: ", ['CSV', 'Excel', 'Txt'])

    if guardar:
        dir_act = os.getcwd()
        if guradar_resultados == 'CSV':
            nombre_fichero = filedialog.asksaveasfile('w', parent=root, initialdir=dir_act, title='Seleccione una carpeta', defaultextension='.csv', filetypes=(('CSV', "*.csv"),))
            tabla_resultados.to_csv(nombre_fichero.name, encoding="utf8")
            st.success("Resultados Guardados correctamente")
        elif guradar_resultados == 'Excel':
            nombre_fichero = filedialog.asksaveasfile('w', parent=root, initialdir=dir_act, title='Seleccione una carpeta', defaultextension='.xlsx', filetypes=(('Excel', "*.xlsx"),))
            tabla_resultados.to_excel(nombre_fichero.name)
            st.success("Resultados Guardados correctamente") 
        elif guradar_resultados == 'Txt':
            nombre_fichero = filedialog.asksaveasfile('w', parent=root, initialdir=dir_act, title='Seleccione una carpeta', defaultextension='.txt', filetypes=(('Texto Plano', "*.txt"),))
            tabla_resultados.to_csv(nombre_fichero.name,encoding="utf8")
            st.success("Resultados Guardados correctamente")