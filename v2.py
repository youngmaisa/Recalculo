import os
import pandas as pd
import streamlit as st

# Ruta de la carpeta local de OneDrive sincronizada
CARPETA_ONEDRIVE = r"C:\Users\mmedinam\OneDrive - Entel Peru S.A\Documentos - EntelDrive_ __ Canal Masivo\Archivos_Recalculo"

def listar_y_leer_archivos(carpeta):
    try:
        # Verificar la ruta de la carpeta
        st.write(f"Ruta de la carpeta: {carpeta}")
        
        archivos = os.listdir(carpeta)
        st.write(f"Archivos encontrados: {archivos}")  # Imprimir los archivos encontrados
        
        # Filtrar los archivos Excel (.xlsx)
        archivos_excel = [archivo for archivo in archivos if archivo.lower().endswith(".xlsx")]
        
        if archivos_excel:
            return archivos_excel
        else:
            return None
    except FileNotFoundError:
        st.error("La carpeta no existe. Verifica la ruta.")
        return None
    except Exception as e:
        st.error(f"Error al procesar los archivos: {e}")
        return None

def leer_archivo(archivo):
    try:
        ruta_completa = os.path.join(CARPETA_ONEDRIVE, archivo)
        # Leer el archivo Excel
        df = pd.read_excel(ruta_completa)
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo {archivo}: {e}")
        return None

# Interfaz de Streamlit
st.title("Lectura de Archivos de OneDrive Local")
st.write("Seleccione un archivo Excel para visualizar los datos:")

# Listar archivos Excel disponibles en la carpeta local
archivos_excel = listar_y_leer_archivos(CARPETA_ONEDRIVE)

if archivos_excel:
    # Mostrar los archivos Excel en un selector
    archivo_seleccionado = st.selectbox("Archivos disponibles", archivos_excel)
    
    # Leer y mostrar el contenido del archivo seleccionado
    if archivo_seleccionado:
        st.write(f"Mostrando datos del archivo: {archivo_seleccionado}")
        df = leer_archivo(archivo_seleccionado)
        if df is not None:
            st.dataframe(df)
else:
    st.write("No se encontraron archivos Excel en la carpeta.")
