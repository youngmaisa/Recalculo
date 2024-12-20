import streamlit as st
import pandas as pd
import numpy as np
import io
from PIL import Image

img = Image.open('entel.jpg')
st.set_page_config(page_title='Recalculo Entel', page_icon=img, layout='wide')

df_original = pd.read_excel('archivos/NUEVA BASE - PROACTIVO OCTUBRE.xlsx')
df_original['DNI'] = df_original['DNI'].astype(str)
df_original['MES'] = df_original['MES'].astype(str)
df_original['HC'] = 1
df_original['URM2%'] = (df_original['Urs'] / df_original['QVENTAS']) * 100
#df_original['QNP%'] = (df_original['FLAG 30'] / df_original['Q JUL']) * 100
#df_original['QNP%'] = df_original['QNP%'].fillna(0)

###### FUNCION CALCULO DE DECILES
def calcular_deciles_personalizados(dataframe, columnas_orden):
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by=columnas_orden, ascending=[False] * len(columnas_orden)).reset_index(drop=True)
    
    total_filas = len(dataframe)
    tam_grupo = total_filas // 10
    restos = total_filas % 10  # Filas adicionales 
    
    # deciles
    grupos = []
    for decil in range(1, 11):
        tam_actual = tam_grupo + (1 if decil <= restos else 0)  # Agrega 1 fila extra- primeros 
        grupos.extend([decil] * tam_actual)
    
    dataframe['DECIL'] = grupos[:total_filas]
    return dataframe

df = calcular_deciles_personalizados(df_original, columnas_orden= ['Urs', 'URM2%'])
df = df[['MES','DEPARTAMENTO', 'SSNN FINAL', 'TIPO', 'DNI', 'Urs', 'DECIL', 'QVENTAS', 'URM2%', 'HC', 'SS']]

######  VACIO EL SIDEBAR
st.sidebar.empty()  # Vaciar el sidebar

######  TITULO NAVEGACION EN EL SIDEBAR
st.sidebar.title("Navegación")

######  FILTROS EN LA PÁGINA PRINCIPAL
#st.title("Recalculo - Entel")

# Filtros en la misma página
st.subheader("Filtros")

# Filtro de meses (Selección única)
opciones_filtros_mes = sorted(df['MES'].unique())
filtros_seleccionados_mes = st.selectbox(
    "MES",
    options=opciones_filtros_mes,
    index=0  # Valor predeterminado (primer mes en la lista)
)

opciones_filtros_subcanal = sorted(df['TIPO'].unique())
filtros_seleccionados_subcanal = st.multiselect(
    "Selecciona Subcanal",
    options=opciones_filtros_subcanal,
    default=[]  
)

opciones_filtros_departamento = sorted(df['DEPARTAMENTO'].unique())
filtros_seleccionados_departamento = st.multiselect(
    "Selecciona Departamento",
    options=opciones_filtros_departamento,
    default=[]  
)

min_val = df["QVENTAS"].min()
max_val = df["QVENTAS"].max()
filtros_seleccionados = st.slider(
    "Selecciona el rango de ventas:",
    min_value=int(min_val),
    max_value=int(max_val),
    value=(int(min_val), int(max_val))
)

######  FILTROS APLICADOS
df_filtrado = df.copy()
if filtros_seleccionados_subcanal:
    df_filtrado = df_filtrado[df_filtrado['TIPO'].isin(filtros_seleccionados_subcanal)]

if filtros_seleccionados:
    df_filtrado = df_filtrado[ 
        (df_filtrado["QVENTAS"] >= filtros_seleccionados[0]) & 
        (df_filtrado["QVENTAS"] <= filtros_seleccionados[1])
    ]

if filtros_seleccionados_departamento:
    df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'].isin(filtros_seleccionados_departamento)]

###### RECALCULO DE DECILES
if not df_filtrado.empty:
    df_recalculado = calcular_deciles_personalizados(df_filtrado, columnas_orden= ['Urs', 'URM2%'])
    df_recalculado['URM2%'] = (df_recalculado['Urs'] / df_recalculado['QVENTAS']) * 100
    #df_recalculado['QNP%'] = (df_recalculado['FLAG 30'] / df_recalculado['Q JUL']) * 100
    #df_recalculado['QNP%'] = df_recalculado['QNP%'].fillna(0)
else:
    df_recalculado = pd.DataFrame(columns=df.columns)

###### RESUMEN INICIAL
if not df_recalculado.empty:
    st.subheader("Resumen Inicial")
    tabla_pivote = df_recalculado.groupby('TIPO').agg(
        QHc=('HC', 'sum'),
        QUrs=('Urs', 'sum'),
        Qventas=('QVENTAS', 'sum'),
        SS=('SS', 'sum'),
    ).reset_index()
    
    tabla_pivote['URM2%'] = round((tabla_pivote['QUrs'] / tabla_pivote['Qventas']) * 100,1)
    #tabla_pivote['QNP%'] = round((tabla_pivote['QFlag30'] / tabla_pivote['SS']) * 100,1)
    #tabla_pivote['QNP%'] = tabla_pivote['QNP%'].fillna(0)  # Manejar valores NaN
    tabla_pivote = tabla_pivote.sort_values(by='QHc', ascending=False).reset_index(drop=True)

    total_row = pd.DataFrame({
        'TIPO': ['Total'],
        'QHc': [tabla_pivote['QHc'].sum()],
        'Qventas': [tabla_pivote['Qventas'].sum()],
        'QUrs': [tabla_pivote['QUrs'].sum()],
        'URM2%': [round((tabla_pivote['QUrs'].sum() / tabla_pivote['Qventas'].sum()) * 100, 1) 
                if tabla_pivote['Qventas'].sum() > 0 else 0],
        'SS': [tabla_pivote['SS'].sum()],
        #'QNP%': [round((tabla_pivote['QFlag30'].sum() / tabla_pivote['SS'].sum()) * 100, 1) 
         #       if tabla_pivote['SS'].sum() > 0 else 0]
    })
    
    tabla_pivote = pd.concat([tabla_pivote, total_row], ignore_index=True)
    tabla_pivote = tabla_pivote[['TIPO', 'QHc', 'Qventas', 'QUrs', 'URM2%', 'SS']]
    tabla_pivote = tabla_pivote.rename(columns={
        'TIPO': 'Subcanal',
        'QHc': 'QHc',
        'Qventas': 'QVentas',
        'QUrs': 'QUrs',
        'URM2%': 'URM2%',
        'SS': 'SS'
       # 'QNP%': 'QNP%'
    })

    st.dataframe(tabla_pivote)

# Puedes seguir con el resto del código de resumen de deciles, detalles de tabla, y diseño tal como lo tenías anteriormente


######  INFO DECILES
def resumen_deciles(dataframe, decil_col):
    resumen = dataframe.groupby(decil_col).agg(
        QHc=('HC', 'sum'),
        QUrs=('Urs', 'sum'),
        QVentas=('QVENTAS', 'sum'),
        SS=('SS', 'sum')
    ).reset_index().rename(columns={decil_col: 'Decil'})

    resumen['URM2%'] = round((resumen['QUrs'] / resumen['QVentas']) * 100,1)
   # resumen['QNP%'] = round((resumen['QFlag30'] / resumen['SS']) * 100,1)
   # resumen['QNP%'] = resumen['QNP%'].fillna(0)  


    totales = pd.DataFrame({
        'Decil': ['Total'],
        'QHc': [resumen['QHc'].sum()],
        'QVentas': [resumen['QVentas'].sum()],
        'QUrs': [resumen['QUrs'].sum()],
        'URM2%': [round((resumen['QUrs'].sum() / resumen['QVentas'].sum()) * 100,1) if resumen['QVentas'].sum() > 0 else 0],
        'SS': [resumen['SS'].sum()]
        #'QNP%': [round((resumen['QFlag30'].sum() / resumen['SS'].sum()) * 100,1) if resumen['SS'].sum() > 0 else 0]
    })

    resumen = pd.concat([resumen, totales], ignore_index=True)

    resumen = resumen[['Decil', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS']]

    return resumen


st.subheader("Deciles")
if not df_recalculado.empty:
    resumen_recalculado = resumen_deciles(df_recalculado, 'DECIL')
    st.dataframe(resumen_recalculado)



######  DATAFRAME RECALCULADO
st.subheader("Tabla detalle DNI")
if not df_recalculado.empty:

    tabla_renombrada2 = df_recalculado.rename(columns={
        'DEPARTAMENTO': 'Departamento',
        'SSNN FINAL': 'Socio',
        'TIPO': 'Subcanal',
        'DNI': 'DNI',
        'DECIL': 'DECIL',
        'QVENTAS': 'QVentas',
        'Urs': 'QUrs',
        'URM2%': 'URM2%',
        'SS': 'SS',
    })
    

    df_descarga = tabla_renombrada2[['Departamento', 'Socio', 'Subcanal', 'DNI', 'DECIL', 'QVentas', 'QUrs', 'URM2%', 'SS']]
    st.dataframe(df_descarga)
    
    # Descarga
    towrite_detallada = io.BytesIO()
    with pd.ExcelWriter(towrite_detallada, engine="xlsxwriter") as writer:
        df_descarga.to_excel(writer, index=False, sheet_name="Tabla Recalculada")
    towrite_detallada.seek(0)

    st.download_button(
        label="Descargar",
        data=towrite_detallada,
        file_name="dataframe_recalculado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.write("No hay datos disponibles después del filtro.")

###### DISEÑO 
st.markdown(
    """
    <style>
    .block-container {
        padding: 50px;
        max-width: 95%
    }
    </style>

    """,
    unsafe_allow_html=True
)

