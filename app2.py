import streamlit as st
import pandas as pd
import numpy as np
import io


df_original = pd.read_excel("ANÁLISIS CALIDAD PROACTIVO - CANALES.xlsx", sheet_name="Hoja")
df_original['DNI'] = df_original['DNI'].astype(str)
df_original['MES'] = df_original['MES'].astype(str)
df_original['HC'] = 1
df_original['URM2%'] = (df_original['Urs'] / df_original['QVENTAS']) * 100
df_original['QNP%'] = (df_original['FLAG 30'] / df_original['Q JUL']) * 100
df_original['QNP%'] = df_original['QNP%'].fillna(0)


def calcular_deciles_personalizados(dataframe, columnas_orden):
    
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by=columnas_orden, ascending=[False] * len(columnas_orden)).reset_index(drop=True)
    
    
    total_filas = len(dataframe)
    tam_grupo = total_filas // 10
    restos = total_filas % 10  # Filas adicionales que no encajan exactamente
    
    # deciles
    grupos = []
    for decil in range(1, 11):
        tam_actual = tam_grupo + (1 if decil <= restos else 0)  # Agregar 1 fila extra a los primeros "restos" grupos
        grupos.extend([decil] * tam_actual)
    
    dataframe['DECIL'] = grupos[:total_filas]
    return dataframe

df = calcular_deciles_personalizados(df_original, columnas_orden= ['Urs', 'URM2%'])
df = df[['DEPARTAMENTO', 'SSNN FINAL', 'TIPO', 'DNI', 'Urs', 'DECIL', 'QVENTAS', 'URM2%', 'QNP%', 'HC', 'Q JUL', 'FLAG 30']]

st.subheader("DataFrame Original")
st.dataframe(df, use_container_width=True)


# resumen deciles iniciales
def resumen_deciles(dataframe, decil_col):
    resumen = dataframe.groupby(decil_col).agg(
        Cantidad=('HC', 'sum'),
        Promedio_Urs=('Urs', 'mean')
    ).reset_index().rename(columns={decil_col: 'Decil'})
    return resumen

st.subheader("Resumen de Deciles Originales")
resumen_original = resumen_deciles(df, 'DECIL')
st.dataframe(resumen_original, use_container_width=True)



# filtros
st.sidebar.title("Filtros iniciales")
opciones_filtros_subcanal = sorted(df['TIPO'].unique())
filtros_seleccionados_subcanal = st.sidebar.multiselect(
    "Selecciona el subcanal que **SI** quieres incluir en el cálculo",
    options=opciones_filtros_subcanal,
    default=[]  
)

min_val = df["QVENTAS"].min()
max_val = df["QVENTAS"].max()
filtros_seleccionados = st.sidebar.slider(
    "Selecciona el rango de ventas (QVENTAS):",
    min_value=int(min_val),
    max_value=int(max_val),
    value=(int(min_val), int(max_val))
)

# filtros aplicación
df_filtrado = df.copy()
if filtros_seleccionados_subcanal:
    df_filtrado = df_filtrado[df_filtrado['TIPO'].isin(filtros_seleccionados_subcanal)]

if filtros_seleccionados:
    df_filtrado = df_filtrado[
        (df_filtrado["QVENTAS"] >= filtros_seleccionados[0]) &
        (df_filtrado["QVENTAS"] <= filtros_seleccionados[1])
    ]


# Recalculo deciles
if not df_filtrado.empty:
    df_recalculado = calcular_deciles_personalizados(df_filtrado, columnas_orden= ['Urs', 'URM2%'])
    df_recalculado['URM2%'] = (df_recalculado['Urs'] / df_recalculado['QVENTAS']) * 100
    df_recalculado['QNP%'] = (df_recalculado['FLAG 30'] / df_recalculado['Q JUL']) * 100
    df_recalculado['QNP%'] = df_recalculado['QNP%'].fillna(0)
   
else:
    df_recalculado = pd.DataFrame(columns=df.columns)
   

#  DataFrame recalculado
st.subheader("DataFrame Recalculado con Filtros Aplicados")
if not df_recalculado.empty:
    st.dataframe(df_recalculado, use_container_width=True)
else:
    st.write("No hay datos disponibles después del filtro.")


# Descarga  DataFrame recalculado
towrite_detallada = io.BytesIO()
with pd.ExcelWriter(towrite_detallada, engine="xlsxwriter") as writer:
    df_recalculado.to_excel(writer, index=False, sheet_name="Tabla Recalculada")
towrite_detallada.seek(0)

st.download_button(
    label="Descargar DataFrame Recalculado",
    data=towrite_detallada,
    file_name="dataframe_recalculado.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# Resumen deciles recalculados
st.subheader("Resumen de Deciles Recalculados")
if not df_recalculado.empty:
    resumen_recalculado = resumen_deciles(df_recalculado, 'DECIL')
    st.dataframe(resumen_recalculado, use_container_width=True)


# tabla pivote
if not df_recalculado.empty:
    st.subheader("Tabla Pivot")
    tabla_pivote = df_recalculado.pivot_table(
        index=['TIPO', 'DECIL'],
        values=['HC', 'Urs', 'QVENTAS', 'FLAG 30', 'Q JUL'],
        aggfunc='sum'
    ).reset_index()

    tabla_pivote['URM2%'] = (tabla_pivote['Urs'] / tabla_pivote['QVENTAS']) * 100
    tabla_pivote['QNP%'] = (tabla_pivote['FLAG 30'] / tabla_pivote['Q JUL']) * 100
    tabla_pivote['QNP%'] = tabla_pivote['QNP%'].fillna(0)

    #  totales
    totales = tabla_pivote[['QVENTAS', 'HC', 'Urs', 'FLAG 30', 'Q JUL']].sum()
    totales['TIPO'] = 'Total'
    totales['DECIL'] = ''
    totales['URM2%'] = (totales['Urs'] / totales['QVENTAS']) * 100 if totales['QVENTAS'] != 0 else 0
    totales['QNP%'] = (totales['FLAG 30'] / totales['Q JUL']) * 100 if totales['Q JUL'] != 0 else 0
    tabla_pivote = tabla_pivote[['TIPO','DECIL', 'HC', 'Urs', 'QVENTAS', 'URM2%', 'QNP%', 'Q JUL', 'FLAG 30']]
    tabla_pivote = pd.concat([tabla_pivote, totales.to_frame().T], ignore_index=True)
    st.dataframe(tabla_pivote, use_container_width=True)

    # Descarga tabla pivote
    towrite_pivote = io.BytesIO()
    with pd.ExcelWriter(towrite_pivote, engine="xlsxwriter") as writer:
        tabla_pivote.to_excel(writer, index=False, sheet_name="Tabla Pivot")
    towrite_pivote.seek(0)

    st.download_button(
        label="Descargar Tabla Pivot",
        data=towrite_pivote,
        file_name="tabla_pivot.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

