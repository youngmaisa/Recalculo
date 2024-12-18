import streamlit as st
import pandas as pd
import io

# Cargar datos
df_original = pd.read_excel("ANÁLISIS CALIDAD PROACTIVO - CANALES.xlsx", sheet_name="Hoja")
df_original['DNI'] = df_original['DNI'].astype(str)
df_original['MES'] = df_original['MES'].astype(str)
df_original['HC'] = 1

def calcular_deciles(dataframe):
    dataframe = dataframe.copy()
    dataframe['DECIL'] = pd.qcut(-dataframe['Urs'], q=10, duplicates='drop', labels=False) + 1
    return dataframe

df = calcular_deciles(df_original)


df['URM2%'] = (df['Urs'] / df['QVENTAS']) * 100
df['QNP%'] = (df['FLAG 30'] / df['Q JUL']) * 100
df['QNP%'] = df['QNP%'].fillna(0)
df = df[['TIPO', 'DECIL' ,'DNI',  'Urs', 'QVENTAS', 'HC', 'FLAG 30', 'Q JUL', 'URM2%', 'QNP%']]

# Resumennn deciles originales
def resumen_deciles(dataframe, decil_col):
    resumen = dataframe.groupby(decil_col).agg(
        Cantidad=('HC', 'sum')
    ).reset_index().rename(columns={decil_col: 'Decil'})
    return resumen

st.subheader("Deciles originales")
if not df.empty:
    resumen_original = resumen_deciles(df, 'DECIL')
    st.dataframe(resumen_original, use_container_width=True)



# Filtros

st.sidebar.title("Filtros iniciales")

opciones_filtros_subcanal = sorted(df['TIPO'].unique())
filtros_seleccionados_subcanal = st.sidebar.multiselect(
    "Selecciona los subcanales que **SI** quieres incluir en el cálculo",
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

st.sidebar.info(
    """
    - Después de aplicar un filtro, los deciles se recalculan dinámicamente.
    """
)


# Filtrar datos según los filtros seleccionados
df_filtrado = df.copy()  # Por defecto, sin filtros

if filtros_seleccionados_subcanal:
    df_filtrado = df_filtrado[df_filtrado['TIPO'].isin(filtros_seleccionados_subcanal)]

if filtros_seleccionados:
    df_filtrado = df_filtrado[
        (df_filtrado["QVENTAS"] >= filtros_seleccionados[0]) &
        (df_filtrado["QVENTAS"] <= filtros_seleccionados[1])
    ]

# Recalcular deciles si hay datos filtrados
if not df_filtrado.empty:
    df_recalculado = calcular_deciles(df_filtrado)
    #df_recalculado = df_recalculado.sort_values(by=['DECIL', 'Urs'], ascending= True)
    df_recalculado = df_recalculado.sort_values(by=['DECIL', 'Urs'], ascending=[True, False])

else:
    df_recalculado = pd.DataFrame(columns=df.columns)


st.subheader("Recalculo")

# Mostrar DataFrame recalculado o inicial
if not df_recalculado.empty:
    tabla_detallada = df_recalculado
    tabla_detallada['URM2%'] = (tabla_detallada['Urs'] / tabla_detallada['QVENTAS']) * 100
    tabla_detallada['QNP%'] = (tabla_detallada['FLAG 30']/ tabla_detallada['Q JUL']) * 100
    tabla_detallada['QNP%'] = tabla_detallada['QNP%'].fillna(0)

    st.dataframe(tabla_detallada, use_container_width=True)
else:
    st.write("No hay datos disponibles después del filtro.")


towrite_detallada = io.BytesIO()
with pd.ExcelWriter(towrite_detallada, engine="xlsxwriter") as writer:
    tabla_detallada.to_excel(writer, index=False, sheet_name="Tabla Pivote")
towrite_detallada.seek(0)

st.download_button(
    label="Descargar",
    data=towrite_detallada,
    file_name="tabla_extendida.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)



## PIVOT TABLA
if not df_recalculado.empty:
    st.subheader("Tabla Pivot")
    
    # Crear tabla pivote
    tabla_pivote = df_recalculado.pivot_table(
        index=['TIPO', 'DECIL'],
        values=['HC','Urs', 'QVENTAS', 'FLAG 30', 'Q JUL'],
        aggfunc={
            'HC': 'sum',
            'Urs': 'sum',
            'QVENTAS': 'sum',
            'FLAG 30': 'sum',
            'Q JUL': 'sum'
        },
        
    ).reset_index()
    
    tabla_pivote['URM2%'] = (tabla_pivote['Urs'] / tabla_pivote['QVENTAS']) * 100
    tabla_pivote['QNP%'] =  (tabla_pivote['FLAG 30'] / tabla_pivote['Q JUL']) * 100
    tabla_pivote['QNP%'] = tabla_pivote['QNP%'].fillna(0)
    tabla_pivote = tabla_pivote[['TIPO', 'DECIL',  'QVENTAS', 'HC', 'Urs', 'URM2%', 'QNP%']]
    st.dataframe(tabla_pivote, use_container_width=True)

towrite_pivote = io.BytesIO()
with pd.ExcelWriter(towrite_pivote, engine="xlsxwriter") as writer:
    tabla_pivote.to_excel(writer, index=False, sheet_name="Tabla Pivote")
towrite_pivote.seek(0)

st.download_button(
    label="Descargar",
    data=towrite_pivote,
    file_name="tabla_resumida.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)




# Resumennn deciles recalculados
def resumen_deciles(dataframe, decil_col):
    resumen = dataframe.groupby(decil_col).agg(
        Cantidad=('HC', 'sum')
    ).reset_index().rename(columns={decil_col: 'Decil'})
    return resumen

st.subheader("Deciles recalculados")
if not df.empty:
    resumen_original = resumen_deciles(df_recalculado, 'DECIL')
    st.dataframe(resumen_original, use_container_width=True)



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