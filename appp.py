import streamlit as st
import pandas as pd
import io

df_original = pd.read_excel("ANÁLISIS CALIDAD PROACTIVO - CANALES.xlsx", sheet_name="Hoja")
df_original['DNI'] = df_original['DNI'].astype(str)
df_original['MES'] = df_original['MES'].astype(str)
df_original['PAGO OCT'] = df_original['PAGO OCT'].astype(float)
df_original['HC'] = 1
df_original['RANGO_ORIGINAL'] = pd.qcut(df_original['QVENTAS'], q=10, duplicates='drop').apply(lambda x: f"[{int(x.left)}-{int(x.right)})")


def calcular_deciles(dataframe):
    dataframe = dataframe.copy()
    dataframe['RANGO_RCALC'] = pd.qcut(dataframe['QVENTAS'], q=10, duplicates='drop').apply(lambda x: f"[{int(x.left)}-{int(x.right)})")
    dataframe['DECIL'] = pd.qcut(dataframe['QVENTAS'], q=10, duplicates='drop', labels=False) + 1
    return dataframe


df = calcular_deciles(df_original)

st.subheader("Recalculo")
st.sidebar.title("Filtros iniciales")
opciones_filtro = df['RANGO_ORIGINAL'].unique().tolist()
filtros_deseleccionados = st.sidebar.multiselect(
    "Selecciona los rangos que **NO** quieres incluir en el cálculo",
    options=opciones_filtro,
    default=[]  
)
st.sidebar.info(
    """
    - Los filtros siempre muestran los **rangos originales de deciles**.
    - Después de aplicar un filtro, los deciles se recalculan dinámicamente.
    """
)
st.sidebar.info("Creado por Mafer Medina - 2024")

if filtros_deseleccionados:
    df_filtrado = df[~df['RANGO_ORIGINAL'].isin(filtros_deseleccionados)]
else:
    df_filtrado = df.copy()


if not df_filtrado.empty:
    df_recalculado = calcular_deciles(df_filtrado)
    df_recalculado = df_recalculado.sort_values(by=['TIPO', 'DECIL'])
else:
    df_recalculado = pd.DataFrame(columns=df.columns)


if not df_recalculado.empty:
    st.dataframe(df_recalculado, use_container_width=True)   
else:
    st.write("No hay datos disponibles después del filtro.")


############################ 

if not df_recalculado.empty:
    towrite_recalc = io.BytesIO()
    with pd.ExcelWriter(towrite_recalc, engine="xlsxwriter") as writer:
        df_recalculado.to_excel(writer, index=False, sheet_name="Datos Recalculados")
    towrite_recalc.seek(0)

    st.download_button(
        label="Descargar",
        data=towrite_recalc,
        file_name="datos_recalculados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


##########################

tabla_detallada = df_recalculado[['TIPO', 'DECIL', 'RANGO_RCALC' ,'DNI', 'HC', 'QVENTAS', 'Urs']]
tabla_detallada['URM2%'] = (tabla_detallada['Urs'] / tabla_detallada['QVENTAS']) * 100
st.subheader("Tabla Limpia")
st.dataframe(tabla_detallada, use_container_width=True)


towrite = io.BytesIO()
with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
    tabla_detallada.to_excel(writer, index=False, sheet_name="Tabla")
towrite.seek(0)

st.download_button(
    label="Descargar",
    data=towrite,
    file_name="tabla.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


##############################
if not df_recalculado.empty:
    st.subheader("Tabla Pivot")
    
    # Crear tabla pivote
    tabla_pivote = df_recalculado.pivot_table(
        index=['TIPO', 'RANGO_RCALC'],
        values=['HC', 'QVENTAS', 'Urs'],
        aggfunc={
            'HC': 'sum',
            'QVENTAS': 'sum',
            'Urs': 'sum'
        },
        
    ).reset_index()
    
    tabla_pivote['URM2%'] = (tabla_pivote['Urs'] / tabla_pivote['QVENTAS']) * 100
    st.dataframe(tabla_pivote, use_container_width=True)
    


towrite_pivote = io.BytesIO()
with pd.ExcelWriter(towrite_pivote, engine="xlsxwriter") as writer:
    tabla_pivote.to_excel(writer, index=False, sheet_name="Tabla Pivote")
towrite_pivote.seek(0)

st.download_button(
    label="Descargar",
    data=towrite_pivote,
    file_name="tabla_pivot.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


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

