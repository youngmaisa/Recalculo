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

    dataframe["URM2%"] = (dataframe["Urs"] / dataframe['QVENTAS'])*100

    return dataframe


# deciles iniciales y RANGO RCALC
df = calcular_deciles(df_original)



st.title("Recalculo de Deciles")
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


st.markdown(
    """
    <style>
    .block-container {
        padding: 2rem;
        max-width: 95%; 
        max-height: 95%
    }
    </style>
    """,
    unsafe_allow_html=True
)


if not df_recalculado.empty:
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        df_recalculado.to_excel(writer, index=False, sheet_name="Datos Recalculados")
    towrite.seek(0)

    st.download_button(
        label="Descargar como Excel",
        data=towrite,
        file_name="datos_recalculados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )