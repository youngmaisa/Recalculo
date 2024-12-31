import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
import io
import traceback

from concurrent.futures import ProcessPoolExecutor



carpeta_canal  = 'data/RECEPTIVO_2024'

columna_DNI = 'DNI LIDER'

subcanales = ['DESARROLLADOR', 'HC EMO MERCADO', 'FULL PREPAGO DD',
            'DAE', 'FULL PREPAGO DAE', 'HC MONOMARCA', 'HC EMO']


departamentos = ['PUNO', 'TUMBES', 'UCAYALI', 'HUANUCO', 'TACNA', 'AMAZONAS', 'LIMA', 
                'CUSCO', 'APURIMAC', 'CAJAMARCA', 'MADRE DE DIOS', 'LORETO', 'ANCASH', 
                'PIURA', 'ICA', 'MOQUEGUA', 'SAN MARTIN', 'AREQUIPA','LAMBAYEQUE', 
                'HUANCAVELICA','LA LIBERTAD', 'JUNIN', 'PASCO', 'AYACUCHO']

kams = ['ALEXANDRA TABOADA', 'KARINA TORRES', 'GERSON RIMACH', 'ROMINNA PINATTE',
        'REYNA CABANILLAS', 'MARIA MARTINEZ', 'RUBEN SAMANEZ', 'JORGE PUELLES', 'NATHALY REYES']


socios = [ 'FANERO', 'D&D', 'GRUPO CAYAO', 'JOKATEL', 'PBD', 'POWER TIMBER', 'DISCOMTECH']


meses_orden = [
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", 
    "JULIO", "AGOSTO", "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
]


tab_vista_normal, tab_vista_historica = \
    st.tabs(['VISTA NORMAL', 'VISTA HISTORICA'])




def calcular_grupos_personalizados(dataframe, num_grupos, columnas_orden):
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by=columnas_orden, ascending=[False] * len(columnas_orden)).reset_index(drop=True)
    
    total_filas = len(dataframe)
    tam_grupo = total_filas // num_grupos  
    restos = total_filas % num_grupos
    
    grupos = []
    for grupo in range(1, num_grupos + 1):
        tam_actual = tam_grupo + (1 if grupo > num_grupos - restos else 0)  # Asignar el resto a los últimos grupos
        grupos.extend([grupo] * tam_actual)

    dataframe['Grupo'] = grupos[:total_filas]

    
    limites = dataframe.groupby('Grupo')[columnas_orden[0]].agg(['min', 'max'])

    dataframe['RangoGrupo'] = dataframe['Grupo'].map(
        lambda grupo: f"{columnas_orden[0]}: ({limites.loc[grupo, ('max')]}-{limites.loc[grupo, ('min')]})"
    )
    
    return dataframe



def archivos_listados(carpeta_archivos):
    return [archivo for archivo in os.listdir(carpeta_archivos) if archivo.endswith('.xlsx')]
    

def meses_disponibles(carpeta_archivos):
    archivos = archivos_listados(carpeta_archivos)
    meses = [archivo.split('_')[-1].replace('.xlsx', '') for archivo in archivos if archivo.split('_')[-1].replace('.xlsx', '').isdigit()]
    return sorted(meses)


def ordenar_meses(meses):
    return sorted(
        [mes for mes in meses if mes.upper() in meses],  
        key=lambda mes: meses_orden.index(mes.upper())
    )


def archivo_mes_ruta(mes, carpeta_archivos):
    archivos = archivos_listados(carpeta_archivos)
    for archivo in archivos:
        if mes in archivo:
            return os.path.join(carpeta_archivos, archivo)
    return None


@st.cache_data
def load_data(ruta, columna_DNI):
    data = pd.read_excel(ruta,dtype={columna_DNI: str})
    return data




def dataframe_mes(mes, carpeta_archivos):

    archivo_path = archivo_mes_ruta(mes, carpeta_archivos)

    #df_original = load_data(archivo_path, columna_DNI=columna_DNI)   

    df_original = pd.read_excel(archivo_path, dtype={columna_DNI: str}) 

    df_original['MES'] = df_original['MES'].astype(str)
    df_original['HC'] = 1
    df_original['URM2%'] = round((df_original['Urs'] / df_original['QVENTAS']) * 100,2)
    df_original['PagoTotal'] = (df_original['ACELERADOR'] + df_original['PLANILLA'] + df_original['BONO']+ df_original['CAMPAÑA']+ df_original['OTROS'])
    
    df_original = df_original.rename(columns={
            'MES': 'Mes',
            'DEPARTAMENTO': 'Departamento',
            'SSNN FINAL': 'Socio',
            'KAM VF': 'Kam',
            'HC': 'QHc',
            'TIPO': 'Subcanal',
            columna_DNI: columna_DNI,
            'GRUPO': 'Grupo',
            'QVENTAS': 'PP',
            'Urs': 'QUrs',
            'PLANILLA': 'Planilla',
            'ACELERADOR': 'Acelerador',
            'PLANILLA': 'Planilla',
            'ACELERADOR': 'Acelerador',
            'BONO': 'Bono',
            'CAMPAÑA': 'Campaña',
            'OTROS': 'Otros'
    })

    return df_original
    


def socios_disponibles(carpeta_archivos):
    socios = set()
    meses = meses_disponibles(carpeta_archivos) 
    for mes in meses:
        df = dataframe_mes(mes, carpeta_archivos)
        socios.update(df['Socio'].unique())  
    return socios

def rango_ventas(carpeta_archivos):
    archivos = os.listdir(carpeta_archivos)
    archivos = [os.path.join(carpeta_archivos, archivo) for archivo in archivos]
    ventas_min = float('inf')
    ventas_max = float('-inf')
    for archivo in archivos:
        try:
            df_mes = pd.read_excel(archivo)  
            if 'QVENTAS' in df_mes.columns:  
                ventas_min = min(ventas_min, df_mes['QVENTAS'].min())
                ventas_max = max(ventas_max, df_mes['QVENTAS'].max())
        except Exception as e:
            print(f"Error procesando el archivo {archivo}: {e}")
    if ventas_min == float('inf') or ventas_max == float('-inf'):
        return None, None  
    return ventas_min, ventas_max





with st.sidebar:

    num_grupos = st.number_input("Número de grupos", min_value=2, max_value=50, value=10)

    filtro_subcanal = st.multiselect(
        "Subcanal",
        options= subcanales,
        default=[]  
    )

    filtro_departamento = st.multiselect(
        "Departamento",
        options= departamentos,
        default=[]  
    )

    filtro_kam = st.multiselect(
        "Kam",
        options= kams,
        default=[]  
    )

    filtro_socio = st.multiselect(
        "Socio",
        options= socios,
        default=[]  
    )

    min_val, max_val =   0, 5000
    min_input = st.number_input(
        "Ingrese el valor mínimo de ventas:",
        min_value=int(min_val),
        max_value=int(max_val),
        value=int(min_val),
        step=1
    )
    max_input = st.number_input(
        "Ingrese el valor máximo de ventas:",
        min_value=int(min_val),
        max_value=int(max_val),
        value=int(max_val),
        step=1
    )

    if min_input > max_input:
        st.error("El valor mínimo no puede ser mayor que el valor máximo.")



def normal(carpeta_archivos=carpeta_canal):
    filtro_mes = st.selectbox("Mes",  meses_disponibles(carpeta_archivos))
    st.markdown("---")
    try:
        df = dataframe_mes(filtro_mes, carpeta_archivos)

        df = calcular_grupos_personalizados(dataframe= df,num_grupos= num_grupos, columnas_orden=['URM2%', 'QUrs', 'PP'])
    
        df_filtrado = df.copy()

        if filtro_subcanal:
            df_filtrado = df_filtrado[df_filtrado['Subcanal'].isin(filtro_subcanal)]

        if pd.notnull(min_input) and pd.notnull(max_input):
            df_filtrado = df_filtrado[
                (df_filtrado["PP"] >= min_input) & 
                (df_filtrado["PP"] <= max_input)
            ]

        if filtro_departamento:
            df_filtrado = df_filtrado[df_filtrado['Departamento'].isin(filtro_departamento)]

        if filtro_kam:
            df_filtrado = df_filtrado[df_filtrado['Kam'].isin(filtro_kam)]

        if filtro_socio:
            df_filtrado = df_filtrado[df_filtrado['Socio'].isin(filtro_socio)]


        # RECALCULO         
        if not df_filtrado.empty:
            df_recalculado = calcular_grupos_personalizados(df_filtrado, num_grupos= num_grupos , columnas_orden= ['URM2%', 'QUrs', 'PP'])
            df_recalculado['URM2%'] = round((df_recalculado['QUrs'] / df_recalculado['PP']) * 100,2)
        else:
            df_recalculado = pd.DataFrame(columns=df.columns)


        if not df_recalculado.empty:


            #  RESUMEN INICIAL --------------------------------------------------------------------
            st.markdown("""### :page_facing_up: Resumen Inicial""")
            st.info("""Puede visualizar un **dataframe segmentado por subcanal**, si deseas el detalle del **PagoTotal**, simplemente haz clic  en **"Mostrar adicionales"**.""")


            tabla_pivote = df_recalculado.groupby('Subcanal').agg(
                QHc=('QHc', 'sum'),
                QUrs=('QUrs', 'sum'),
                PP=('PP', 'sum'),
                PagoTotal = ('PagoTotal', 'sum'),
                Acelerador = ('Acelerador', 'sum'),
                Planilla = ('Planilla', 'sum'),
                Bono = ('Bono', 'sum'),
                Campaña = ('Campaña', 'sum'),
                Otros = ('Otros', 'sum')
            ).reset_index()
            
            tabla_pivote['URM2%'] = (tabla_pivote['QUrs'] / tabla_pivote['PP']) * 100
        
            tabla_pivote = tabla_pivote.sort_values(by='QHc', ascending=False).reset_index(drop=True)

            total_row = pd.DataFrame({
                'Subcanal': ['Total'],
                'QHc': [tabla_pivote['QHc'].sum()],
                'PP': [tabla_pivote['PP'].sum()],
                'QUrs': [tabla_pivote['QUrs'].sum()],
                'URM2%': [round((tabla_pivote['QUrs'].sum() / tabla_pivote['PP'].sum()) * 100, 1) 
                        if tabla_pivote['PP'].sum() > 0 else 0],
                'PagoTotal': [tabla_pivote['PagoTotal'].sum()],
                'Acelerador':  [tabla_pivote['Acelerador'].sum()],
                'Planilla': [tabla_pivote['Planilla'].sum()],
                'Bono':  [tabla_pivote['Bono'].sum()],
                'Campaña':  [tabla_pivote['Campaña'].sum()],
                'Otros':  [tabla_pivote['Otros'].sum()],
            
            })
                            
            tabla_pivote = pd.concat([tabla_pivote, total_row], ignore_index=True)
            
            mostrar_columnas_adicionales_pivote = st.checkbox("Mostrar adicionales .")

            if mostrar_columnas_adicionales_pivote:
                columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'PP', 'QUrs','URM2%', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']
            else:
                columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'PP', 'QUrs', 'URM2%', 'PagoTotal'] 
                
            st.dataframe(tabla_pivote[columnas_a_mostrar_pivote], use_container_width=True)
            st.markdown("---")

            # --------------------------------------------------------------------------------------


            # TABLA INFO GRUPOS --------------------------------------------------------------------
            st.markdown("""### :rocket: Grupos""")
            st.info("""
            Puedes visualizar un **dataframe segmentado por grupos calculados**, el cual muestra la **distribución de QHcs** dentro de cada grupo resultante del cálculo. Si deseas visualizar el detalle del **Pago Total**, simplemente haz clic en la opción **"Mostrar adicionales"**.
            """)

            def resumen_deciles(dataframe, decil_col):
                resumen = dataframe.groupby(decil_col).agg(
                    RangoGrupo= ('RangoGrupo', 'first'),
                    QHc=('QHc', 'sum'),
                    QUrs=('QUrs', 'sum'),
                    PP=('PP', 'sum'),
                    PagoTotal = ('PagoTotal', 'sum'),
                    Acelerador = ('Acelerador', 'sum'),
                    Planilla = ('Planilla', 'sum'),
                    Bono = ('Bono', 'sum'),
                    Campaña = ('Campaña', 'sum'),
                    Otros = ('Otros', 'sum')
                ).reset_index().rename(columns={decil_col: 'Grupo'})

                resumen['URM2%'] = (resumen['QUrs'] / resumen['PP']) * 100

                totales = pd.DataFrame({
                    'Grupo': ['Total'],
                    'RangoGrupo': [' '],
                    'QHc': [resumen['QHc'].sum()],
                    'PP': [resumen['PP'].sum()],
                    'QUrs': [resumen['QUrs'].sum()],
                    'URM2%': [round((resumen['QUrs'].sum() / resumen['PP'].sum()) * 100,1) if resumen['PP'].sum() > 0 else 0],
                    'PagoTotal': [resumen['PagoTotal'].sum()],
                    'Acelerador':  [resumen['Acelerador'].sum()],
                    'Planilla': [resumen['Planilla'].sum()],
                    'Bono':  [resumen['Bono'].sum()],
                    'Campaña':  [resumen['Campaña'].sum()],
                    'Otros':  [resumen['Otros'].sum()],
                })

                resumen = pd.concat([resumen, totales], ignore_index=True)

                resumen = resumen[['Grupo', 'RangoGrupo','QHc', 'PP', 'QUrs', 'URM2%', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']]

                return resumen
            
        
            resumen_recalculado = resumen_deciles(df_recalculado, 'Grupo')

            mostrar_columnas_adicionales_resumen = st.checkbox("Mostrar adicionales ..")

            if mostrar_columnas_adicionales_resumen:
                columnas_a_mostrar_resumen = ['Grupo', 'RangoGrupo','QHc', 'PP', 'QUrs', 'URM2%',  'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                
            else:
                columnas_a_mostrar_resumen = ['Grupo', 'RangoGrupo', 'QHc', 'PP', 'QUrs', 'URM2%', 'PagoTotal']  
                
            st.dataframe(resumen_recalculado[columnas_a_mostrar_resumen], use_container_width=True)


            # descarga
            towrite_resumen = io.BytesIO()
            with pd.ExcelWriter(towrite_resumen, engine="xlsxwriter") as writer:
                resumen_recalculado.to_excel(writer, index=False, sheet_name="Tabla Recalculada")
            towrite_resumen.seek(0)

            st.download_button(
                label="Descargar grupos",
                data=towrite_resumen,
                file_name="dataframe_grupos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


            st.markdown("---")

            # --------------------------------------------------------------------------------------


            # TABLA RECALCULADA --------------------------------------------------------------------
            st.markdown(f"""### :mag: Tabla detalle {columna_DNI}""")
            st.info("""Puedes visualizar un **dataframe detallado** con los grupos calculados, mostrando los **DNIs asociados a cada grupo**.  Si deseas observar el detalle del **Pago Total**, simplemente haz clic en la opción **"Mostrar adicionales"**.""")

            #df_descarga = df_recalculado.sort_values(by=['URM2%'], ascending=[False])
            df_descarga = df_recalculado

            filtro1, filtro2 = st.columns([2,2])
            with filtro1:
                grupos_disponibles = df_recalculado['Grupo'].unique()
                grupos_seleccionados = st.multiselect(
                    "Selecciona uno o varios grupos",
                    options= grupos_disponibles,  
                    default= []  
                )
            with filtro2:
                dni_ingresado = st.text_input(f"Ingresa un {columna_DNI}:")


            if grupos_seleccionados:
                df_descarga = df_recalculado[df_recalculado['Grupo'].isin(grupos_seleccionados)]
            else:
                df_descarga = df_recalculado

            mostrar_columnas_adicionales = st.checkbox("Mostrar adicionales ...")

            if mostrar_columnas_adicionales:
                columnas_a_mostrar = ['Departamento', 'Socio', 'Kam' ,'Subcanal', columna_DNI, 'Grupo', 'PP', 'QUrs', 'URM2%', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                
            else:
                columnas_a_mostrar = ['Departamento', 'Socio', 'Kam', 'Subcanal', columna_DNI, 'Grupo', 'PP', 'QUrs', 'URM2%', 'PagoTotal']  
                
            
            if dni_ingresado: 
                try:
                    dni_ingresado = str(dni_ingresado)  
                    df_descarga = df_descarga[df_descarga[columna_DNI] == dni_ingresado]

                    if df_descarga.empty:  
                        st.warning(f"El {columna_DNI} ingresado no se encuentra en los datos.")
                    else:
                        st.dataframe(df_descarga[columnas_a_mostrar], use_container_width=True)
                except ValueError:
                    st.error(f"Por favor, ingresa un {columna_DNI} válido (número entero).")
            else:  
                st.dataframe(df_descarga[columnas_a_mostrar], use_container_width=True )
            

            # descarga
            towrite_detallada = io.BytesIO()
            with pd.ExcelWriter(towrite_detallada, engine="xlsxwriter") as writer:
                df_descarga.to_excel(writer, index=False, sheet_name="Tabla Recalculada")
            towrite_detallada.seek(0)

            st.download_button(
                label="Descargar tabla detalle",
                data=towrite_detallada,
                file_name="dataframe_recalculado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        
        
        # --------------------------------------------------------------------------------------
        else:
            st.warning("No hay datos para los filtros seleccionados.")

        
    except Exception as e:
        st.write(f"Error al cargar el archivo: {e}")
        st.write("Detalles de la excepción:")
        st.text(traceback.format_exc())  



with tab_vista_normal:
    normal()
with tab_vista_historica:
    normal()