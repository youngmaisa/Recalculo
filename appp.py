import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import traceback  
import io

# CONFIGURACION INICIAL  ----------------------------------------------------------------

img = Image.open('entel.jpg')
st.set_page_config(page_title='Recalculo Entel', page_icon=img, layout='wide')
st.sidebar.title('Recalculo Entel')


tab_vista_normal, tab_vista_historica = \
    st.tabs(['VISTA NORMAL', 'VISTA HISTORICA'])


with st.sidebar:
    
        with st.expander("📖 Cómo usar este programa"):
            st.markdown("""
            ## Bienvenidos! 🌟
            
            Este programa está diseñado para ayudarte a analizar y dividir una población en grupos personalizados. 
            
            ### Funcionalidades principales:
            
            **1. Vista Normal**  
            - Calcula y muestra la segmentación de un mes según los filtros aplicados.
            - Los filtros disponibles son: **Mes**, **Nro de grupos**, **Subcanal**, **Departamento**, y un rango de **Ventas** definido por el usuario.
            - La segmentación del mes se actualiza dinámicamente de acuerdo con los filtros seleccionados.
            
            **2. Vista Histórica**  
            - Permite observar la evolución de los grupos a lo largo de los meses.
            - Permite visualizar un grafico según el DNI ingresado.
            - Los filtros de **Subcanal**,  **Nro de grupos**, **Departamento**, y **Ventas** afectan esta vista, excepto el filtro de mes, ya que se enfoca en consolidar datos de varios períodos.
          
            ### Detalles Adicionales 🛠️
            - Los grupos se ordenan desde el **Grupo 1** (mayor cantidad de URM2%, QUrs y QVentas) hacia abajo.
            """)





# FUNCIONES DE APOYO ----------------------------------------------------------------



def canales_disponibles():
    canales = ["PROACTIVO", "RECEPTIVO"]
    return canales


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
    

meses_orden = [
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", 
    "JULIO", "AGOSTO", "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
]

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



def dataframe_mes(mes, carpeta_archivos, canal):
    columna_DNI = ''
    if canal == "PROACTIVO":
        columna_DNI = 'DNI'
    else:
        columna_DNI = 'DNI LIDER'

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
            'QVENTAS': 'QVentas',
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
    

def subcanales_disponibles(carpeta_archivos, canal):
    subcanales = set()
    meses = meses_disponibles(carpeta_archivos)  
    for mes in meses:
        df = dataframe_mes(mes, carpeta_archivos, canal)
        subcanales.update(df['Subcanal'].unique())  
    return subcanales

def departamentos_disponibles(carpeta_archivos, canal):
    departamentos = set()
    meses = meses_disponibles(carpeta_archivos) 
    for mes in meses:
        df = dataframe_mes(mes, carpeta_archivos, canal)
        departamentos.update(df['Departamento'].unique())  
    return departamentos
 
def kams_disponibles(carpeta_archivos, canal):
    kams = set()
    meses = meses_disponibles(carpeta_archivos) 
    for mes in meses:
        df = dataframe_mes(mes, carpeta_archivos, canal)
        kams.update(df['Kam'].unique())  
    return kams

def socios_disponibles(carpeta_archivos, canal):
    socios = set()
    meses = meses_disponibles(carpeta_archivos) 
    for mes in meses:
        df = dataframe_mes(mes, carpeta_archivos, canal)
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


def definir_carpeta_canal(canal):
    carpeta = ""
    if canal == "PROACTIVO":
        carpeta = r"C:\Users\mmedinam\OneDrive - Entel Peru S.A\Documentos - EntelDrive_ __ Canal Masivo\Archivos_Recalculo\BASES\2024\PROACTIVO_2024"
    else:
        
        carpeta = r"C:\Users\mmedinam\OneDrive - Entel Peru S.A\Documentos - EntelDrive_ __ Canal Masivo\Archivos_Recalculo\BASES\2024\RECEPTIVO_2024"
    return carpeta


# FILTROS GLOBALES ----------------------------------------------------------------

with st.sidebar.expander("Filtros", expanded=True):

   
    filtro_canal = st.selectbox("Canal",  canales_disponibles())

    # VARIABLE GLOBAL * IMPORTANTE *
    carpeta_canal = definir_carpeta_canal(filtro_canal)

    num_grupos = st.number_input("Número de grupos", min_value=2, max_value=50, value=10)

    filtro_subcanal = st.multiselect(
        "Subcanal",
        options= subcanales_disponibles(carpeta_archivos=carpeta_canal, canal=filtro_canal),
        default=[]  
    )

    filtro_departamento = st.multiselect(
        "Departamento",
        options= departamentos_disponibles(carpeta_archivos=carpeta_canal, canal=filtro_canal),
        default=[]  
    )

    filtro_kam = st.multiselect(
        "Kam",
        options= kams_disponibles(carpeta_archivos=carpeta_canal, canal=filtro_canal),
        default=[]  
    )

    filtro_socio = st.multiselect(
        "Socio",
        options= socios_disponibles(carpeta_archivos=carpeta_canal, canal=filtro_canal),
        default=[]  
    )

    min_val, max_val = rango_ventas(carpeta_archivos=carpeta_canal)
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
    


def normal(carpeta_archivos=carpeta_canal, canal = filtro_canal):
    
    #st.write(vista)

    filtro_mes = st.selectbox("Mes",  meses_disponibles(carpeta_archivos=carpeta_canal))
    st.markdown("---")


    columna_DNI = ''
    if canal == "PROACTIVO":
        columna_DNI = "DNI"
    else:
        columna_DNI = "DNI LIDER"
        

    try:
        df = dataframe_mes(filtro_mes, carpeta_archivos, canal)

        df = calcular_grupos_personalizados(dataframe= df,num_grupos= num_grupos, columnas_orden=['URM2%', 'QUrs', 'QVentas'])
      
        df_filtrado = df.copy()

        if filtro_subcanal:
            df_filtrado = df_filtrado[df_filtrado['Subcanal'].isin(filtro_subcanal)]

        if pd.notnull(min_input) and pd.notnull(max_input):
            df_filtrado = df_filtrado[
                (df_filtrado["QVentas"] >= min_input) & 
                (df_filtrado["QVentas"] <= max_input)
            ]

        if filtro_departamento:
            df_filtrado = df_filtrado[df_filtrado['Departamento'].isin(filtro_departamento)]

        if filtro_kam:
            df_filtrado = df_filtrado[df_filtrado['Kam'].isin(filtro_kam)]

        if filtro_socio:
            df_filtrado = df_filtrado[df_filtrado['Socio'].isin(filtro_socio)]


        # RECALCULO         
        if not df_filtrado.empty:
            df_recalculado = calcular_grupos_personalizados(df_filtrado, num_grupos= num_grupos , columnas_orden= ['URM2%', 'QUrs', 'QVentas'])
            df_recalculado['URM2%'] = round((df_recalculado['QUrs'] / df_recalculado['QVentas']) * 100,2)
        else:
            df_recalculado = pd.DataFrame(columns=df.columns)


        if not df_recalculado.empty:


            #  RESUMEN INICIAL --------------------------------------------------------------------
            st.markdown("""### :page_facing_up: Resumen Inicial""")
            st.info("""Puede visualizar un **dataframe segmentado por subcanal**, si deseas el detalle del **PagoTotal**, simplemente haz clic  en **"Mostrar adicionales"**.""")


            tabla_pivote = df_recalculado.groupby('Subcanal').agg(
                QHc=('QHc', 'sum'),
                QUrs=('QUrs', 'sum'),
                QVentas=('QVentas', 'sum'),
                PagoTotal = ('PagoTotal', 'sum'),
                Acelerador = ('Acelerador', 'sum'),
                Planilla = ('Planilla', 'sum'),
                Bono = ('Bono', 'sum'),
                Campaña = ('Campaña', 'sum'),
                Otros = ('Otros', 'sum')
            ).reset_index()
            
            tabla_pivote['URM2%'] = (tabla_pivote['QUrs'] / tabla_pivote['QVentas']) * 100
        
            tabla_pivote = tabla_pivote.sort_values(by='QHc', ascending=False).reset_index(drop=True)

            total_row = pd.DataFrame({
                'Subcanal': ['Total'],
                'QHc': [tabla_pivote['QHc'].sum()],
                'QVentas': [tabla_pivote['QVentas'].sum()],
                'QUrs': [tabla_pivote['QUrs'].sum()],
                'URM2%': [round((tabla_pivote['QUrs'].sum() / tabla_pivote['QVentas'].sum()) * 100, 1) 
                        if tabla_pivote['QVentas'].sum() > 0 else 0],
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
                columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'QVentas', 'QUrs','URM2%', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']
            else:
                columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal'] 
                
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
                    QVentas=('QVentas', 'sum'),
                    PagoTotal = ('PagoTotal', 'sum'),
                    Acelerador = ('Acelerador', 'sum'),
                    Planilla = ('Planilla', 'sum'),
                    Bono = ('Bono', 'sum'),
                    Campaña = ('Campaña', 'sum'),
                    Otros = ('Otros', 'sum')
                ).reset_index().rename(columns={decil_col: 'Grupo'})

                resumen['URM2%'] = (resumen['QUrs'] / resumen['QVentas']) * 100

                totales = pd.DataFrame({
                    'Grupo': ['Total'],
                    'RangoGrupo': [' '],
                    'QHc': [resumen['QHc'].sum()],
                    'QVentas': [resumen['QVentas'].sum()],
                    'QUrs': [resumen['QUrs'].sum()],
                    'URM2%': [round((resumen['QUrs'].sum() / resumen['QVentas'].sum()) * 100,1) if resumen['QVentas'].sum() > 0 else 0],
                    'PagoTotal': [resumen['PagoTotal'].sum()],
                    'Acelerador':  [resumen['Acelerador'].sum()],
                    'Planilla': [resumen['Planilla'].sum()],
                    'Bono':  [resumen['Bono'].sum()],
                    'Campaña':  [resumen['Campaña'].sum()],
                    'Otros':  [resumen['Otros'].sum()],
                })

                resumen = pd.concat([resumen, totales], ignore_index=True)

                resumen = resumen[['Grupo', 'RangoGrupo','QHc', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']]

                return resumen
            
           
            resumen_recalculado = resumen_deciles(df_recalculado, 'Grupo')

            mostrar_columnas_adicionales_resumen = st.checkbox("Mostrar adicionales ..")

            if mostrar_columnas_adicionales_resumen:
                columnas_a_mostrar_resumen = ['Grupo', 'RangoGrupo','QHc', 'QVentas', 'QUrs', 'URM2%',  'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                
            else:
                columnas_a_mostrar_resumen = ['Grupo', 'RangoGrupo', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal']  
                
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
                columnas_a_mostrar = ['Departamento', 'Socio', 'Kam' ,'Subcanal', columna_DNI, 'Grupo', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                
            else:
                columnas_a_mostrar = ['Departamento', 'Socio', 'Kam', 'Subcanal', columna_DNI, 'Grupo', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal']  
                
            
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



def asignar_bandera_promedio(row, meses_validos, num_grupos):
    grupos_mes = [row[mes] for mes in meses_validos if row[mes] != 0]
    
    if not grupos_mes:
        return "⚪", "Neutro"
    
    promedio = sum(grupos_mes) / len(grupos_mes)
    umbral_bajo = num_grupos * 0.5
    umbral_alto = num_grupos * 0.6

    # Determinar si el desempeño es Bueno, Malo o Neutro
    if promedio <= umbral_bajo:
        tendencia = "Flecha arriba" if grupos_mes[-1] < grupos_mes[0] else "Flecha abajo"
        simbolo = "✅🔼" if tendencia == "Flecha arriba" else "✅🔽"
        return simbolo, "Bueno"
    elif promedio >= umbral_alto:
        tendencia = "Flecha arriba" if grupos_mes[-1] < grupos_mes[0] else "Flecha abajo"
        simbolo = "❌🔼" if tendencia == "Flecha arriba" else "❌🔽"
        return simbolo, "Malo"
    else:
        return "⚪", "Neutro"




        

def historico(carpeta_archivos=carpeta_canal, canal = filtro_canal):
         
    # define columna DNI segun canal seleccionado
    columna_DNI = ''

    if canal == "PROACTIVO":
        columna_DNI = 'DNI'
    else:
        columna_DNI = 'DNI LIDER'


    # lee carpeta de archivos del canal
    archivos = os.listdir(carpeta_archivos) 
    archivos = [os.path.join(carpeta_archivos, archivo) for archivo in archivos]
    meses = meses_disponibles(carpeta_archivos)

    
    # arma resultados por mes 
    resultados = []

    for archivo, mes in zip(archivos, meses):

        df = pd.read_excel(archivo,  dtype={columna_DNI: str}) 

        #df[columna_DNI] = df[columna_DNI].str.zfill(8)  # rellenar 8 espacios

        df = dataframe_mes(mes, carpeta_archivos, canal)

        #df['URM2%'] = round((df['QUrs'] / df['QVentas'])*100, 1)

        df_filtrado = df.copy()

        if filtro_subcanal:
            df_filtrado = df_filtrado[df_filtrado['Subcanal'].isin(filtro_subcanal)]

        if pd.notnull(min_input) and pd.notnull(max_input):
            df_filtrado = df_filtrado[
                (df_filtrado["QVentas"] >= min_input) & 
                (df_filtrado["QVentas"] <= max_input)
            ]

        if filtro_departamento:
            df_filtrado = df_filtrado[df_filtrado['Departamento'].isin(filtro_departamento)]

        if filtro_kam:
            df_filtrado = df_filtrado[df_filtrado['Kam'].isin(filtro_kam)]

        if filtro_socio:
            df_filtrado = df_filtrado[df_filtrado['Socio'].isin(filtro_socio)]

        if not df_filtrado.empty:
            df_mes_con_grupos = calcular_grupos_personalizados(df_filtrado, num_grupos, columnas_orden=["URM2%", 'QUrs', 'QVentas'])
            df_mes_con_grupos['Mes'] = mes
            resultados.append(df_mes_con_grupos)


    if resultados:

        df_total = pd.concat(resultados, ignore_index=True)
        
        todos_dnis = pd.DataFrame(df_total[columna_DNI].unique(), columns=[columna_DNI]).sort_values(by=columna_DNI)
        todos_dnis = todos_dnis.reset_index(drop=True)
        df_completo = todos_dnis.copy()
         

        st.markdown("""### :bar_chart: Tabla resumen""")
        #dni_ingresado = st.text_input(f"Ingresa un {columna_DNI} para buscar su progreso:")

        filtro1, filtro2 = st.columns([2,2])
        with filtro1:
            dni_ingresado = st.text_input(f"Ingresa un {columna_DNI} para buscar su progreso:")
        

        meses_validos = [mes for mes in meses if not df_total[df_total['Mes'] == mes]['Grupo'].isnull().all()]
        for mes in meses_validos:
            df_mes = df_total[df_total['Mes'] == mes][[columna_DNI, 'Grupo']]
            df_mes_completo = todos_dnis.merge(df_mes, on=columna_DNI, how='left').rename(columns={'Grupo': mes})
            df_completo = df_completo.merge(df_mes_completo[[columna_DNI, mes]], on=columna_DNI, how='left').fillna(0)


        df_completo['Evolución'] = df_completo.apply(
        lambda row: [row[mes] for mes in meses_validos],
        axis=1
        )

       #df_completo['Bandera'] = df_completo.apply(lambda row: asignar_bandera_promedio(row, meses_validos), axis=1)

       
        df_completo[['Bandera', 'Desempeño']] = df_completo.apply(
            lambda row: pd.Series(asignar_bandera_promedio(row, meses_validos, num_grupos)),
            axis=1
        )
         
        desempeños = ["Bueno", "Malo", "Neutro"]
        with filtro2:
             desempeño_ingresado = st.multiselect(
                "Selecciona el desempeño",
                options=desempeños,
                default=[]  # Selecciona "Bueno" como opción predeterminada
            )
                    
        if dni_ingresado: 
            try:
                
                df_seleccionado = df_completo[df_completo[columna_DNI] == dni_ingresado]
                

                if df_seleccionado.empty:  
                    st.warning(f"El {columna_DNI} ingresado no se encuentra en los datos.")
                else:
                    st.write(f"Registro del {columna_DNI} seleccionado:")
                    st.dataframe(
                        df_seleccionado,
                        column_config={
                            "Evolución": st.column_config.BarChartColumn(
                                "Evolución de grupos",
                                y_min=0,
                                y_max=num_grupos
                            )
                        },
                        use_container_width=True
                    )

                    #st.markdown("---")
                    #progreso = df_seleccionado[meses].values.flatten()
                    #df_progreso = pd.DataFrame({
                    #    'Mes': meses,
                    #    'Progreso': progreso
                    #})

                
                #df_progreso['Mes'] = pd.Categorical(df_progreso['Mes'], categories=meses, ordered=True)
                #df_progreso = df_progreso.sort_values('Mes')
                #df_progreso['Progreso'] = num_grupos + 1 - df_progreso['Progreso']
                #st.line_chart(df_progreso.set_index('Mes')['Progreso'])

                
            except ValueError:
                st.error(f"Por favor, ingresa un {columna_DNI} válido.")

        else:  
            st.write("Mostrando todos los registros:")
            
            if not desempeño_ingresado:  # Si la lista está vacía
                df_seleccionado = df_completo
            else:
                df_seleccionado = df_completo[df_completo['Desempeño'].isin(desempeño_ingresado)]

            st.dataframe(
                df_seleccionado,
                column_config={
                    "Evolución": st.column_config.BarChartColumn(
                        "Evolución de grupos",
                        y_min=0,
                        y_max=num_grupos
                    )
                },
                use_container_width=True
            )

            
            #df_seleccionado = df_completo


        # descargar
        towrite_seleccionado = io.BytesIO()
        df_seleccionado.to_csv(towrite_seleccionado, index=False, encoding='utf-8')
        towrite_seleccionado.seek(0)

        st.download_button(
            label="Descargar",
            data=towrite_seleccionado,
            file_name="dataframe_historico.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("No hay datos para los filtros seleccionados.")




#--------------------------------------------------------------------------

with tab_vista_normal:
    normal()
with tab_vista_historica:
    historico()