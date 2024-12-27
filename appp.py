import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import traceback  
import io


carpeta = 'archivos'
img = Image.open('entel.jpg')
st.set_page_config(page_title='Recalculo Entel', page_icon=img, layout='wide')
st.sidebar.title('Recalculo Entel')


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

def calcular_grupos_personalizados(dataframe, num_grupos, columnas_orden):
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by=columnas_orden, ascending=[False] * len(columnas_orden)).reset_index(drop=True)
    
    total_filas = len(dataframe)
    tam_grupo = total_filas // num_grupos  # Tamaño base de cada grupo
    restos = total_filas % num_grupos    

    
    grupos = []
    for grupo in range(1, num_grupos + 1):
        tam_actual = tam_grupo + (1 if grupo > num_grupos - restos else 0)  # Asignar el resto a los últimos grupos
        grupos.extend([grupo] * tam_actual)
    
    
    dataframe['Grupo'] = grupos[:total_filas]
    
    
    limites = dataframe.groupby('Grupo')[columnas_orden].agg(['min', 'max'])
  
    dataframe['RangoGrupo'] = dataframe['Grupo'].map(
        lambda grupo: ', '.join(
            f"{col}: ({limites.loc[grupo, (col, 'max')]}-{limites.loc[grupo, (col, 'min')]})"
            for col in columnas_orden
        )
    )
    
    return dataframe

def archivos_listados(carpeta_archivos):
    if os.path.exists(carpeta_archivos):
        archivos = os.listdir(carpeta_archivos)
        archivos_filtrados = [archivo for archivo in archivos if 'NUEVA BASE - PROACTIVO' in archivo]
        return archivos_filtrados
    else:
        print(f"La carpeta no existe: {carpeta_archivos}")
        return []  

def meses_disponibles(carpeta_archivos):
    archivos = archivos_listados(carpeta_archivos)
    meses = [archivo.split()[-1].replace('.xlsx', '') for archivo in archivos]
    return sorted(meses)

def archivo_mes(mes_elegido, carpeta_archivos):
    archivos = archivos_listados(carpeta_archivos)
    for archivo in archivos:
        if mes_elegido in archivo:
            return  os.path.join(carpeta, archivo)             
    return None

meses_orden = [
     "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", 
    "JULIO", "AGOSTO", "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
]

def ordenar_meses(meses):
    return sorted(
        [mes for mes in meses if mes.upper() in meses],  # Filtrar meses válidos
        key=lambda mes: meses_orden.index(mes.upper())
    )


def dataframe_mes(mes, carpeta_archivos):
    archivo_path = archivo_mes(mes, carpeta_archivos)
    df_original = pd.read_excel(archivo_path, dtype={'DNI': str})   
    df_original['MES'] = df_original['MES'].astype(str)
    df_original['HC'] = 1
    df_original['URM2%'] = round((df_original['Urs'] / df_original['QVENTAS']) * 100,2)
    df_original['PagoTotal'] = (df_original['ACELERADOR'] + df_original['PLANILLA'] + df_original['BONO']+ df_original['CAMPAÑA']+ df_original['OTROS'])
    
    df_original = df_original.rename(columns={
            'MES': 'Mes',
            'DEPARTAMENTO': 'Departamento',
            'SSNN FINAL': 'Socio',
            'HC': 'QHc',
            'TIPO': 'Subcanal',
            'DNI': 'DNI',
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

def subcanales_disponibles(mes_elegido, carpeta_archivos):
    df_mes = dataframe_mes(mes_elegido, carpeta_archivos)
    return sorted(df_mes['Subcanal'].unique())

def departamentos_disponibles(mes_elegido, carpeta_archivos):
    df_mes = dataframe_mes(mes_elegido, carpeta_archivos)
    return sorted(df_mes['Departamento'].unique())


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


with st.sidebar.expander("Filtros", expanded=True):
    filtros_mes = st.selectbox("Mes", meses_disponibles(carpeta_archivos=carpeta))

    num_grupos = st.number_input("Número de grupos", min_value=2, max_value=50, value=10)

    filtros_seleccionados_subcanal = st.multiselect(
            "Subcanal",
            options= subcanales_disponibles(mes_elegido=filtros_mes, carpeta_archivos=carpeta),
            default=[]  
    )

    filtros_seleccionados_departamento = st.multiselect(
        "Departamento",
        options=departamentos_disponibles(mes_elegido=filtros_mes, carpeta_archivos=carpeta),
        default=[]  
    )

    min_val, max_val = rango_ventas(carpeta_archivos=carpeta)
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
    

def recalculo():
     
    archivo_path = archivo_mes(mes_elegido=filtros_mes, carpeta_archivos=carpeta)

    try:
        df = dataframe_mes(mes=filtros_mes, carpeta_archivos=carpeta)
       
        df = calcular_grupos_personalizados(dataframe= df,num_grupos= num_grupos, columnas_orden=['URM2%', 'QUrs', 'QVentas'])
       # st.dataframe(df)

        df_filtrado = df.copy()

        if filtros_seleccionados_subcanal:
            df_filtrado = df_filtrado[df_filtrado['Subcanal'].isin(filtros_seleccionados_subcanal)]

        if pd.notnull(min_input) and pd.notnull(max_input):
            df_filtrado = df_filtrado[
                (df_filtrado["QVentas"] >= min_input) & 
                (df_filtrado["QVentas"] <= max_input)
            ]

        if filtros_seleccionados_departamento:
            df_filtrado = df_filtrado[df_filtrado['Departamento'].isin(filtros_seleccionados_departamento)]


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
                SS=('SS', 'sum'),
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
                'SS': [tabla_pivote['SS'].sum()],
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
                columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'QVentas', 'QUrs','URM2%', 'SS', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']
            else:
                columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'QVentas', 'QUrs', 'URM2%','SS', 'PagoTotal'] 
                
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
                    SS=('SS', 'sum'),
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
                    'SS': [resumen['SS'].sum()],
                    'PagoTotal': [resumen['PagoTotal'].sum()],
                    'Acelerador':  [resumen['Acelerador'].sum()],
                    'Planilla': [resumen['Planilla'].sum()],
                    'Bono':  [resumen['Bono'].sum()],
                    'Campaña':  [resumen['Campaña'].sum()],
                    'Otros':  [resumen['Otros'].sum()],
                })

                resumen = pd.concat([resumen, totales], ignore_index=True)

                resumen = resumen[['Grupo', 'RangoGrupo','QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']]

                return resumen
            
           
            resumen_recalculado = resumen_deciles(df_recalculado, 'Grupo')

            mostrar_columnas_adicionales_resumen = st.checkbox("Mostrar adicionales ..")

            if mostrar_columnas_adicionales_resumen:
                columnas_a_mostrar_resumen = ['Grupo', 'RangoGrupo','QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                
            else:
                columnas_a_mostrar_resumen = ['Grupo', 'RangoGrupo', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal']  
                
            st.dataframe(resumen_recalculado[columnas_a_mostrar_resumen], use_container_width=True)
            st.markdown("---")

            # --------------------------------------------------------------------------------------


            # TABLA RECALCULADA --------------------------------------------------------------------
            st.markdown("""### :mag: Tabla detalle DNI""")
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
                dni_ingresado = st.text_input("Ingresa un DNI:")


            if grupos_seleccionados:
                df_descarga = df_recalculado[df_recalculado['Grupo'].isin(grupos_seleccionados)]
            else:
                df_descarga = df_recalculado

            mostrar_columnas_adicionales = st.checkbox("Mostrar adicionales ...")

            if mostrar_columnas_adicionales:
                columnas_a_mostrar = ['Departamento', 'Socio', 'Subcanal', 'DNI', 'Grupo', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                
            else:
                columnas_a_mostrar = ['Departamento', 'Socio', 'Subcanal', 'DNI', 'Grupo', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal']  
                
            
            if dni_ingresado: 
                try:
                    dni_ingresado = str(dni_ingresado)  
                    df_descarga = df_descarga[df_descarga['DNI'] == dni_ingresado]

                    if df_descarga.empty:  
                        st.warning("El DNI ingresado no se encuentra en los datos.")
                    else:
                        st.dataframe(df_descarga[columnas_a_mostrar], use_container_width=True)
                except ValueError:
                    st.error("Por favor, ingresa un DNI válido (número entero).")
            else:  
                st.dataframe(df_descarga[columnas_a_mostrar], use_container_width=True)


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

        
           
         # --------------------------------------------------------------------------------------


        else:
            st.warning("No hay datos para los filtros seleccionados.")

        
    except Exception as e:
        st.write(f"Error al cargar el archivo: {e}")
        st.write("Detalles de la excepción:")
        st.text(traceback.format_exc())  

   
def historico(carpeta_archivos=carpeta):
     
    archivos = os.listdir(carpeta_archivos) 
    archivos = [os.path.join(carpeta_archivos, archivo) for archivo in archivos]
   
    meses = ordenar_meses(meses_disponibles(carpeta_archivos))
   
    
    resultados = []

    for archivo, mes in zip(archivos, meses):
        df = pd.read_excel(archivo, dtype={'DNI': str})   
        df['DNI'] = df['DNI'].str.zfill(8)  
        df = dataframe_mes(mes, carpeta_archivos)

        df['URM2%'] = round((df['QUrs'] / df['QVentas'])*100, 1)

        df_filtrado = df.copy()

        if filtros_seleccionados_subcanal:
            df_filtrado = df_filtrado[df_filtrado['Subcanal'].isin(filtros_seleccionados_subcanal)]

        if pd.notnull(min_input) and pd.notnull(max_input):
            df_filtrado = df_filtrado[
                (df_filtrado["QVentas"] >= min_input) & 
                (df_filtrado["QVentas"] <= max_input)
            ]


        if filtros_seleccionados_departamento:
            df_filtrado = df_filtrado[df_filtrado['Departamento'].isin(filtros_seleccionados_departamento)]


        if not df_filtrado.empty:
            df_con_grupos = calcular_grupos_personalizados(
                df_filtrado, 
                num_grupos, 
                columnas_orden=["URM2%", 'QUrs', 'QVentas']
            )
            df_con_grupos['Mes'] = mes
            resultados.append(df_con_grupos)


    if resultados:
        
        df_total = pd.concat(resultados, ignore_index=True)
        
        todos_dnis = pd.DataFrame(df_total['DNI'].unique(), columns=['DNI']).sort_values(by='DNI')
        todos_dnis = todos_dnis.reset_index(drop=True)
        df_completo = todos_dnis.copy()
         
        st.markdown("""### :bar_chart: Tabla resumen""")
        dni_ingresado = st.text_input("Ingresa un DNI para buscar su progreso:")

       
        for mes in meses:
            df_mes = df_total[df_total['Mes'] == mes][['DNI', 'Grupo']]
            df_mes_completo = todos_dnis.merge(df_mes, on='DNI', how='left').rename(columns={'Grupo': mes})
            df_completo = df_completo.merge(df_mes_completo[['DNI', mes]], on='DNI', how='left').fillna(0)


        df_completo['Evolución'] = df_completo.apply(
            lambda row: [row[mes] for mes in meses],
            axis=1
        )

        df_completo['Bandera'] = "Vacio"

        
      
        if dni_ingresado: 
            try:
                
                df_seleccionado = df_completo[df_completo['DNI'] == dni_ingresado]

                if df_seleccionado.empty:  
                    st.warning("El DNI ingresado no se encuentra en los datos.")
                else:
                    st.write("Registro del DNI seleccionado:")
                    st.dataframe(
                        df_seleccionado,
                        column_config={
                            "Evolución": st.column_config.BarChartColumn(
                                "Evolución de deciles",
                                y_min=0,
                                y_max=num_grupos
                            )
                        },
                        use_container_width=True
                    )

                    st.markdown("---")
                    progreso = df_seleccionado[meses].values.flatten()
                    df_progreso = pd.DataFrame({
                        'Mes': meses,
                        'Progreso': progreso
                    })

                    df_progreso['Mes'] = pd.Categorical(df_progreso['Mes'], categories=meses, ordered=True)
                    df_progreso = df_progreso.sort_values('Mes')

                    st.line_chart(df_progreso.set_index('Mes')['Progreso'])

            except ValueError:
                st.error("Por favor, ingresa un DNI válido (número entero).")
        else:  
            st.write("Mostrando todos los registros:")
            st.dataframe(
                df_completo,
                column_config={
                    "Evolución": st.column_config.BarChartColumn(
                        "Evolución de deciles",
                        y_min=0,
                        y_max=num_grupos
                    )
                },
                use_container_width=True
            )
    else:
        st.warning("No hay datos para los filtros seleccionados.")



tab_vista_normal, tab_vista_historica = \
st.tabs(["VISTA NORMAL", "VISTA HISTORICA"])

with tab_vista_normal:
    recalculo()
with tab_vista_historica:
    historico(carpeta)
