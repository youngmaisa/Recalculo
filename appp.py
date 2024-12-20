import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import traceback  
import io
import calendar



img = Image.open('entel.jpg')
st.set_page_config(page_title='Recalculo Entel', page_icon=img, layout='wide')


carpeta_archivos = 'archivos'


def recalculo():

    if os.path.exists(carpeta_archivos):
        archivos = os.listdir(carpeta_archivos)
        
        archivos_filtrados = [archivo for archivo in archivos if "NUEVA BASE - PROACTIVO" in archivo]
        
        meses_disponibles = [archivo.split()[-1].replace('.xlsx', '') for archivo in archivos_filtrados]
        meses_disponibles.sort()

        if meses_disponibles:

            st.subheader("Filtros")
            mes_elegido = st.selectbox("Mes", meses_disponibles)
        
            archivo_mes = None
            for archivo in archivos_filtrados:
                if mes_elegido in archivo:
                    archivo_mes = archivo
                    break
            
            if archivo_mes:
                #st.write(f"Archivo cargado: {archivo_mes}")
                archivo_path = os.path.join(carpeta_archivos, archivo_mes)

                try:
                    df_original = pd.read_excel(archivo_path)
                
                    df_original['DNI'] = df_original['DNI'].astype(str)
                    df_original['MES'] = df_original['MES'].astype(str)
                    df_original['HC'] = 1
                    df_original['URM2%'] = (df_original['Urs'] / df_original['QVENTAS']) * 100
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

                    print(df_original.columns)
                    num_grupos = st.number_input("Número de grupos", min_value=2, max_value=50, value=10)


                    def calcular_grupos_personalizados(dataframe, num_grupos, columnas_orden):
                        dataframe = dataframe.copy()
                        dataframe = dataframe.sort_values(by=columnas_orden, ascending=[False] * len(columnas_orden)).reset_index(drop=True)
                        
                        total_filas = len(dataframe)
                        tam_grupo = total_filas // num_grupos
                        restos = total_filas % num_grupos  # Filas adicionales 
                        
                        # Asignar grupos
                        grupos = []
                        for grupo in range(1, num_grupos + 1):
                            tam_actual = tam_grupo + (1 if grupo <= restos else 0)  # Agrega 1 fila extra- primeros 
                            grupos.extend([grupo] * tam_actual)
                        
                        dataframe['Grupo'] = grupos[:total_filas]
                        return dataframe

                    df = calcular_grupos_personalizados(df_original, num_grupos, columnas_orden=['QUrs', 'URM2%'])
                    df = df[['Mes', 'Departamento', 'Socio', 'Subcanal', 'DNI', 'QUrs', 'Grupo', 'QVentas', 'URM2%', 'QHc', 'SS', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']]


                    # SIDE BAR 
                    #st.sidebar.title("Navegación")


                    # FILTROS
                                
                    opciones_filtros_subcanal = sorted(df['Subcanal'].unique())
                    filtros_seleccionados_subcanal = st.multiselect(
                        "Subcanal",
                        options=opciones_filtros_subcanal,
                        default=[]  
                    )

                    opciones_filtros_departamento = sorted(df['Departamento'].unique())
                    filtros_seleccionados_departamento = st.multiselect(
                        "Departamento",
                        options=opciones_filtros_departamento,
                        default=[]  
                    )

                    min_val = df["QVentas"].min()
                    max_val = df["QVentas"].max()
                    filtros_seleccionados = st.slider(
                        "Rango de ventas:",
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=(int(min_val), int(max_val))
                    )

                    df_filtrado = df.copy()
                    if filtros_seleccionados_subcanal:
                        df_filtrado = df_filtrado[df_filtrado['Subcanal'].isin(filtros_seleccionados_subcanal)]

                    if filtros_seleccionados:
                        df_filtrado = df_filtrado[ 
                            (df_filtrado["QVentas"] >= filtros_seleccionados[0]) & 
                            (df_filtrado["QVentas"] <= filtros_seleccionados[1])
                        ]

                    if filtros_seleccionados_departamento:
                        df_filtrado = df_filtrado[df_filtrado['Departamento'].isin(filtros_seleccionados_departamento)]


                    # RECALCULO               
                    if not df_filtrado.empty:
                        df_recalculado = calcular_grupos_personalizados(df_filtrado, num_grupos= num_grupos , columnas_orden= ['QUrs', 'URM2%'])
                        df_recalculado['URM2%'] = (df_recalculado['QUrs'] / df_recalculado['QVentas']) * 100
                    else:
                        df_recalculado = pd.DataFrame(columns=df.columns)


                    # TABLA - RESUMEN INICIAL

                    if not df_recalculado.empty:
                        st.subheader("Resumen Inicial")
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
                        
                        tabla_pivote['URM2%'] = round((tabla_pivote['QUrs'] / tabla_pivote['QVentas']) * 100,1)
                    
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
                        tabla_pivote = tabla_pivote[['Subcanal', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']]
                       
                        mostrar_columnas_adicionales_pivote = st.checkbox("Mostrar adicionales .")


                        if mostrar_columnas_adicionales_pivote:
                            columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']
                            
                        else:
                            columnas_a_mostrar_pivote = ['Subcanal', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal'] 
                            

                        st.table(tabla_pivote[columnas_a_mostrar_pivote])


                        # TABLA INFO DECILES

                        def resumen_deciles(dataframe, decil_col):
                            resumen = dataframe.groupby(decil_col).agg(
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

                            resumen['URM2%'] = round((resumen['QUrs'] / resumen['QVentas']) * 100,1)
                    

                            totales = pd.DataFrame({
                                'Grupo': ['Total'],
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

                            resumen = resumen[['Grupo', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']]

                            return resumen


                        st.subheader("Grupos")

                        resumen_recalculado = resumen_deciles(df_recalculado, 'Grupo')


                        mostrar_columnas_adicionales_resumen = st.checkbox("Mostrar adicionales ..")


                        if mostrar_columnas_adicionales_resumen:
                            columnas_a_mostrar_resumen = ['Grupo', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                            
                        else:
                            columnas_a_mostrar_resumen = ['Grupo', 'QHc', 'QVentas', 'QUrs', 'URM2%', 'SS', 'PagoTotal']  
                            
                        
                        st.table(resumen_recalculado[columnas_a_mostrar_resumen])


                        # TABLA RECALCULADA
                        st.subheader("Tabla detalle DNI")

                        grupos_disponibles = df_recalculado['Grupo'].unique()  # Obtenemos los grupos disponibles en el dataframe
                    
                        grupos_seleccionados = st.multiselect(
                            "Seleccionar grupos (puedes elegir varios)",
                            options= grupos_disponibles,  
                            default= []  
                        )

                        
                        df_descarga = df_recalculado.sort_values(by=['QUrs', 'URM2%'], ascending=[False, False])

                        if grupos_seleccionados:
                            df_descarga = df_recalculado[df_recalculado['Grupo'].isin(grupos_seleccionados)]
                        else:
                            df_descarga = df_recalculado


                        mostrar_columnas_adicionales = st.checkbox("Mostrar adicionales ...")

                        if mostrar_columnas_adicionales:
                            columnas_a_mostrar = ['Departamento', 'Socio', 'Subcanal', 'DNI', 'Grupo', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                            
                        else:
                            columnas_a_mostrar = ['Departamento', 'Socio', 'Subcanal', 'DNI', 'Grupo', 'QVentas', 'QUrs', 'URM2%', 'PagoTotal']  
                            
                        
                        st.dataframe(df_descarga[columnas_a_mostrar])


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

                    st.markdown(
                        """
                        <style>
                        .block-container {
                            padding: 60px;
                            max-width: 95%
                        }
                        </style>

                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.write(f"Error al cargar el archivo: {e}")
                    st.write("Detalles de la excepción:")
                    st.text(traceback.format_exc())  # Muestra la traza completa del error
            else:
                st.write("No se encontró un archivo para el mes seleccionado.")
        else:
            st.write("No hay datos disponibles para los meses.")
    else:
        st.write(f"La carpeta no existe: {carpeta_archivos}")




def calcular_grupos(dataframe, num_grupos, columnas_orden):
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by=columnas_orden, ascending=[False] * len(columnas_orden)).reset_index(drop=True)
    
    total_filas = len(dataframe)
    tam_grupo = total_filas // num_grupos
    restos = total_filas % num_grupos  
    
   
    grupos = []
    for grupo in range(1, num_grupos + 1):
        tam_actual = tam_grupo + (1 if grupo <= restos else 0)   
        grupos.extend([grupo] * tam_actual)
    
    dataframe['Grupo'] = grupos[:total_filas]
   
    return dataframe



def ordenar_meses(meses):
    orden_cronologico = [
        "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", 
        "JULIO", "AGOSTO", "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
    ]
    meses_ordenados = sorted(meses, key=lambda mes: orden_cronologico.index(mes.upper()))
    return meses_ordenados


def historico():
    if not os.path.exists(carpeta_archivos):
        st.error(f"La carpeta '{carpeta_archivos}' no existe.")
        return

    archivos = os.listdir(carpeta_archivos)
    archivos_filtrados = [
        os.path.join(carpeta_archivos, archivo)
        for archivo in archivos
        if "NUEVA BASE - PROACTIVO" in archivo and archivo.endswith(".xlsx")
    ]

    if not archivos_filtrados:
        st.warning("No se encontraron archivos con el nombre esperado.")
        return

    meses_disponibles = [archivo.split()[-1].replace('.xlsx', '') for archivo in archivos_filtrados]
    meses_disponibles = ordenar_meses(meses_disponibles)
 

    num_grupos2 = st.number_input("Número de grupos", min_value=2, max_value=50, value=10)
    subcanal_filtro2 = st.multiselect("Subcanal", options=["DAE", "DESARROLLADOR", "FULL PREPAGO DAE", "FULL PREPAGO DD", "HC EMO", "HC EMO MERCADO"], default=[])
    departamento_filtro2 = st.multiselect("Departamento", options=["AMAZONAS", "ANCASH", "APURIMAC", "LIMA"], default=[])

    
    filtros_seleccionados = st.slider(
        "Rango de ventas:",
        min_value=int(0),
        max_value=int(1000),
        value=(int(0), int(1000))
    )

    
    resultados = []

    for archivo, mes in zip(archivos_filtrados, meses_disponibles):
        df = pd.read_excel(archivo)  
        df['URM2%'] = round((df['Urs'] / df['QVENTAS'])*100, 1)

        df_filtrado = df.copy()

        if subcanal_filtro2:
            df_filtrado = df[df["TIPO"].isin(subcanal_filtro2)]
        if departamento_filtro2:
            df_filtrado = df[df["DEPARTAMENTO"].isin(departamento_filtro2)]

        if filtros_seleccionados:
            df_filtrado = df_filtrado[ 
                (df_filtrado["QVENTAS"] >= filtros_seleccionados[0]) & 
                (df_filtrado["QVENTAS"] <= filtros_seleccionados[1])
            ]


        if not df_filtrado.empty:
            df_con_grupos = calcular_grupos(df_filtrado, num_grupos2, columnas_orden=["Urs", "URM2%"])
            df_con_grupos['MES'] = mes
            resultados.append(df_con_grupos)
        else:
            df_con_grupos = calcular_grupos(pd.DataFrame(columns=df_filtrado.columns), num_grupos2, columnas_orden=["Urs", "URM2%"])
            df_con_grupos['MES'] = mes
            resultados.append(df_con_grupos)



    if resultados:
       
        try:
            df_total = pd.concat(resultados, ignore_index=True)
        except Exception as e:
            st.error(f"Error al concatenar resultados: {e}")
            return

        if 'DNI' not in df_total.columns or 'Grupo' not in df_total.columns or 'MES' not in df_total.columns:
            st.error("El DataFrame no contiene las columnas requeridas: DNI, Grupo, MES")
            return

        df_total['DNI'] = df_total['DNI'].astype('str')
        todos_dnis = pd.DataFrame(df_total['DNI'].unique(), columns=['DNI']).sort_values(by='DNI')
        todos_dnis = todos_dnis.reset_index(drop=True)
        df_completo = todos_dnis.copy()
        

        for mes in meses_disponibles:
           
            df_mes = df_total[df_total['MES'] == mes][['DNI', 'Grupo']]

            if df_mes.empty:
                st.warning(f"No hay datos para el mes {mes}. Se llenará con NaN.")
                df_mes['Grupo'] = None
           
            df_mes_completo = todos_dnis.merge(df_mes, on='DNI', how='left')
           
            
            df_mes_completo = df_mes_completo.rename(columns={'Grupo': mes})
           
            df_completo = df_completo.merge(df_mes_completo[['DNI', mes]], on='DNI', how='left')
            df_completo = df_completo.fillna(0)
            

      
        
        df_completo['Evolución'] = df_completo.apply(
            lambda row: [row[mes] for mes in meses_disponibles],
            axis=1
        )

        df_completo['Bandera'] = "Vacio"
       
        st.dataframe(
            df_completo,
            column_config={
                "Evolución": st.column_config.BarChartColumn(
                    "Evolución de deciles",
                    y_min=0,
                    y_max=num_grupos2
                )
            },
            use_container_width=True
        )

        

        dni_seleccionado = st.selectbox("Selecciona un DNI para ver el progreso", df_completo['DNI'])

        if dni_seleccionado:
            progreso = df_completo[df_completo['DNI'] == dni_seleccionado][meses_disponibles].values.flatten()
            
          
            df_progreso = pd.DataFrame({'Mes': meses_disponibles, 'Progreso': progreso})
            df_progreso['Mes'] = pd.Categorical(df_progreso['Mes'], categories=meses_disponibles, ordered=True)
            df_progreso = df_progreso.sort_values('Mes')

            st.line_chart(df_progreso.set_index('Mes'))

    else:
        st.warning("No hay datos para los filtros seleccionados.")

       

st.sidebar.title("Navegación")
opcion = st.sidebar.radio("", ["Recalculo", "Historico"])


if opcion == "Recalculo":
    recalculo()
elif opcion == "Historico":
    historico()
