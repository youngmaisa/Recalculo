import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
import io
import traceback

#st.title("Herramienta de Recálculo")
st.set_page_config(layout="wide")

st.markdown("""## Herramienta de Recálculo""")

columna_DNI = 'DNI'
carpeta_archivos  = 'data/PROACTIVO'
subcanales = ['DESARROLLADOR', 'HC EMO MERCADO', 'FULL PREPAGO DD',
            'DAE', 'FULL PREPAGO DAE', 'HC MONOMARCA', 'HC EMO']
departamentos = ['PUNO', 'TUMBES', 'UCAYALI', 'HUANUCO', 'TACNA', 'AMAZONAS', 'LIMA', 
                'CUSCO', 'APURIMAC', 'CAJAMARCA', 'MADRE DE DIOS', 'LORETO', 'ANCASH', 
                'PIURA', 'ICA', 'MOQUEGUA', 'SAN MARTIN', 'AREQUIPA','LAMBAYEQUE', 
                'HUANCAVELICA','LA LIBERTAD', 'JUNIN', 'PASCO', 'AYACUCHO']
kams = ['ALEXANDRA TABOADA', 'KARINA TORRES', 'GERSON RIMACH', 'ROMINNA PINATTE',
        'REYNA CABANILLAS', 'MARIA MARTINEZ', 'RUBEN SAMANEZ', 'JORGE PUELLES', 'NATHALY REYES']
socios = ['TVOLUTION', 'FANERO', 'D&D', 'GRUPO CAYAO', 'JOKATEL', 'PBD', 'POWER TIMBER', 'DISCOMTECH']
meses_orden = [
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", 
    "JULIO", "AGOSTO", "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
]


tab_vista_normal, tab_vista_historica = \
    st.tabs(['VISTA NORMAL', 'VISTA HISTORICA'])


min = 0
max = 5000

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

    min_val, max_val = min, max
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
        st.error("El valor mínimo de ventas no puede ser mayor que el valor máximo.")

     
    min_urm2, max_urm2 = 0, 100
    min_input_urm2 = st.number_input(
        "Ingrese el valor mínimo de URM2%:",
        min_value=int(min_urm2),
        max_value=int(max_urm2),
        value=int(min_urm2),
        step=1
    )
    max_input_urm2 = st.number_input(
        "Ingrese el valor máximo de URM2%:",
        min_value=int(min_urm2),
        max_value=int(max_urm2),
        value=int(max_urm2),
        step=1
    )

    if min_urm2 > max_urm2:
        st.error("El valor mínimo de URM2% no puede ser mayor que el valor máximo.")



def aplicar_filtros(df):
        if filtro_subcanal:
            df = df[df['Subcanal'].isin(filtro_subcanal)]
        if pd.notnull(min_input) and pd.notnull(max_input):
            df = df[(df["PP"] >= min_input) & (df["PP"] <= max_input)]
        if filtro_departamento:
            df = df[df['Departamento'].isin(filtro_departamento)]
        if filtro_kam:
            df = df[df['Kam'].isin(filtro_kam)]
        if filtro_socio:
            df = df[df['Socio'].isin(filtro_socio)]
        if pd.notnull(min_input_urm2) and pd.notnull(max_input_urm2):
            df = df[(df["URM2%"] >= min_input_urm2) & (df["URM2%"] <= max_input_urm2)]
        return df


def tabla_inicial(df):
    tabla = df.groupby('Subcanal').agg(
        QHc=('QHc', 'sum'),
        PP=('PP', 'sum'),
        QUrs=('QUrs', 'sum'),
        SS = ('SS', 'sum'),
        SUSM2 = ('SUSM2', 'sum'),
        PERM2 = ('PERM2', 'sum'),
        PagoTotal = ('PagoTotal', 'sum'),
        Acelerador = ('Acelerador', 'sum'),
        Planilla = ('Planilla', 'sum'),
        Bono = ('Bono', 'sum'),
        Campaña = ('Campaña', 'sum'),
        Otros = ('Otros', 'sum')
    ).reset_index()
    tabla['URM2%'] = round((tabla['QUrs'] / tabla['PP']) * 100,1)
    tabla['SUSM2%'] = round(((tabla['SUSM2'] + tabla['PERM2'])/tabla['SS'])*100,1)
    tabla = tabla.sort_values(by='QHc', ascending=False).reset_index(drop=True)
    total_row = pd.DataFrame({
        'Subcanal': ['Total'],
        'QHc': [tabla['QHc'].sum()],
        'PP': [tabla['PP'].sum()],
        'QUrs': [tabla['QUrs'].sum()],
        'URM2%': [round((tabla['QUrs'].sum() / tabla['PP'].sum()) * 100, 1) 
                if tabla['PP'].sum() > 0 else 0],
        'SS': [tabla['SS'].sum()],
        'SUSM2': [tabla['SUSM2'].sum()],
        'PERM2': [tabla['PERM2'].sum()],
        'SUSM2%': [round(((tabla['SUSM2'].sum() + tabla['PERM2'].sum())/tabla['SS'].sum())*100,1)],
        'PagoTotal': [tabla['PagoTotal'].sum()],
        'Acelerador':  [tabla['Acelerador'].sum()],
        'Planilla': [tabla['Planilla'].sum()],
        'Bono':  [tabla['Bono'].sum()],
        'Campaña':  [tabla['Campaña'].sum()],
        'Otros':  [tabla['Otros'].sum()],
    
    })     
    tabla = pd.concat([tabla, total_row], ignore_index=True)
    tabla = tabla.fillna(0)
    return tabla

def tabla_resumen_grupos(dataframe):
    tabla = dataframe.groupby('Grupo').agg(
        RangoGrupo= ('RangoGrupo', 'first'),
        QHc=('QHc', 'sum'),
        PP=('PP', 'sum'),
        QUrs=('QUrs', 'sum'),
        SS = ('SS', 'sum'),
        SUSM2 = ('SUSM2', 'sum'),
        PERM2 = ('PERM2', 'sum'),
        PagoTotal = ('PagoTotal', 'sum'),
        Acelerador = ('Acelerador', 'sum'),
        Planilla = ('Planilla', 'sum'),
        Bono = ('Bono', 'sum'),
        Campaña = ('Campaña', 'sum'),
        Otros = ('Otros', 'sum')
    ).reset_index()

    tabla['URM2%'] = round((tabla['QUrs'] / tabla['PP']) * 100,1)
    tabla['SUSM2%'] = round(((tabla['SUSM2'] + tabla['PERM2'])/tabla['SS'])*100,1)

    total_row = pd.DataFrame({
        'Grupo': ['Total'],
        'RangoGrupo': [' '],
        'QHc': [tabla['QHc'].sum()],
        'PP': [tabla['PP'].sum()],
        'QUrs': [tabla['QUrs'].sum()],
        'URM2%': [round((tabla['QUrs'].sum() / tabla['PP'].sum()) * 100,1) if tabla['PP'].sum() > 0 else 0],
        'SS': [tabla['SS'].sum()],
        'SUSM2': [tabla['SUSM2'].sum()],
        'PERM2': [tabla['PERM2'].sum()],
        'SUSM2%': [round(((tabla['SUSM2'].sum() + tabla['PERM2'].sum())/tabla['SS'].sum())*100,1)],
        'PagoTotal': [tabla['PagoTotal'].sum()],
        'Acelerador':  [tabla['Acelerador'].sum()],
        'Planilla': [tabla['Planilla'].sum()],
        'Bono':  [tabla['Bono'].sum()],
        'Campaña':  [tabla['Campaña'].sum()],
        'Otros':  [tabla['Otros'].sum()],
    })
    tabla = pd.concat([tabla, total_row], ignore_index=True)
    tabla = tabla.fillna(0)
    return tabla
            

#def normal(carpeta_archivos=carpeta_canal):

   
    
# Filro Meses 
def normal():
    filtro_mes = st.selectbox("Mes",  meses_disponibles(carpeta_archivos))
    st.markdown("---")

    try:
        df = dataframe_mes(filtro_mes, carpeta_archivos)
        df = calcular_grupos_personalizados(dataframe= df,num_grupos= num_grupos, columnas_orden=['URM2%', 'QUrs', 'PP'])

        df_filtrado = aplicar_filtros(df)

        if not df_filtrado.empty:
            df_recalculado = calcular_grupos_personalizados(df_filtrado, num_grupos= num_grupos , columnas_orden= ['URM2%', 'QUrs', 'PP'])
            
        else:
            st.warning("No hay datos para los filtros seleccionados.")        
            df_recalculado = pd.DataFrame(columns=df.columns)


        if not df_recalculado.empty:


            # Tabla Resumen Inicial  ------------------------------
            st.markdown("""### :page_facing_up: Resumen Inicial""")
            st.info("""Puede visualizar un **dataframe segmentado por subcanal**, si deseas el detalle del **PagoTotal**, simplemente haz clic  en **"Mostrar adicionales"**.""")
            mostrar_columnas_adicionales_pivote = st.checkbox("Mostrar adicionales .")

            tabla_ini = tabla_inicial(df_recalculado)
            
            if mostrar_columnas_adicionales_pivote:
                columnas_a_mostrar_pivot = ['Subcanal', 'QHc', 'PP', 'QUrs','URM2%', 'SS', 'SUSM2', 'PERM2', 'SUSM2%', 'PagoTotal', 'Acelerador', 'Planilla', 'Bono', 'Campaña', 'Otros']
            else:
                columnas_a_mostrar_pivot = ['Subcanal', 'QHc', 'PP', 'QUrs', 'URM2%', 'SS', 'SUSM2', 'PERM2','SUSM2%', 'PagoTotal'] 
            
            st.dataframe(tabla_ini[columnas_a_mostrar_pivot], use_container_width=True)
            
            # descarga
            towrite_inicial = io.BytesIO()
            with pd.ExcelWriter(towrite_inicial, engine="xlsxwriter") as writer:
                tabla_ini.to_excel(writer, index=False, sheet_name="Tabla Inicial")
            towrite_inicial.seek(0)

            st.download_button(
                label="Descargar tabla inicial",
                data=towrite_inicial,
                file_name="dataframe_inicial.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.markdown("---")
        
            
            # Tabla por grupos ------------------------------
            st.markdown("""### :rocket: Grupos""")
            st.info("""
            Puedes visualizar un **dataframe segmentado por grupos calculados**, el cual muestra la **distribución de QHcs** dentro de cada grupo resultante del cálculo. Si deseas visualizar el detalle del **Pago Total**, simplemente haz clic en la opción **"Mostrar adicionales"**.
            """)
            mostrar_columnas_adicionales_grupos = st.checkbox("Mostrar adicionales ..")

            tabla_gru = tabla_resumen_grupos(df_recalculado)

            if mostrar_columnas_adicionales_grupos:
                columnas_a_mostrar_tabla_grupos = ['Grupo', 'RangoGrupo','QHc', 'PP', 'QUrs', 'URM2%', 'SS', 'SUSM2', 'PERM2', 'SUSM2%', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
                
            else:
                columnas_a_mostrar_tabla_grupos = ['Grupo', 'RangoGrupo', 'QHc', 'PP', 'QUrs', 'URM2%',  'SS', 'SUSM2', 'PERM2', 'SUSM2%','PagoTotal']  
                
            st.dataframe(tabla_gru[columnas_a_mostrar_tabla_grupos], use_container_width=True)

            # descarga
            towrite_resumen = io.BytesIO()
            with pd.ExcelWriter(towrite_resumen, engine="xlsxwriter") as writer:
                tabla_gru.to_excel(writer, index=False, sheet_name="Tabla Grupos")
            towrite_resumen.seek(0)

            st.download_button(
                label="Descargar grupos",
                data=towrite_resumen,
                file_name="dataframe_grupos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.markdown("---")


            # Tabla Detalle DNI  --------------------------------------------------------------------
            st.markdown(f"""### :mag: Tabla detalle {columna_DNI}""")
            st.info("""Puedes visualizar un **dataframe detallado** con los grupos calculados, mostrando los **DNIs asociados a cada grupo**.  Si deseas observar el detalle del **Pago Total**, simplemente haz clic en la opción **"Mostrar adicionales"**.""")

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

            mostrar_columnas_adicionales = st.checkbox("Mostrar adicionales ...")
            
            df_descarga = df_recalculado # df_descarga es igual a df recalculado

            if mostrar_columnas_adicionales:
                columnas_a_mostrar = ['Departamento', 'Socio', 'Kam' ,'Subcanal', columna_DNI, 'Grupo', 'PP', 'QUrs', 'URM2%', 'SS', 'SUSM2', 'PERM2', 'SUSM2%', 'PagoTotal',  'Acelerador' , 'Planilla', 'Bono', 'Campaña', 'Otros']  
            else:
                columnas_a_mostrar = ['Departamento', 'Socio', 'Kam', 'Subcanal', columna_DNI, 'Grupo', 'PP', 'QUrs', 'URM2%', 'SS', 'SUSM2', 'PERM2', 'SUSM2%','PagoTotal']  
                

            if grupos_seleccionados:
                df_descarga = df_recalculado[df_recalculado['Grupo'].isin(grupos_seleccionados)] # si se aplica este filtro es df recalculado segun el grupo seleccionado
            else:
                df_descarga = df_recalculado


            if dni_ingresado: 
                try:
                    dni_ingresado = str(dni_ingresado)  
                    df_descarga = df_descarga[df_descarga[columna_DNI] == dni_ingresado]

                    if df_descarga.empty:  
                        st.warning(f"El {columna_DNI} ingresado no se encuentra en los datos.")
                    else:
                        st.dataframe(df_descarga[columnas_a_mostrar], use_container_width=True)

                except ValueError:
                    st.error(f"Por favor, ingresa un {columna_DNI} con el formao válido.")
            else:  
                st.dataframe(df_descarga[columnas_a_mostrar], use_container_width=True )
            

            # descarga
            towrite_detallada = io.BytesIO()
            with pd.ExcelWriter(towrite_detallada, engine="xlsxwriter") as writer:
                df_descarga.to_excel(writer, index=False, sheet_name="Tabla Detalle DNI")
            towrite_detallada.seek(0)

            st.download_button(
                label="Descargar tabla detalle",
                data=towrite_detallada,
                file_name="dataframe_detalle_dni.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        

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
    


def historico_optimizado(carpeta_archivos=carpeta_archivos):

    st.markdown("""### :bar_chart: Tabla resumen""")
    desempeños = ["Bueno", "Malo", "Neutro"]
    filtro1, filtro2 = st.columns([2, 2])
    with filtro1:
        dni_ingresado = st.text_input(f"Ingresa un {columna_DNI} para buscar su progreso:")
    with filtro2:
        desempeño_ingresado = st.multiselect(
            "Selecciona el desempeño",
            options=desempeños,
            default=[]
        )

    @st.cache_data
    def cargar_datos(carpeta_archivos):
        archivos = os.listdir(carpeta_archivos) 
        archivos = [os.path.join(carpeta_archivos, archivo) for archivo in archivos]
        meses = meses_disponibles(carpeta_archivos)
        resultados = []
        for archivo, mes in zip(archivos, meses):
            df = pd.read_excel(archivo, dtype={columna_DNI: str})
            df = dataframe_mes(mes, carpeta_archivos)
            df['Mes'] = mes
            df = calcular_grupos_personalizados(
                df,
                num_grupos=num_grupos,
                columnas_orden=["URM2%", "QUrs", "PP"]
                )
            resultados.append(df)
        return pd.concat(resultados, ignore_index=True), meses

   

    df_total, meses = cargar_datos(carpeta_archivos)


    df_filtrado = aplicar_filtros(df_total)


    if not df_filtrado.empty:
        todos_dnis = pd.DataFrame(df_filtrado[columna_DNI].unique(), columns=[columna_DNI]).sort_values(by=columna_DNI).reset_index(drop=True)
        df_completo = todos_dnis.copy()

        meses_validos = [mes for mes in meses if not df_filtrado[df_filtrado['Mes'] == mes]['Grupo'].isnull().all()]
        
        for mes in meses_validos:
            df_mes = df_filtrado[df_filtrado['Mes'] == mes][[columna_DNI, 'Grupo']]
            df_mes_completo = todos_dnis.merge(df_mes, on=columna_DNI, how='left').rename(columns={'Grupo': mes})
            df_completo = df_completo.merge(df_mes_completo[[columna_DNI, mes]], on=columna_DNI, how='left').fillna(0)

        #df_completo['Evolución'] = df_completo.apply(
         #   lambda row: [row[mes] for mes in meses_validos],
         #   axis=1
        #)

        df_completo[['Bandera', 'Desempeño']] = df_completo.apply(
            lambda row: pd.Series(asignar_bandera_promedio(row, meses_validos, num_grupos)),
            axis=1
        )

      
        if dni_ingresado:
            df_seleccionado = df_completo[df_completo[columna_DNI] == dni_ingresado]
            if df_seleccionado.empty:
                st.warning(f"El {columna_DNI} ingresado no se encuentra en los datos.")
        else:
            if not desempeño_ingresado:
                df_seleccionado = df_completo
            else:
                df_seleccionado = df_completo[df_completo['Desempeño'].isin(desempeño_ingresado)]

        #st.dataframe(
         #   df_seleccionado,
         #  column_config={
          #      "Evolución": st.column_config.BarChartColumn(
           #         "Evolución de grupos",
           #         y_min=0,
          #          y_max=num_grupos
           #     )
        #    },
          #  use_container_width=True
        #)

        st.dataframe(df_seleccionado, use_container_width=True)


      
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


with tab_vista_normal:
    normal()
with tab_vista_historica:
    historico_optimizado()