
###### TABLA PIVOT
if not df_recalculado.empty:
    st.subheader("Detalle por Subcanal")
    
    # Crear la tabla pivote
    tabla_pivote = df_recalculado.pivot_table(
        index=['DECIL', 'TIPO'],
        values=['HC', 'Urs', 'QVENTAS', 'FLAG 30', 'Q JUL'],
        aggfunc='sum'
    ).reset_index()
    
    # Calcular columnas adicionales
    tabla_pivote['URM2%'] = (tabla_pivote['Urs'] / tabla_pivote['QVENTAS']) * 100
    tabla_pivote['QNP%'] = (tabla_pivote['FLAG 30'] / tabla_pivote['Q JUL']) * 100
    tabla_pivote['QNP%'] = tabla_pivote['QNP%'].fillna(0)
    
    # Calcular totales
    totales = pd.DataFrame(tabla_pivote[['HC', 'Urs', 'QVENTAS', 'FLAG 30', 'Q JUL']].sum()).T
    totales['TIPO'] = 'Total'
    totales['DECIL'] = ''
    totales['URM2%'] = (totales['Urs'] / totales['QVENTAS']) * 100 if totales['QVENTAS'].iloc[0] != 0 else 0
    totales['QNP%'] = (totales['FLAG 30'] / totales['Q JUL']) * 100 if totales['Q JUL'].iloc[0] != 0 else 0
    
    # Ajustar el orden de las columnas
    totales = totales[['TIPO', 'DECIL', 'HC', 'Urs', 'QVENTAS', 'URM2%', 'QNP%', 'Q JUL']]
    tabla_pivote = pd.concat([tabla_pivote, totales], ignore_index=True)
    
    # Mostrar la tabla

    tabla_renombrada = tabla_pivote.rename(columns={
    'DECIL': 'Decil',
    'TIPO': 'Subcanal',
    'HC': 'QHc',
    'QVENTAS': 'QVentas',
    'Urs': 'QUrs',
    'URM2%': 'URM2%',
    'Q JUL': 'SS',
    'QNP%': 'QNP%'
    })
    
    st.dataframe(tabla_renombrada[['Decil','Subcanal', 'QHc','QVentas', 'QUrs', 'URM2%', 'SS', 'QNP%']], use_container_width=True)

    # Descarga
    towrite_pivote = io.BytesIO()
    with pd.ExcelWriter(towrite_pivote, engine="xlsxwriter") as writer:
        tabla_pivote.to_excel(writer, index=False, sheet_name="Tabla Pivot")
    towrite_pivote.seek(0)

    st.download_button(
        label="Descargar",
        data=towrite_pivote,
        file_name="tabla_pivot.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
