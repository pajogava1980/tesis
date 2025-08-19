import pandas as pd

# Cargar el archivo Excel
archivo_excel = 'Libro3 (1).xlsx'
df = pd.read_excel(archivo_excel)

# Asegurar que los nombres de las columnas sean correctos
df.columns = ['pd_id_posicion_dir', 'pd_id_operacion', 'fecha', 'moneda', 'tipo_negociacion', 
              'tipo_negociacion_codigo', 'forma_general', 'identificacion', 'nombres', 'numero_operacion', 
              'estado', 'venta_monto_n', 'venta_tc', 'tc_venta', 'compra_monto_n', 'compra_tc', 
              'tc_compra', 'tc_posicion', 'pd_utilidad_nego', 'uo_cotizacion_p', 'uo_utilidad_operacion',
              'Oficial', 'Banca', 'Pais_Destino', 'Spread']

# Convertir la columna Fecha al tipo datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Filtrar solo las transacciones de compra o venta válidas
df = df[(df['tc_compra'] > 0) | (df['tc_venta'] > 0)]
print(df)

# Crear columnas para identificar el tipo de cambio y volumen asociados
df['tipo_cambio'] = df.apply(lambda x: x['tc_compra'] if x['tipo_negociacion'] == 'Compra' else x['tc_venta'], axis=1)
df['volumen'] = df.apply(lambda x: x['compra_monto_n'] if x['tipo_negociacion'] == 'Compra' else x['venta_monto_n'], axis=1)

# Calcular el promedio ponderado por fecha, moneda y tipo de negociación
weighted_avg = df.groupby(['fecha', 'moneda', 'nombres','identificacion', 'tipo_negociacion']).apply(
    lambda x: (x['tipo_cambio'] * x['volumen']).sum() / x['volumen'].sum()
).reset_index()

# Renombrar la columna resultante
weighted_avg.columns = ['Fecha', 'Moneda', 'Nombres','Identificacion', 'Tipo de Negociación', 'Promedio Ponderado']

# Guardar los resultados en un nuevo archivo Excel
output_file = 'resultado_por_tipo_operacion_03.xlsx'
weighted_avg.to_excel(output_file, index=False, sheet_name='Promedios Ponderados')

print(f'Resultado guardado en {output_file}')
