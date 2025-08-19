import pandas as pd
"""
    Para obtener el Margen de Ganancia  necesito que, por: 
    cada fecha, divisa y tipo de operación, se multiplique 
    el tipo de cambio (columna N en caso de venta y columna Q en caso de compra) 
    por el valor en dólares del cambio (columna M en caso de venta y columna P en caso de compra), 
    todo esto dividido para la suma del valor en dólares del cambio 
    (suma de la columna M en caso de venta y P en caso de compra). 
    Una vez con se obtiene el valor, este resta de la columna N en caso de venta o de Q en caso de compra

        (Q1xP1) + (Q2xP2) + (Q3xP3)
    pp = ----------------------------=SUma ponderada
             (P1 + P2 + P3)

             N - X = CUANTÓ GANÉ 
    Q representa el tipo de cambio (N o Q en tu descripción inicial).
    PP representa el monto en dólares del cambio (M o P en tu descripción inicial).
    X es el valor promedio ponderado de la tasa de cambio.
    N - X representa Margen de Ganancia

"""

# Cargar el archivo Excel
file_path = "Libro3.xlsx"  # Asegúrate de cambiar esto a la ruta correcta
xls = pd.ExcelFile(file_path)

df = pd.read_excel(xls, sheet_name='Clientes_Spread (2)')

# Renombrar columnas clave para facilitar el manejo
df.rename(columns={
    'fecha': 'Fecha',
    'moneda': 'Divisa',
    'tipo_negociacion': 'Tipo_Operacion',
    'tc_venta': 'TC_Venta',
    'tc_compra': 'TC_Compra',
    'venta_monto_me': 'Venta_Monto_ME',
    'compra_me': 'Compra_Monto_ME'
}, inplace=True)

# Convertir fechas a formato datetime
#df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df['Fecha'] = pd.to_datetime(df['Fecha'])

#----------------------------------------------------------------------------------------------------------------------------------------
# Agrupar por Fecha y Divisa para calcular los valores totales, ya que queremos calcular el margen para cada combinación de estos valores.
# Estos valores se usarán como denominador en la fórmula.

#Total_Venta_ME es la suma de la columna M (Venta_Monto_ME) por cada Fecha y Divisa.
ventas_totales = df[df['Tipo_Operacion'] == 'Venta'].groupby(['Fecha', 'Divisa'])['Venta_Monto_ME'].sum().rename('Total_Venta_ME')

#Total_Compra_ME es la suma de la columna P (Compra_Monto_ME) por cada Fecha y Divisa.
compras_totales = df[df['Tipo_Operacion'] == 'Compra'].groupby(['Fecha', 'Divisa'])['Compra_Monto_ME'].sum().rename('Total_Compra_ME')

#----------------------------------------------------------------------------------------------------------------------------------------

# Unir los valores totales al dataframe: Para poder hacer cálculos fila por fila, unimos estas sumas al dataframe origina

# Ahora, cada fila en el dataframe tiene la suma total correspondiente a su Fecha y Divisa.
df = df.merge(ventas_totales, on=['Fecha', 'Divisa'], how='left')
df = df.merge(compras_totales, on=['Fecha', 'Divisa'], how='left')

# Calcular el margen de ganancia de manera vectorizada: Ahora aplicamos la fórmula condicionalmente para ventas y compras
df['Margen_Ganancia'] = None
venta_mask = df['Tipo_Operacion'] == 'Venta'    # Para las ventas
compra_mask = df['Tipo_Operacion'] == 'Compra'  # Para las compras

# Resultado: Una vez con se obtiene el valor, este resta de la columna N en caso de venta o de Q en caso de compra

#              X-->SUM PONDERADA        N (Tipo de Cambio en Venta)-->Q   M (Monto en dólares de la venta)-->P     Línea 38 del script--> SUM-->P                 N (Tipo de Cambio en Venta)
df.loc[venta_mask, 'Margen_Ganancia'] = (df.loc[venta_mask, 'TC_Venta'] * df.loc[venta_mask, 'Venta_Monto_ME'] / df.loc[venta_mask, 'Total_Venta_ME']) - df.loc[venta_mask, 'TC_Venta']

#                           X                Q (Tipo de Cambio en Compra)        P (Monto en dólares de la compra)         Línea 41 del script                 Q (Tipo de Cambio en Compra)
df.loc[compra_mask, 'Margen_Ganancia'] = (df.loc[compra_mask, 'TC_Compra'] * df.loc[compra_mask, 'Compra_Monto_ME'] / df.loc[compra_mask, 'Total_Compra_ME']) - df.loc[compra_mask, 'TC_Compra']

# Guardar el resultado en un nuevo archivo
output_file = "margen_ganancia.xlsx"
df.to_excel(output_file, index=False)

print(f"Cálculo completado. Archivo guardado en: {output_file}")
