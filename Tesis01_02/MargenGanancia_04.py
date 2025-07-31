# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 00:30:46 2025

@author: Paula Gamboa
"""
import pandas as pd
from datetime import datetime

# Ruta del archivo Excel
archivo_excel = "Libro3.xlsx"

# Leer el archivo Excel
df = pd.read_excel(archivo_excel)

# Asegurar que los nombres de las columnas sean correctos
df.columns = ['pd_id_posicion_diaria', 'pd_id_operacion', 'fecha', 'moneda', 'tipo_negociacion', 
              'tipo_negociacion1', 'forma_general', 'identificacion', 'Nombres', 'numero_operacion', 
              'estado', 'venta_monto_me', 'venta_monto_mn', 'tc_venta', 'compra_me', 'compra_monto_mn', 
              'tc_compra', 'tc_posicion', 'pd_utilidad_negocio', 'uo_cotizacion_pool', 
              'uo_utilidad_operacion', 'Oficial', 'Banca', 'Pais_Destino', 'Spread']

# ✅ Convertir la columna 'fecha' a formato datetime correctamente
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date  

# ✅ Asegurar que "moneda" y "Nombres" no tengan espacios y estén en mayúsculas
df["moneda"] = df["moneda"].str.strip().str.upper()
df["Nombres"] = df["Nombres"].str.strip().str.upper()

# Función para calcular el margen de ganancia considerando el tipo de negociación
def calcular_margen_ganancia(df, fecha, moneda, nombre, tipo_negociacion):
    """
    Calcula el Margen de Ganancia filtrando por fecha, moneda, nombre y tipo de negociación.

    Parámetros:
    df (DataFrame): El conjunto de datos original cargado desde Excel.
    fecha (str): La fecha a filtrar en formato 'YYYY-MM-DD'.
    moneda (str): La moneda a filtrar (ejemplo: 'EURO').
    nombre (str): El nombre de la entidad a filtrar (ejemplo: 'COMMERZBANK AG').
    tipo_negociacion (str): Puede ser "Compra" o "Venta".

    Retorna:
    DataFrame con el promedio ponderado y el margen de ganancia calculado.
    """

    # Convertir fecha a datetime.date para asegurar coincidencia con el DataFrame
    fecha_filtro = datetime.strptime(fecha, "%Y-%m-%d").date()

    # ✅ Aplicar los filtros con el nuevo parámetro tipo_negociacion
    df_filtrado = df[
        (df["fecha"] == fecha_filtro) &
        (df["moneda"] == moneda) &
        (df["Nombres"] == nombre) &
        (df["tipo_negociacion"] == tipo_negociacion)
    ].copy()  # Usamos .copy() para evitar advertencias de Pandas

    if df_filtrado.empty:
        print(f" No hay datos para la fecha {fecha}, moneda {moneda}, entidad {nombre} y tipo de negociación {tipo_negociacion}.")
        return None

    # Calcular el promedio ponderado según el tipo de negociación seleccionado
    if tipo_negociacion == "Compra":
        df_filtrado["Ponderado"] = df_filtrado["compra_monto_mn"] * df_filtrado["tc_compra"]
        suma_montos = df_filtrado["compra_monto_mn"].sum()
    else:  # "Venta"
        df_filtrado["Ponderado"] = df_filtrado["venta_monto_mn"] * df_filtrado["tc_venta"]
        suma_montos = df_filtrado["venta_monto_mn"].sum()

    # Evitar divisiones por cero asegurando que la suma sea mayor a cero
    promedio_ponderado = df_filtrado["Ponderado"].sum() / suma_montos if suma_montos > 0 else 0

    # Crear una nueva columna con el Promedio Ponderado
    df_filtrado["Promedio_Ponderado"] = promedio_ponderado

    # Calcular el margen de ganancia usando el nuevo cálculo del PP
    df_filtrado["Margen_Ganancia"] = df_filtrado.apply(
        lambda row: row["tc_venta"] - promedio_ponderado if tipo_negociacion == "Venta" 
        else row["tc_compra"] - promedio_ponderado, axis=1
    )

    return df_filtrado

# Parámetros de prueba (puedes cambiarlos según necesites)
fecha_filtro = "2024-01-02"
moneda_filtro = "EURO"
nombre_filtro = "CASTILLO NARANJO CARLOS ALBERTO"
tipo_negociacion_filtro = "Venta"  # Cambia a "Venta" si deseas filtrar por ventas

# Aplicar la función
resultado_df = calcular_margen_ganancia(df, fecha_filtro, moneda_filtro, nombre_filtro, tipo_negociacion_filtro)

# Guardar los resultados en un archivo Excel con la nueva versión
if resultado_df is not None:
    output_file = f"Margen_Ganancia_{fecha_filtro}_{tipo_negociacion_filtro}_v04.xlsx"
    resultado_df.to_excel(output_file, index=False)

    print(f"Archivo guardado como {output_file}. Ábrelo manualmente en Excel.")
