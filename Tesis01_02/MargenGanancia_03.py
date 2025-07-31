# Cargar el archivo Excel proporcionado por el usuario
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

# Función para calcular el margen de ganancia con datos desde Excel
def calcular_margen_ganancia(df, fecha, moneda, nombre):
    """
    Calcula el Margen de Ganancia filtrando por fecha, moneda y nombre.

    Parámetros:
    df (DataFrame): El conjunto de datos original cargado desde Excel.
    fecha (str): La fecha a filtrar en formato 'YYYY-MM-DD'.
    moneda (str): La moneda a filtrar (ejemplo: 'EURO').
    nombre (str): El nombre de la entidad a filtrar (ejemplo: 'COMMERZBANK AG').

    Retorna:
    DataFrame con el promedio ponderado y el margen de ganancia calculado.
    """

   # Convertir fecha a datetime.date para asegurar coincidencia con el DataFrame
    fecha_filtro = datetime.strptime(fecha, "%Y-%m-%d").date()

    # Filtrar datos según los criterios
    df_filtrado = df[
        (df["fecha"] == fecha_filtro) &
        (df["moneda"] == moneda) &
        (df["Nombres"] == nombre)
    ]

    if df_filtrado.empty:
        print(f" No hay datos para la fecha {fecha}, moneda {moneda} y entidad {nombre}.")
        return None

    # ✅ Calcular el promedio ponderado según la fórmula:
    # Promedio Ponderado = SUM(Q * P) / SUM(P)
    df_filtrado.loc[:, "Ponderado"] = df_filtrado["compra_monto_mn"] * df_filtrado["tc_compra"]
    promedio_ponderado = df_filtrado["Ponderado"].sum() / df_filtrado["compra_monto_mn"].sum()

    # Calcular el margen de ganancia
    df_filtrado.loc[:, "Margen_Ganancia"] = df_filtrado.apply(
        lambda row: row["tc_venta"] - promedio_ponderado if row["tipo_negociacion"] == "Venta" 
        else row["tc_compra"] - promedio_ponderado, axis=1
    )

    return df_filtrado

# ✅ Parámetros de prueba (puedes cambiarlos según necesites)
fecha_filtro = "2024-01-03"
moneda_filtro = "EURO"
nombre_filtro = "COMMERZBANK AG"

# Aplicar la función
resultado_df = calcular_margen_ganancia(df, fecha_filtro, moneda_filtro, nombre_filtro)

# Guardar los resultados en un archivo Excel si hay datos
if resultado_df is not None:
    output_file = "Margen_Ganancia_Actualizado_01.xlsx"
    resultado_df.to_excel(output_file, index=False)

    print(f"✅ Archivo guardado como {output_file}. Ábrelo manualmente en Excel.")
