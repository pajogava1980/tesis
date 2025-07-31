import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from tkcalendar import DateEntry

def cargar_datos():
    """Carga los datos del archivo Excel y estandariza los valores"""
    archivo_excel = "Libro3.xlsx"
    df = pd.read_excel(archivo_excel)
    df.columns = [col.strip() for col in df.columns]  # Eliminar espacios en nombres de columnas
    df["moneda"] = df["moneda"].str.strip().str.upper()
    df["Nombres"] = df["Nombres"].str.strip().str.upper()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date  
    return df

def obtener_monedas(df):
    return sorted(df["moneda"].unique())

def obtener_entidades(df):
    return sorted(df["Nombres"].unique())

def calcular_margen_ganancia(fecha, moneda, df):
    """Aplica filtros y calcula el margen de ganancia basado en COMMERZBANK AG"""
    """Aplica filtros y calcula el margen de ganancia"""
    fecha_filtro = datetime.strptime(fecha, "%Y-%m-%d").date()
    df_filtrado = df[(df["fecha"] == fecha_filtro) & (df["moneda"] == moneda) & (df["Nombres"] == "COMMERZBANK AG")].copy()

    if df_filtrado.empty:
        messagebox.showerror("Error", "No hay datos para los filtros seleccionados.")
        return None

    df_filtrado["Ponderado"] = df_filtrado["compra_monto_mn"] * df_filtrado["tc_compra"]
    suma_montos = df_filtrado["compra_monto_mn"].sum()
    promedio_ponderado = df_filtrado["Ponderado"].sum() / suma_montos if suma_montos > 0 else 0

    df_filtrado["Valor_Tipo_Cambio"] = df_filtrado["tc_compra"]
    df_filtrado["Promedio_Ponderado"] = promedio_ponderado
    df_filtrado["Margen_Ganancia"] = df_filtrado["tc_compra"] - promedio_ponderado
    
    return df_filtrado[["Valor_Tipo_Cambio", "Promedio_Ponderado", "Margen_Ganancia"]]

def calcular_spread_cliente(fecha, moneda, df, promedio_ponderado):
    """Calcula spread_cliente usando tc_venta de cualquier entidad"""
    """Calcula spread_cliente usando tc_venta de cualquier entidad"""
    fecha_filtro = datetime.strptime(fecha, "%Y-%m-%d").date()
    df_filtrado = df[(df["fecha"] == fecha_filtro) & (df["moneda"] == moneda) & (df["tipo_negociacion"] == "Venta")].copy()

    if df_filtrado.empty:
        messagebox.showerror("Error", "No hay datos de ventas para calcular el spread del cliente.")
        return None

    df_filtrado["spread_cliente"] = df_filtrado["tc_venta"] - promedio_ponderado
    return df_filtrado[["tc_venta", "spread_cliente"]]

def ejecutar_calculo():
    global df, promedio_ponderado_actual
    fecha = entry_fecha.get_date().strftime("%Y-%m-%d")
    moneda = combo_moneda.get()
    
    if not fecha or not moneda:
        messagebox.showerror("Error", "Por favor, seleccione todos los campos antes de calcular.")
        return
    
    resultado_df = calcular_margen_ganancia(fecha, moneda, df)
    
    if resultado_df is not None:
        promedio_ponderado_actual = resultado_df["Promedio_Ponderado"].iloc[0]
        archivo_guardado = f"Margen_Ganancia_{fecha}.xlsx"
        resultado_df.to_excel(archivo_guardado, index=False)
        
        text_resultado.delete("1.0", tk.END)
        text_resultado.insert("1.0", resultado_df.to_string(index=False))
        
        messagebox.showinfo("Éxito", f"Archivo guardado: {archivo_guardado}")

def ejecutar_spread_cliente():
    if promedio_ponderado_actual is None:
        messagebox.showerror("Error", "Debe calcular primero el margen de ganancia antes de calcular el spread del cliente.")
        return
    
    fecha = entry_fecha.get_date().strftime("%Y-%m-%d")
    moneda = combo_moneda.get()
    
    resultado_df = calcular_spread_cliente(fecha, moneda, df, promedio_ponderado_actual)
    
    if resultado_df is not None:
        archivo_guardado = f"Spread_Cliente_{fecha}.xlsx"
        resultado_df.to_excel(archivo_guardado, index=False)
        
        text_resultado.delete("1.0", tk.END)
        text_resultado.insert("1.0", resultado_df.to_string(index=False))
        
        messagebox.showinfo("Éxito", f"Archivo guardado: {archivo_guardado}")

# Cargar datos
df = cargar_datos()
promedio_ponderado_actual = None

# Crear ventana principal
root = tk.Tk()
root.title("Cálculo de Margen de Ganancia")
root.geometry("500x400")

# Etiquetas y entradas
tk.Label(root, text="Fecha (YYYY-MM-DD):").pack()
entry_fecha = DateEntry(root, date_pattern="yyyy-mm-dd")
entry_fecha.pack()

monedas_disponibles = obtener_monedas(df)
tk.Label(root, text="Moneda:").pack()
combo_moneda = ttk.Combobox(root, values=monedas_disponibles, state='readonly')
combo_moneda.pack()

# Botones
tk.Button(root, text="Calcular Margen", command=ejecutar_calculo).pack()
tk.Button(root, text="Calcular Spread Cliente", command=ejecutar_spread_cliente).pack()

# Área de texto para mostrar resultados
tk.Label(root, text="Resultados:").pack()
text_resultado = tk.Text(root, height=10, width=60)
text_resultado.pack()

# Iniciar la interfaz
root.mainloop()
