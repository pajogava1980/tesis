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

def filtrar_entidades(event):
    texto_ingresado = combo_nombre.get().upper()
    nuevas_entidades = [nombre for nombre in entidades_disponibles if texto_ingresado in nombre]
    combo_nombre["values"] = nuevas_entidades

def calcular_margen_ganancia(fecha, moneda, nombre, tipo_negociacion, df):
    """Aplica filtros y calcula el margen de ganancia"""
    fecha_filtro = datetime.strptime(fecha, "%Y-%m-%d").date()
    df_filtrado = df[
        (df["fecha"] == fecha_filtro) &
        (df["moneda"] == moneda) &
        (df["Nombres"] == nombre) &
        (df["tipo_negociacion"] == tipo_negociacion)
    ].copy()

    if df_filtrado.empty:
        messagebox.showerror("Error", "No hay datos para los filtros seleccionados.")
        return None

    columna_tc = "tc_compra" if tipo_negociacion == "Compra" else "tc_venta"
    columna_monto = "compra_monto_mn" if tipo_negociacion == "Compra" else "venta_monto_mn"

    df_filtrado["Ponderado"] = df_filtrado[columna_monto] * df_filtrado[columna_tc]
    suma_montos = df_filtrado[columna_monto].sum()
    promedio_ponderado = df_filtrado["Ponderado"].sum() / suma_montos if suma_montos > 0 else 0

    df_filtrado["Valor_Tipo_Cambio"] = df_filtrado[columna_tc]
    df_filtrado["Promedio_Ponderado"] = promedio_ponderado
    df_filtrado["Margen_Ganancia"] = df_filtrado[columna_tc] - promedio_ponderado
    
    df_filtrado["Promedio_Ponderado_Commerz"] = 0  # Asegurar que la columna existe
    df_filtrado["Spread"] = 0  # Asegurar que la columna existe
    return df_filtrado[["Valor_Tipo_Cambio", "Promedio_Ponderado", "Promedio_Ponderado_Commerz", "Spread", "Margen_Ganancia"]]


def ejecutar_calculo():
    fecha = entry_fecha.get_date().strftime("%Y-%m-%d")
    moneda = combo_moneda.get()
    nombre = combo_nombre.get()
    tipo_negociacion = combo_tipo.get()
    
    if not fecha or not moneda or not nombre:
        messagebox.showerror("Error", "Por favor, seleccione todos los campos antes de calcular.")
        return
    
    resultado_df = calcular_margen_ganancia(fecha, moneda, nombre, tipo_negociacion, df)
    
    if resultado_df is not None:
        archivo_guardado = f"Margen_Ganancia_{fecha}_{tipo_negociacion}.xlsx"
        resultado_df.to_excel(archivo_guardado, index=False)
        
        #text_resultado.insert("1.0", tk.END, resultado_df.to_string(index=False, columns=["Valor_Tipo_Cambio", "Promedio_Ponderado", "Promedio_Ponderado_Commerz", "Spread", "Margen_Ganancia"]))
        text_resultado.insert("1.0", tk.END, resultado_df.to_string(index=False, columns=["Valor_Tipo_Cambio", "Promedio_Ponderado", "Promedio_Ponderado_Commerz", "Spread", "Margen_Ganancia"]))

        #text_resultado.delete("1.0", tk.END)
        #text_resultado.insert(tk.END, resultado_df.to_string(index=False))
        
        messagebox.showinfo("Éxito", f"Archivo guardado: {archivo_guardado}")

# Cargar datos
df = cargar_datos()

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

entidades_disponibles = obtener_entidades(df)
tk.Label(root, text="Nombre de la Entidad:").pack()
combo_nombre = ttk.Combobox(root, values=entidades_disponibles)
combo_nombre.pack()
combo_nombre.bind("<KeyRelease>", filtrar_entidades)

tk.Label(root, text="Tipo de Negociación:").pack()
combo_tipo = tk.StringVar(value="Compra")
tk.OptionMenu(root, combo_tipo, "Compra", "Venta").pack()

# Botón de ejecución
tk.Button(root, text="Calcular Margen", command=ejecutar_calculo).pack()

# Área de texto para mostrar resultados
tk.Label(root, text="Resultados:").pack()
text_resultado = tk.Text(root, height=10, width=60)
text_resultado.pack()

# Iniciar la interfaz
root.mainloop()
