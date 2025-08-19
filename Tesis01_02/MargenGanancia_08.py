import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkcalendar import DateEntry

def obtener_monedas():
    # Leer el archivo Excel
    archivo_excel = "Libro3.xlsx"
    df = pd.read_excel(archivo_excel)
    df["moneda"] = df["moneda"].str.strip().str.upper()
    return sorted(df["moneda"].unique())

def calcular_margen_ganancia(fecha, moneda, nombre, tipo_negociacion):
    # Leer el archivo Excel
    archivo_excel = "Libro3.xlsx"
    df = pd.read_excel(archivo_excel)

    # Asegurar que los nombres de las columnas sean correctos
    df.columns = ['pd_id_posicion_diaria', 'pd_id_operacion', 'fecha', 'moneda', 'tipo_negociacion', 
                  'tipo_negociacion1', 'forma_general', 'identificacion', 'Nombres', 'numero_operacion', 
                  'estado', 'venta_monto_me', 'venta_monto_mn', 'tc_venta', 'compra_me', 'compra_monto_mn', 
                  'tc_compra', 'tc_posicion', 'pd_utilidad_negocio', 'uo_cotizacion_pool', 
                  'uo_utilidad_operacion', 'Oficial', 'Banca', 'Pais_Destino', 'Spread']

    # Convertir la columna 'fecha' a formato datetime
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date  
    df["moneda"] = df["moneda"].str.strip().str.upper()
    df["Nombres"] = df["Nombres"].str.strip().str.upper()

    # Aplicar filtros
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

    # Calcular el promedio ponderado
    if tipo_negociacion == "Compra":
        df_filtrado["Ponderado"] = df_filtrado["compra_monto_mn"] * df_filtrado["tc_compra"]
        suma_montos = df_filtrado["compra_monto_mn"].sum()
        df_filtrado["Valor_Tipo_Cambio"] = df_filtrado["tc_compra"]
    else:
        df_filtrado["Ponderado"] = df_filtrado["venta_monto_mn"] * df_filtrado["tc_venta"]
        suma_montos = df_filtrado["venta_monto_mn"].sum()
        df_filtrado["Valor_Tipo_Cambio"] = df_filtrado["tc_venta"]
    
    promedio_ponderado = df_filtrado["Ponderado"].sum() / suma_montos if suma_montos > 0 else 0
    df_filtrado["Promedio_Ponderado"] = promedio_ponderado

    df_filtrado["Margen_Ganancia"] = df_filtrado.apply(
        lambda row: row["tc_venta"] - promedio_ponderado if tipo_negociacion == "Venta" 
        else row["tc_compra"] - promedio_ponderado, axis=1
    )
    
    return df_filtrado[["Valor_Tipo_Cambio", "Promedio_Ponderado", "Margen_Ganancia"]]

def ejecutar_calculo():
    fecha = entry_fecha.get_date().strftime("%Y-%m-%d")
    moneda = combo_moneda.get()
    nombre = entry_nombre.get().upper()
    tipo_negociacion = combo_tipo.get()
    
    resultado_df = calcular_margen_ganancia(fecha, moneda, nombre, tipo_negociacion)
    
    if resultado_df is not None:
        archivo_guardado = f"Margen_Ganancia_{fecha}_{tipo_negociacion}.xlsx"
        resultado_df.to_excel(archivo_guardado, index=False)
        
        # Mostrar valores seleccionados en la interfaz
        text_resultado.delete("1.0", tk.END)
        text_resultado.insert(tk.END, resultado_df.to_string(index=False))
        
        messagebox.showinfo("Éxito", f"Archivo guardado: {archivo_guardado}")

# Crear ventana principal
root = tk.Tk()
root.title("Cálculo de Margen de Ganancia")
root.geometry("500x400")

# Etiquetas y entradas
tk.Label(root, text="Fecha (YYYY-MM-DD):").pack()
entry_fecha = DateEntry(root, date_pattern="yyyy-mm-dd")
entry_fecha.pack()

# Obtener monedas disponibles
monedas_disponibles = obtener_monedas()
tk.Label(root, text="Moneda:").pack()
combo_moneda = ttk.Combobox(root, values=monedas_disponibles)
combo_moneda.pack()
combo_moneda.current(0)

tk.Label(root, text="Nombre de la Entidad:").pack()
entry_nombre = tk.Entry(root)
entry_nombre.pack()

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
