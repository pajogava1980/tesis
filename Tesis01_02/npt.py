import matplotlib.pyplot as plt
import pandas as pd

# Datos extraídos de las imágenes proporcionadas
tests = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", 
         "T15", "T16", "T17", "T18", "T19", "T20", "T21", "T22"]
delay = [1.451, 1.451, 1.401, 1.401, 1.401, 1.401, 1.401, 1.401, 1.401, 1.401, 1.401, 1.401, 1.401, 1.401, 1.438, 1.438, 1.438, 1.438, 1.447, 1.447, 1.432, 1.400]
offset = [1.921, 1.921, 2.837, 2.837, 2.837, 2.837, 2.837, 2.837, 2.837, 2.837, 2.837, 2.837, 2.837, 2.837, -2.406, -2.406,-2.406, -0.019, 0.722, 0.722, 0.822, -1.678]
jitter = [0.000, 0.291, 1.537, 1.537, 1.537, 1.537, 1.537, 1.537, 1.537, 1.537, 1.537, 1.945, 1.945, 1.945, 1.445, 1.138, 1.138, 0.921, 0.295, 0.315, 2.277, 1.232]
reach = [1, 3, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377, 377]

# Crear DataFrame
data = pd.DataFrame({
    "Test": tests,
    "Delay (ms)": delay,
    "Offset (ms)": offset,
    "Jitter (ms)": jitter,
    "Reach": reach
})

# Gráficas individuales
plt.figure(figsize=(10, 6))

# Gráfica 1: Delay
plt.subplot(4, 1, 1)
plt.plot(data["Test"], data["Delay (ms)"], marker='o', linestyle='-', label="Delay (ms)")
plt.title("Servidor 172.17.120.182 - Delay")
plt.ylabel("Delay (ms)")
plt.grid()
plt.legend()

# Gráfica 2: Offset
plt.subplot(4, 1, 2)
plt.plot(data["Test"], data["Offset (ms)"], marker='o', linestyle='-', label="Offset (ms)", color='orange')
plt.title("Servidor 172.17.120.182 - Offset")
plt.ylabel("Offset (ms)")
plt.grid()
plt.legend()

# Gráfica 3: Jitter
plt.subplot(4, 1, 3)
plt.plot(data["Test"], data["Jitter (ms)"], marker='o', linestyle='-', label="Jitter (ms)", color='green')
plt.title("Servidor 172.17.120.182 - Jitter")
plt.ylabel("Jitter (ms)")
plt.xlabel("Tests")
plt.grid()
plt.legend()

# Gráfica 4: reach
plt.subplot(4, 1, 4)
plt.plot(data["Test"], data["Reach"], marker='o', linestyle='-', label="Reach", color='red')
plt.title("Servidor 172.17.120.182 - Reach")
plt.ylabel("Reach")
plt.xlabel("Pruebas Sync GPS-RTU")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
