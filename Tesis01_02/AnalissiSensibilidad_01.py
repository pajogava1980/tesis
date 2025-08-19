import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos de voltajes antes y después del cambio de tap
before = pd.read_csv('voltages_before.csv')  # Voltajes antes del cambio de tap
after = pd.read_csv('voltages_after.csv')    # Voltajes después del cambio de tap

# Calcular la diferencia en voltajes
diff = after['Voltage'] - before['Voltage']

# Graficar la sensibilidad de las barras
plt.figure(figsize=(10, 6))
plt.plot(before['Bus'], diff, marker='o', linestyle='-', color='b')
plt.title('Sensibilidad de Voltajes en las Barras ante Cambios de Tap')
plt.xlabel('Barras')
plt.ylabel('Cambio en el Voltaje (pu)')
plt.grid(True)
plt.show()
