import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Crear datos
x = np.linspace(0, 2 * np.pi, 1000)  # Valores entre 0 y 2π
seno = np.sin(x)
coseno = np.cos(x)


# 1. Gráfica simple de seno y coseno
plt.figure(figsize=(10, 6))
plt.plot(x, seno, label='Seno', linewidth=2)
plt.plot(x, coseno, label='Coseno', linewidth=2)
plt.title("Seno y Coseno - Gráfica Simple")
plt.xlabel("Ángulo [radianes]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# 2. Gráfica con estilos personalizados
plt.figure(figsize=(10, 6))
plt.plot(x, seno, linestyle='--', color='r', label='Seno (línea punteada)')
plt.plot(x, coseno, linestyle='-.', color='b', label='Coseno (línea guiones)')
plt.title("Seno y Coseno - Estilos Personalizados")
plt.xlabel("Ángulo [radianes]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# 3. Gráfica en subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(x, seno, color='g')
axs[0].set_title("Función Seno")
axs[0].set_xlabel("Ángulo [radianes]")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)

axs[1].plot(x, coseno, color='purple')
axs[1].set_title("Función Coseno")
axs[1].set_xlabel("Ángulo [radianes]")
axs[1].set_ylabel("Amplitud")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# 4. Gráfica polar de seno y coseno
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(x, seno, label="Seno")
ax.plot(x, coseno, label="Coseno")
ax.set_title("Gráfica Polar")
ax.legend()
plt.show()

# 5. Gráfica con relleno entre las curvas
plt.figure(figsize=(10, 6))
plt.plot(x, seno, label='Seno', color='blue', linewidth=2)
plt.plot(x, coseno, label='Coseno', color='red', linewidth=2)
plt.fill_between(x, seno, coseno, where=(seno > coseno), color='blue', alpha=0.2, label="Seno > Coseno")
plt.fill_between(x, seno, coseno, where=(seno < coseno), color='red', alpha=0.2, label="Coseno > Seno")

plt.title("Seno y Coseno con Relleno")
plt.xlabel("Ángulo [radianes]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# 6. Gráfica con puntos y líneas combinadas
plt.figure(figsize=(10, 6))
plt.plot(x, seno, 'o-', label="Seno (Puntos y líneas)", markevery=50, markersize=5)
plt.plot(x, coseno, 's--', label="Coseno (Cuadrados y líneas punteadas)", markevery=50, markersize=5)

plt.title("Seno y Coseno con Puntos y Líneas")
plt.xlabel("Ángulo [radianes]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# 7. Gráfica con colores degradados (color map)
colors = cm.viridis(np.linspace(0, 1, len(x)))

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar el seno con el degradado de colores
for i in range(len(x) - 1):
    ax.plot(x[i:i+2], seno[i:i+2], color=colors[i], linewidth=2)

# Crear un ScalarMappable para la barra de colores
sm = plt.cm.ScalarMappable(cmap=cm.viridis)
sm.set_array([])  # Necesario para asociar el mappable a la barra de colores

# Agregar la barra de colores
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Degradado de Colores')

# Personalizar la gráfica
ax.set_title("Seno con Degradado de Colores")
ax.set_xlabel("Ángulo [radianes]")
ax.set_ylabel("Amplitud")
ax.grid(True)

plt.show()

# 8. Gráfica 3D del seno y coseno
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Crear datos para la gráfica 3D
z = np.linspace(0, 2 * np.pi, 1000)
x = np.sin(z)
y = np.cos(z)

ax.plot3D(x, y, z, label="Seno y Coseno en 3D", color='purple')
ax.set_title("Gráfica 3D del Seno y Coseno")
ax.set_xlabel("Eje X (Seno)")
ax.set_ylabel("Eje Y (Coseno)")
ax.set_zlabel("Eje Z (Ángulo)")

plt.legend()
plt.show()

# 9. Configurar la figura
fig, ax = plt.subplots(figsize=(10, 6))
line_sin, = ax.plot([], [], label='Seno', color='blue')
line_cos, = ax.plot([], [], label='Coseno', color='red')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.1, 1.1)
ax.set_title("Animación del Seno y Coseno")
ax.set_xlabel("Ángulo [radianes]")
ax.set_ylabel("Amplitud")
ax.legend()

# Función de inicialización
def init():
    line_sin.set_data([], [])
    line_cos.set_data([], [])
    return line_sin, line_cos

# Función de actualización
def update(frame):
    x = np.linspace(0, frame, 500)
    line_sin.set_data(x, np.sin(x))
    line_cos.set_data(x, np.cos(x))
    return line_sin, line_cos

# Crear la animación
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 100), init_func=init, blit=True)

plt.show()