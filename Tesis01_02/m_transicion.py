import gym
import pandas as pd

# Cargar el entorno FrozenLake 4x4 sin deslizamiento
env = gym.make("FrozenLake-v1", is_slippery=False)

# Obtener la matriz de transición P(s' | s, a)
transition_matrix = env.P

# Formatear los datos en una estructura de tabla
data = []
for s in range(16):  # 16 estados
    for a in range(4):  # 4 acciones
        for prob, next_state, reward, done in transition_matrix[s][a]:
            data.append([s, a, next_state, prob, reward, done])

# Crear DataFrame
df_transition = pd.DataFrame(data, columns=["Estado", "Acción", "Siguiente Estado", "Probabilidad", "Recompensa", "Terminal"])

# Mostrar la matriz de transición
print(df_transition)
