import numpy as np
pos_max_tap = 16
pos_min_tap = -16
tap_steps = 33
max_reg_perce = 0.10  # +-10% de regulación
sensitivity = (max_reg_perce * 2) / (tap_steps - 1)



num_muestras = 1
tap_position = np.random.randint(pos_min_tap, pos_max_tap + 1, num_muestras)
print(tap_position)