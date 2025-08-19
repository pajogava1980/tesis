# -*- coding: utf-8 -*-
#START
import matplotlib.pyplot as plt
import matlab.engine
import numpy as np
'''
Creado 27-08-2024 11:36 p.m.

@author: pablo.gamboa

Ejemplo de DP Policy iteration applied to frozenLake, 4 acciones, son las direcciones

0 = left
1 = down
2 = right
3 = up

Agent debe moverse en el lado opuesto de la cuadricula sin caer en los hoyos.
Los movimientos son inciertos, el agente-Policy puede moverse en otras direcciones.
Una recompensa de +1 es entregada cuando el objeativo es alcansado-END
OJO, se lo usa para encontrar los valores óptimos de las POLICYs.

---------------------------
|START |  B   |  B  |  B  |
0  0      1      2     3
---------------------------
|  B   | HOLE |  B  | HOLE|
1   4     5      6     7
---------------------------
|  B   |  B   |  B  | HOLE|
2   8    9      10     11
---------------------------
| HOLE |  B   |  B  | END |
3  12    13      14    15
---------------------------

PlicyIteration.py corre de forma iterativa, la polítca de evalacuación y mejora.

Antes se usaba una matriz de transición 'P' para una transición dinámica de una red discreta de un ENV,
especifica para 'Frozen Lake', aqui se caracteriza por S_t, A_t y transisicones S_(t+1).

En este caso, usamos un modelo de SImulink para determinar los S_t y R_t

'''
#INICIALIZACIÓN
# Defino las constantes del entorno
nS = 16         # Número de S_t 4x4-->grid
nA = 4          # Numero de A_t left, down, right, up
gamma = 0.99    # Factor de descuento
eps = 1e-6      # Umbral de estabilidad

eng = matlab.engine.start_matlab()   # Inicio el motor de Matlab
eng = matlab.engine.connect_matlab() # Quita el # de la ventana de MATLAB si ya estamos compartiendo Matlab

# Cargo el modelo y parámetros de Simulink
eng.load_system('AC_Feeder_Control')
#eng.open_system('AC_Feeder_Control', nargout=0)
eng.run('AC_Feeder_Control_Param_02.m', nargout=0)

# Defino mi política inicial aleatoria, escojo una A_t por cada S_t
policy = np.random.choice(nA, size=nS)

# Inicializo mi función F_value
V = np.zeros(nS)

# Almacenamiento de los valores Y_reg y tap_position para graficar
Y_reg_val = []
tap_pos_val = []

#-----------------------------------------------------------------------------------------------------
def eval_state_action(V, Y_reg, a, gamma = 0.99):
    """_summary: En esta función voy a adaptar la Ec. de Bellma a mi ENV de Simulink, en donde voy a
    calcular las S_t y los R_t de forma dinámica.

    Args:
        V (np.array): El actual ValueFunction
        s (int): S_t
        a (int): A_t a ser tomada
        gamma (float): Factor de descuento. Defaults to 0.99.

    Returns:
        _type_: _description_
    """
    print(V, Y_reg, a)
    #Determino la posición del TAP basado en S_t y la A_t
    tap_position = get_tap_position(Y_reg, a)

    print(tap_position)

    #Set la posición en el modelo de Simulink.
    eng.set_param('AC_Feeder_Control/Tap', 'Value', str(tap_position))

    #Corro el modelo de Simulink
    eng.set_param('AC_Feeder_Control', 'SimulationCommand', 'start')
    eng.pause(3)  # Pausa para sincronización de la simulación

    # recupero (retrive) lasalida del modelo de Simulink
    matObj = eng.workspace['matObj']    # Obtengo datos de la simulación de 'matObj'
    Y_reg = matObj.Vreg[0][-1]          # De cada iteración el útimo valor de Vreg
    print(Y_reg)

    # Paro la simulación
    eng.set_param('AC_Feeder_Control', 'SimulationCommand', 'stop')

    #Determino mi S_(t+1) con base en mi output
    next_state = get_next_state_from_simulation_output(Y_reg)

    #Calculo el R_tcon base en mi output Y_reg, llamando a la función calculate_reward
    reward = calculate_reward(Y_reg)

    #Reviso si el S_(t+1) es el final, llamando a la funcion is_terminal_state
    done = is_terminal_state(next_state)

    #Se guarda en las listas de almacenamiento
    Y_reg_val.append(Y_reg)
    tap_pos_val.append(tap_position)

    if done:
        return reward
    else:
        # Actualización de la ecuación de Bellman
        return reward + gamma*V[next_state]

#-----------------------------------------------------------------------------------------------------
def policy_evaluation (V, policy, eps):
    while True:             # Formula 3.8 pag. 62, pseudocode pg. 64 miesntras PI no es estable
        delta = 0           # la función es etable cuando delta sea < que eps
        for s in range(nS): # Lazo para todos los estados nS = 16-->cuadriculas estados de observación
            old_v = V[s]

            # Obtengo los resultados de la simulación y los utilizo para modificar R_t o los S_t
            V[s] = eval_state_action(V, s, policy[s])
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < eps:
            break

#-----------------------------------------------------------------------------------------------------
def policy_improvement(V, policy):
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]           #Acción anterior es tomada del array de la policy en la ubicación 's'
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(nA)])
        if old_a != policy[s]:
            policy_stable = False   #Significa que la policy[s] no es estable
    return policy_stable            #Cuando todo se cumpla

#-----------------------------------------------------------------------------------------------------
def get_next_state_from_simulation_output(Y_reg):
    """_summary: Mapea la salida output en el S_t

    Args:
        Y_reg (float): El voltaje regulado de salida para la simulación

    Returns:
        int: S_(t+1) corresponde al dado por Y_reg
    """
    if Y_reg < 0.9:
        return 0                # S_(t+1)= State 0: El Y_reg es muy bajo
    elif 0.9 <= Y_reg < 0.95:
        return 1                # S_(t+1)= State 1: El Y_reg es un poco bajo
    elif 0.95 <= Y_reg < 1.05:
        return 2                # S_(t+1)= State 2: El Y_reg esta en el rango deseado
    elif 1.05 <= Y_reg < 1.1:
        return 3                # S_(t+1)= State 3: El Y_reg es un poco alto
    else:
        return 4                # S_(t+1)= State 4: El Y_reg es muy alto
#-----------------------------------------------------------------------------------------------------
def is_terminal_state(next_state):
    """_summary: Revisa si un S_t dato es terminal basado en las condiciones de predicción.

    Args:
        next_state (int): El estado a revisar

    Returns:
        bool: True si es terminal, False si es otro.
    """
    terminal_state = [0,4]          #Considerar los state de la función get_next_state_from_simulation_output
    if next_state in terminal_state:
        return True
    else:
        return False
#-----------------------------------------------------------------------------------------------------
def get_tap_position(Y_reg,a):
    """_summary: Determina la posición del TAP basada en S_t y A_t. La tap_position se calcula
    considerando el S_t y aplicando un ajuste con base en A_t.
    Se asegura que los valores del TAP esten dentro de los rangos min y max definidos.

    Args:
        Y_reg (_type_): Y_reg, estado actual
        a (_type_): A_t, puede corresponder a un incremento o decremento pequeño en la posición
        del TAP.

    Returns:
        _type_: _description_
    """
    pos_max_tap = 20
    pos_min_tap = -20

    #Sensibilidad del TAP
    sensitivity = 3

    #Posiciones base del TAP dependiendo de los S_t
    if Y_reg < 0.95:
        base_tap_position = sensitivity  #Significativamente bajo voltaje
    elif Y_reg > 1.05:
        base_tap_position = -sensitivity   #Suave bajo el voltaje
    else:
        base_tap_position = 0

    # Ajuste de la posición del TAP con A_t
    if a == 0:  #A_t 0: Bajar el TAP
        tap_position = base_tap_position - sensitivity
    elif a == 1:    #A_t 1: Subir el TAP
        tap_position = base_tap_position + sensitivity
    else:
        tap_position = base_tap_position

    # Se asegura que la posición del TAP este en los rangos establecidos
    tap_position = max(pos_min_tap, min(tap_position, pos_max_tap))

    return tap_position

#-----------------------------------------------------------------------------------------------------
def calculate_reward(Y_reg):
    """_summary_: Esta función es directa, revisa si los valores del Vreg('Y_reg') están entre los valores
    máximos y mínimos aceptables para el sistema. Se la llama en eval_state_action.

    Args:
        Y_reg: Voltaje regulado--> Y_reg

    Returns:
        _type_: _description_:
    """
    #Son los rangos de valores esperado para el voltaje regulado
    desired_min = 0.95    # 95% del Vreg p.u.
    desired_max = 1.05    # 105% del Vreg p.u.

    #Defino los valores de R_t
    reward_in_range = 1.0    #R_t para mi Y_reg esperado
    reward_out_range = -1.0 #R_T penalizado para un Y_reg fuera de los rangos

    if desired_min <= Y_reg <= desired_max:
        return reward_in_range
    else:
        return reward_out_range

#-----------------------------------------------------------------------------------------------------
def plot_simulation_results(Y_reg_val, tap_pos_val):
    plt.figure(figsize=(12,6))

    plt.subplot(2, 1, 1)
    plt.plot(Y_reg_val, label = 'Y_reg')
    plt.xlabel('Time step')
    plt.ylabel('Y_reg')
    plt.title('Regulación del Voltaje en el Tiempo')
    plt.legend()

    plt.subplot(2, 1, 1)
    plt.plot(tap_pos_val, label = 'Tap Position', color='orange')
    plt.xlabel('Time step')
    plt.ylabel('Tap Position')
    plt.title('Posición del TAP en el tiempo')
    plt.legend()

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------
def run_episodes(policy, num_games):
    tot_rew = 0
    for _ in range(num_games):
        state = 0
        done = False
        while not done:
            action = policy[state]
            reward = eval_state_action(V, state, action)
            next_state = get_next_state_from_simulation_output(reward)
            done = is_terminal_state(next_state)
            state = next_state
            tot_rew += reward
    print('Gano %i de %i juegos!'%(tot_rew, num_games))

#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #Ciclo principal
    policy_stable = False                               #Inicializo la Política PI FALSE = no es etable
    it = 0                                              #it = iteración

    while not policy_stable:                            #Mientras que plicy_stable no cambie a True..
        policy_evaluation (V, policy, eps)              #Primer paso PI
        if policy_improvement(V, policy):               #Segundo paso PI'
            break
        it += 1

    print('Convergencia despues de %i  interaciones --> policy (Politicas)'%(it))
    run_episodes(policy,1000)
    print("\n La matriz de la Funcion del Valor Vpi: ",V.reshape((4,4)))
    print("\n La matriz de la politica PI es: ", policy.reshape((4, 4)))

    #llamo a mi función para graficar
    plot_simulation_results(Y_reg_val, tap_pos_val)

    #Cierro Matlab
    eng.quit()