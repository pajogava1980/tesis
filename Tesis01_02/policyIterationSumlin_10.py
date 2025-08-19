# -*- coding: utf-8 -*-
#START
import matplotlib.pyplot as plt
import matlab.engine
import numpy as np
import h5py
import time
import os   #Me ayuda a garantizar que la lectura del simulink sea correcta
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
class PolicyIterationAgent:
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def __init__(self, nS, nA, gamma, eps, eng):
        self.nS = nS #El porque debe ser mas o menos este valor ????
        self.nA = nA #Por las acciones binarias... 0/1 siempre multiplo de 2
        self.gamma = gamma
        self.eps = eps
        self.eng = eng
        self.policy = np.random.choice(nA, size=nS)
        self.V = np.zeros(nS)
        self.Y_reg_val = []
        self.tap_pos_val = []
        self.initial_Y_reg = 1.0                    #Valor asumido inicialmente
        self.V_nominal = 13.8e3
        self.V_base_fase = self.V_nominal/np.sqrt(3)
        self.pausa = eng.workspace['T']*1

        #self.clk = False
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def eval_state_action(self, Y_reg, a):
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
        #Verifico el estado de la simulacion
        sim_status = eng.get_param('AC_Feeder_Control', 'SimulationStatus')

        #Empiezo la simualción si aun no esta corriendo
        if sim_status == 'stopped':
            print("La simualción esta parada, empezando la simulación...")
            eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
            time.sleep(self.pausa)
        elif sim_status == 'pause':
            print("La simulación esta pausada, continuando con la simulación...")
            eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')", nargout=0)
            time.sleep(self.pausa)
        elif sim_status == 'running':
            print("La simulación esta correindo.")

        #Actualizo la posición del TAP
        tap_position = self.get_tap_position(Y_reg, a)
        eng.workspace['tap'] = float(tap_position) #Escribo el valor del TAP en el workspace
        eng.eval("assignin('base', 'tap', tap)", nargout=0)

        #eng.set_param('AC_Feeder_Control/Tap','Value',str(tap_position))
        eng.eval("set_param('AC_Feeder_Control/Tap','Value','tap')", nargout=0)
        
        clk = eng.workspace['clk']      #Leo lo que tengo en el workspace
        eng.workspace['clk'] = not clk  #Cambio en el Workspace el valor
        eng.eval("assignin('base', 'clk', clk)", nargout=0)

        eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)

        self.tap_pos_val.append(tap_position)
        #time.sleep(self.pausa)
        eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'pause')", nargout=0)
        #eng.set_param('AC_Feeder_Control', 'SimulationCommand', 'pause')
        time.sleep(self.pausa)

        # Resume the simulation
        Y_reg_end = self.matObj()
        # Ser evisa por valores extremadamente bajos 
        if Y_reg_end is None:
            print("Error: Y_reg_end  es None. Revisar el archivo .mat")
            return -1, Y_reg    #Estoy devolviendo un reward -1, pq el valor es muy bajo
        elif Y_reg_end < 0.9 or Y_reg_end > 1.1:
            print("Error: Y_reg_end  esta fuera del rango aceptable")
            return -1, Y_reg    #Estoy devolviendo un reward -1, pq el valor es muy bajo
        
        #eng.set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')
        eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')", nargout=0)
        time.sleep(self.pausa)

        self.Y_reg_val.append(Y_reg_end)
        next_state = self.get_next_state_from_simulation_output(Y_reg_end)
        #Calculo el R_tcon base en mi output Y_reg, llamando a la función calculate_reward
        reward = self.calculate_reward(Y_reg_end)
        #Reviso si el S_(t+1) es el final, llamando a la funcion is_terminal_state
        done = self.is_terminal_state(next_state)

        if done:
            return reward, Y_reg_end
        else:
            # Actualización de la ecuación de Bellman
            return reward + self.gamma*self.V[next_state], Y_reg_end
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def matObj(self):
        """_summary_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        file_path = 'Register.mat'
        try:
            if not os.path.exists(file_path):
                print(f"Error: El archivo '{file_path}' no existe.")
                return None
            max_wait_time = 5   #Segundos 
            time_waited = 0
            while True:
                try:
                    with h5py.File(file_path, 'r') as mat_file:
                        print("Claves en el archivo:", list(mat_file.keys()))
                        break   #El archivo es accesible.
                except OSError:
                    time.sleep(0.1)
                    time_waited += 0.1
                    if time_waited > max_wait_time:
                        print("Error: El tiempo de espera para 'Register.mat' se agoto.")
                        return None

            # Abrir el archivo .mat
            with h5py.File(file_path, 'r') as mat_file:
                if 'Vreg' not in mat_file:
                    print("Error: La variable 'Vreg' esta vacía.")
                    return None
                vreg_data = np.array(mat_file.get('Vreg'))

                if vreg_data is None:
                    raise ValueError("La variable 'Vreg' no existe en el archivo.")
                # Comprobar la dimensión de la variable (suponiendo que es una matriz 2D)
                if len(vreg_data.shape) < 2:
                    raise ValueError(f"Se esperaba una matriz 2D para 'Vreg', pero se encontró con {len(vreg_data.shape)} dimensiones.")
                #print(f"Forma de 'Vreg': {mi_variable.shape}")

                # Convertir la variable a un array de NumPy
                mi_variable_np = np.array(vreg_data)
                #print(f"Contenido de 'Vreg':\n{mi_variable_np}")  # Mostrar el contenido para depuración

                # Verificar si la variable contiene datos
                if mi_variable_np.size > 0:
                    # Obtener el último valor de la última fila y columna
                    Y_reg_end = mi_variable_np[-1][-1]/self.V_base_fase
                    #Y_reg_end_pu = Y_reg_mat / self.V_base_fase  # Calcular el valor por unidad
                    #Y_reg_end = Y_reg_end_pu
                    print(f"Último valor de 'Vreg' en pu: {Y_reg_end}")
                else:
                    #print("La variable 'Vreg' está vacía.")
                    Y_reg_end = None
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo '{file_path}'.")
            Y_reg_end = None
        except ValueError as ve:
            print(f"Error de valor: {ve}")
            Y_reg_end = None
        except Exception as e:
            print(f"Error inesperado al procesar el archivo .mat: {e}")
            Y_reg_end = None

        return Y_reg_end
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def policy_evaluation (self):
        iteracion = 0
        delta_values = []
        #Inicio el rgistro del tiempo para verificar tiempo de convergencia 
        start_time =time.time()

        while True:             # Formula 3.8 pag. 62, pseudocode pg. 64 miesntras PI no es estable
            delta = 0           # la V_función es etable cuando delta sea < que eps
            Y_reg = self.initial_Y_reg
            for s in range(self.nS): # Lazo para todos los estados nS = 16-->cuadriculas estados de observación
                old_v = self.V[s]
                reward, Y_reg = self.eval_state_action(Y_reg, self.policy[s])

                #next_state = self.get_next_state_from_simulation_output(Y_reg)
                # Obtengo los resultados de la simulación y los utilizo para modificar R_t o los S_t
                #self.V[s] = (1 - alpha) * old_v + alpha * (reward + self.gamma * self.V[next_state])
                self.V[s] = reward
                delta = max(delta, np.abs(old_v - self.V[s]))
                print(f"State: {s}, Old V[s]: {old_v}, New V[s]: {self.V[s]}, Reward: {reward}, Delta: {delta}")
                #print(delta)

            delta_values.append(delta)

            print(f"Iteration: {iteracion}, Delta: {delta}")
            iteracion += 1
            if delta < self.eps:
                break
        # Tiempo que toma para que la evaluación de la política sea estable 
        end_time = time.time()
        time_diff = (end_time-start_time) / 60.0

        print(f"El tiempo total de evaluación es: {time_diff:.2f} minutes")

        # Grafico los valores de delta e  iteraciones
        plt.figure()
        plt.plot(delta_values)
        plt.xlabel('Iteraciones')
        plt.ylabel('Delta')
        plt.title('Convergecnia de Delta respecto a las Iteraciones')
        plt.show()
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def policy_improvement(self):
        policy_stable = True
        #Y_reg = self.initial_Y_reg
        print('Estoy mejorando my politica')
        for s in range(self.nS):
            old_a = self.policy[s]           #Acción anterior es tomada del array de la policy en la ubicación 's'
            action_values = []
            for a in range(self.nA):
                reward, Y_reg = self.eval_state_action(self.initial_Y_reg, a)
                action_values.append(reward)
                print(f"State {s}, Action {a}, Reward: {reward}")  # Monitor reward
            best_action = np.argmax(action_values)
            print(f"State {s} Action values: {action_values}, Best action: {best_action}")  # Monitor action values

            #self.policy[s] = best_action
            #self.policy[s] = np.argmax([self.eval_state_action(self.V[s], a) for a in range(self.nA)])
            if old_a != best_action:
                print(f"Policy changed at state {s}: Old action: {old_a}, New action: {best_action}")  # Monitor policy changes
                policy_stable = False   #Significa que la policy[s] no es estable
            self.policy[s] = best_action
        return policy_stable            #Cuando todo se cumpla
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def get_tap_position(self, Y_reg, a):
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
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def get_next_state_from_simulation_output(self, Y_reg_end):
        """_summary: Mapea la salida output en el S_t

        Args:
            Y_reg (float): El voltaje regulado de salida para la simulación

        Returns:
            int: S_(t+1) corresponde al dado por Y_reg
        """
        print(f"Y_reg_end recibido del S_t de transición: {Y_reg_end}")
        if Y_reg_end is None:
            print("Error: Y_reg_end es None, Regresa al estado inicial")
            return 0

        if Y_reg_end < 0.9:
            return 0                # S_(t+1)= State 0: El Y_reg es muy bajo, es inaceptable, -->Done
        elif 0.9 <= Y_reg_end < 0.95:
            return 1                # S_(t+1)= State 1: El Y_reg es un poco bajo
        elif 0.95 <= Y_reg_end <= 1.05:
            return 2                # S_(t+1)= State 2: El Y_reg esta en el rango deseado
        elif 1.05 < Y_reg_end <= 1.1:
            return 3                # S_(t+1)= State 3: El Y_reg es un poco alto
        else:
            return 4                # S_(t+1)= State 4: El Y_reg es muy alto, es inaceptable, --> Done
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def calculate_reward(self, Y_reg_end):
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
        print(f"Y_reg_end: {Y_reg_end}, Rango requerido: ({desired_min}, {desired_max})")

        #Defino los valores de R_t, pq no puede ser el reward = 0??
        reward_in_range = 1.0    #R_t para mi Y_reg esperado
        reward_out_range = -1.0 #R_T penalizado para un Y_reg fuera de los rangos

        if desired_min <= Y_reg_end <= desired_max:
            print(f"Y_reg_end {Y_reg_end} está dentro del rango deseado. Reward: {reward_in_range}")
            return reward_in_range
        else:
            print(f"Y_reg_end {Y_reg_end} está fuera del rango deseado. Penalty: {reward_out_range}")
            return reward_out_range
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def is_terminal_state(self, next_state):
        """_summary: Revisa si un S_t dato es terminal basado en las condiciones de predicción.
        Si es = 0 quiere decir qeu el valor Y_reg_end < 0.9 es muy bajo y es inaceptable.
        Si es = 4 quiere decir qeu el valor Y_reg_end > 1.1 es muy alto y es inaceptable.

        Args:
            next_state (int): El estado a revisar

        Returns:
            bool: True si es terminal, False si es otro.
        """
        terminal_state = [0,4]          #Considerar los S_t de la función get_next_state_from_simulation_output
        if next_state in terminal_state:
            return True
        else:
            return False
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def plot_simulation_results(self):
        plt.figure(figsize=(12,6))
        plt.subplot(2, 1, 1)
        plt.plot(self.Y_reg_val, label = 'Y_reg_val')
        plt.xlabel('Time step')
        plt.ylabel('Y_reg')
        plt.title('Regulación del Voltaje en el Tiempo')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.tap_pos_val, label = 'Tap Position', color='orange')
        plt.xlabel('Time step')
        plt.ylabel('Tap Position')
        plt.title('Posición del TAP en el tiempo')
        plt.legend()

        plt.tight_layout()

        plt.savefig('Resultados_Simulacion.png')
        plt.show()
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def run_episodes(self, num_games, Y_reg_init):
        tot_rew = 0
        action_taken = []
        for _ in range(num_games):
            Y_reg = Y_reg_init
            state = 0
            done = False
            while not done:
                action = self.policy[state]
                action_taken.append(action)
                reward, Y_reg = self.eval_state_action(Y_reg, action)           #Es una tupla con la recompenza y Y_reg-->linea 138
                next_state = self.get_next_state_from_simulation_output(Y_reg)
                done = self.is_terminal_state(next_state)
                state = next_state
                tot_rew += reward
        print('Gano %i de %i juegos!'%(tot_rew, num_games))
        return action_taken
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    eng = matlab.engine.start_matlab()   # Inicio el motor de Matlab
    eng = matlab.engine.connect_matlab() # Quita el # de la ventana de MATLAB si ya estamos compartiendo Matlab

    # Cargo el modelo y parámetros de Simulink
    eng.load_system('AC_Feeder_Control')
    eng.run('AC_Feeder_Control_Param_02.m', nargout=0)

    agent = PolicyIterationAgent(nS=5, nA=2, gamma=0.99, eps=0.8, eng=eng)  # Inicializo el agent

    # Ciclo principal
    policy_stable = False                               # Inicializo la Política PI FALSE = no es etable
    it = 0

    while not policy_stable:                            # Mientras que plicy_stable no cambie a True..
        agent.policy_evaluation ()                      # Evaluación de las policy
        policy_stable = agent.policy_improvement()
        it +=1
    print('Convergencia despues de %i  interaciones --> policy (Politicas)'%(it))

    # Llamo a mi función para graficar
    agent.plot_simulation_results()

    #Para evaluar con valores randomicos de mi Y_reg
    random_Y_reg_init_values = np.random.uniform(0.9, 1.1, 5)
    actions_by_Y_reg = []
    for Y_reg_init in random_Y_reg_init_values:
        print(f"\nEsta corriendo el episodio con Y_reg_init = {Y_reg_init}")
        #agent.V = np.zeros(agent.nS)                    #Reseteo la V_function
        #agent.policy = np.random.choice(agent.nA, size=agent.nS)    # Reset policy
        actions = agent.run_episodes(10, Y_reg_init=Y_reg_init)
        actions_by_Y_reg.append(actions)
    plt.figure(figsize=(10,6))

    for i, Y_reg_init in enumerate(random_Y_reg_init_values):
        plt.plot(range(len(actions_by_Y_reg[i])), actions_by_Y_reg[i], label = f'Y_reg_init = {Y_reg_init:.2f}')

    plt.xlabel('Episode Step')
    plt.ylabel('Actions Taken')
    plt.title('Y_reg_init vs Actions')
    plt.legend()
    plt.savefig('Resultados_Evaluacion.png')
    plt.show()

    print("\n La matriz de la Funcion del Valor Vpi: ",agent.V.reshape((4,4)))
    print("\n La matriz de la politica PI es: ", agent.policy.reshape((4, 4)))

    #Cierro Matlab
    eng.quit()
