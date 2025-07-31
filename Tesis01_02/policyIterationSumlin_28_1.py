# -*- coding: utf-8 -*-
#START
import matplotlib.pyplot as plt
import matlab.engine
import numpy as np
import h5py
import time
import os   #Me ayuda a garantizar que la lectura del simulink sea correcta

#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

def veri_compartir_matlab_engine():
    try:
        eng = matlab.engine.start_matlab()
        eng = matlab.engine.connect_matlab()
        es_compartida = eng.eval("matlab.engine.isEngineShared", nargout=1)

        if not es_compartida:
            print("El motor de MATLAB no está compartido, compartiéndolo ahora...")
            eng.eval("matlab.engine.shareEngine", nargout=0)
            print("Motor de MATLAB compartido exitosamente.")
        else:
            print("El motor de MATLAB ya está compartido.")
    except Exception as e:
        print(f"Error al verificar o compartir el motor de MATLAB: {e}")

    return eng

#eng = veri_compartir_matlab_engine()


#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

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

----------------------------------------------
    | 1.1. POLICY | OR DECISION MAKER | OR AGENT |
    ----------------------------------------------
    Es una estrategia o regla que define el comportamiento del AGENTE. Una PI especifica la A_t
    que el agente debe tomar en cada S_t del ENV para Maximizar alguna medida de rendimiento, tipicamente
    el REWARD R_t

    -----------------------------------------------------------------------------------------------
    | 1.2. RETURN G(tau)--> TRAJECTORY OR ROLLOUT proporciona un buen valor interno NO en calidad |
    -----------------------------------------------------------------------------------------------
    Es la recompensa total acumulada que un AGENTE Rx a lo largo del tiempo (t) a partir de un S_t o A_t.
    Este es el valor que el AGENTE trata de maximizar mientras interactua con el ENV. 
    -------------------
    | 1.3. V-Functions|
    -------------------
    Las Value Functions son herramientas fundamentales para evaluar cuán "bueno" es un estado (S_t)
    o una acción (A_t) dentro de un EVN. Estas funciones (State Value Function (Vs) y Action Value Function Q(s,a),
    miden la recompensa futura a partir de un estado (S_t) o de una acción (A_t).

    Este indicador de calidad "Quality", es importante, pq la PI lo puede usarlo para elegir
    la mejor A_(t+1). La PI solo elige la A_t que va a dar como resultado la > Quality en el S_(t+1).

    El State Value FUnction V_(pi)(s) y de forma similar el Action Value Function Q_(pi)(s,a) lo realizan.
    ----------------------------------
    | 1.3.1 State Value Function V(s) |
    ----------------------------------
    Estima la calidad en terminos del valor esperado, cuando el AGENTE esta en un (S_t) y sigue una PI a partir
    de ese (S_t).
    V_(pi)(s) = E_(pi)[R_t|s_0=s]
    -----------------------------------------------------------------
    | 1.3.1 Action Value Function Q_(pi)(S_t, A_t)--> Q-Function(s,a)|
    -----------------------------------------------------------------
    Q(s,a), evalua la calidad de tomar una A_t en un S_t, y luego seguir PI
    desde ahí (La A_t que debe tomar en ese S_t).
    Representa la R_(t+1) esperada si el AGENTE toma la A_t en el S_t y sigue la PI.
     ---------------------------------------------------------
    | 1.4. BELLMAN EQUATION  F_Function A-Function 3.6 y 3.7 |
    ---------------------------------------------------------
    Es fundamental en el campo de la toma de decisiones secuenciales, como en problemas de control óptimo
    y de programación dinámica. Establece una relación recursiva (se llama así misma) entre el valor de un
    S_t y el valor de los S_(t+1), permite descomponet el problema de optimización a largo plazo en problemas
    más pequeños y manejables.

    La ecuación de Bellman dice que el valor de un S_t es = a la R_t inmediata que se obtiene en ese S_t, más
    el valor descontado del próximo estado S_(t+1).

    1.4.1 Bellman Equation para State Value Function V(s)
    Describe el valor de un S_t como la recompensa inmediata que el agente Rx al estar en ese S_t, mas
    el valor descontado de los estados futuros S_(t+1).

    V_(pi)(s) = E(pi)[R_t + factor * V(pi)(S_(t+1))] S_t = S, A_t ~ PI(S_t)

    1.4.2 Bellman Equation para Action Value Function Q(s,a)
    Describe el valor de tomar una A_t en un S_t y luego seguir la PI, como la R_t imediata más
    el valor descontado de las futuras A_(t+1) y S_(t+1).
'''

class PolicyIterationAgent:
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def __init__(self, nS, nA, gamma, eps, eng):
        self.nS = nS #El porque debe ser mas o menos este valor ????
        self.nA = nA #Por las acciones binarias... 0/1 siempre multiplo de 2
        self.gamma = gamma  #Factor de descuento
        self.eps = eps
        self.eng = eng
        #self.policy = np.random.choice(nA, size=nS)
        self.policy = np.zeros(nS)
        self.V = np.zeros(nS)
        #self.V = np.random.rand(nS) #Inicializo V(s) con valores aleatorios, en lugar de ceros. 
        self.Y_reg_val = []
        self.tap_pos_val = []
        self.initial_Y_reg = 1.0                   #Valor asumido inicialmente
        self.V_nominal = 13.8e3
        self.V_base_fase = self.V_nominal/np.sqrt(3)
        self.pausa = eng.workspace['T']/2
        self.pos_max_tap = 16
        self.pos_min_tap = -16
        self.desired_min = 0.95
        self.desired_max = 1.05
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1

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
        """
        # Verifico si el modelo 'AC_Feeder_Control' está cargado.
        # mod_cargados = self.eng.eval("Simulink.allBlockDiagrams('model')", nargout=1)
        # if 'AC_Feeder_Control' not in mod_cargados:
        #     print("El modelo 'AC_Feeder_Control' no esta cargado, cargando el modelo...")
        #     self.eng.load_system('AC_Feeder_Control')
        # else:
        #     print("El modelo 'AC_Feeder_Control' ya esta cargado")

        #Verifico el estado de la simulacion
        sim_status = eng.get_param('AC_Feeder_Control', 'SimulationStatus')
        #Empiezo la simualción si aun no esta corriendo
        if sim_status == 'stopped':
            #print("La simualción esta parada, empezando la simulación...")
            eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
            #time.sleep(self.pausa)
        elif sim_status == 'paused':
            #print("La simulación esta pausada, continuando con la simulación...")
            eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')", nargout=0)
            #time.sleep(self.pausa)
        elif sim_status == 'running':
            print("La simulación esta corriendo.")

        #Actualizo la posición del TAP
        tap_position = self.get_tap_position(Y_reg, a)
        eng.workspace['tap'] = float(tap_position)
        eng.eval("set_param('AC_Feeder_Control/Tap','Value','tap')", nargout=0)

        clk = eng.workspace['clk']      #Leo lo que tengo en el workspace
        if clk is False:
            eng.workspace['clk'] = not clk  #Cambio en el Workspace el valor
            eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)
        else:
            eng.workspace['clk'] = not clk  #Cambio en el Workspace el valor
            eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)
            clk = eng.workspace['clk']      #Leo lo que tengo en el workspace
            eng.workspace['clk'] = not clk  #Cambio en el Workspace el valor
            eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)
       
        eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'pause')", nargout=0)

        Y_reg_end = self.matObj()
        next_state = self.next_state(Y_reg_end)
        reward = self.calculo_reward(Y_reg_end)
        done = self.is_terminal_state(next_state)

        self.tap_pos_val.append(tap_position)
        self.Y_reg_val.append(Y_reg_end)

        if done:
            return reward, Y_reg_end
        else:
            # Actualización de la ecuación de Bellman B = R(s,a) + gamma * V(s_(t+1))
            reward  = reward + self.gamma*self.V[next_state]
            return reward, Y_reg_end

    #--------------------------------------------------------------------------------------------------------------------

    def action_basada_Y_reg(self, Y_reg):
        if Y_reg is None:
            return np.random.choice([0,1])
        elif Y_reg < self.desired_min:
            return 1    # Subir el TAP
        elif Y_reg > self.desired_max:
            return 0    # Bajar el TAP
        else:
            return np.random.choice([0, 1])

    #--------------------------------------------------------------------------------------------------------------------
    def policy_evaluation (self):
        """
        Es un proceso clave para encontrar la PI óptima. La combinación de policy_evaluation con policy_improvement 
        ayuda a encontrar una mejor PI del AGENT en un EVN determinado.

        Calcúla de V(s) o Q(s,a) dado que el AGENT sigue una PI. Responde a la pregunta:
            ¿Qué tan buena es una política dada?
        Se calcula V(s) o Q(s,a) bajo una PI.
            Se estima el R_t TOTAL, q es el valor esperado desde cada S_t asumiendo que el AGENT siempre 
            sigue la PI.
            Paso 1: V(s) aleatorio
            Paso 2: Actualizo de forma iterativa la Ec. de BELLMAN
            Paso 3: Repetir hasta que converja a un V(s) estable. 
        """
        max_iteracion = 20
        iteracion = 0
        valores_delta = []
        theta = 1e-05   #Criterio de convergencia

        #Inicio el rgistro del tiempo para verificar tiempo de convergencia 
        start_time =time.time()

        while iteracion <= max_iteracion:
            delta = 0   # la V_función es etable cuando delta sea < que eps
            for s in range(self.nS): # Para cada estado S_t
                old_v = self.V[s]

                Y_reg = self.matObj()
                a = self.action_basada_Y_reg(Y_reg)

                reward, Y_reg_end = self.eval_state_action(self.V[s], s, self.policy[s])
                sig_state = self.next_state(Y_reg_end)
                v_funtion = reward + self.gamma * self.V[sig_state]   # Paso 2: Actualizo V(s) con Bellman

                self.V[s] = v_funtion
                delta = max(delta, np.abs(old_v - self.V[s]))
                print(f"State: {s}, Old V[s]: {old_v}, New V[s]: {self.V[s]}, Reward: {reward}, Delta: {delta}")

            valores_delta.append(delta)
            print(f"Iteration: {iteracion}, Delta: {delta}")
            iteracion += 1

            if delta < theta:   # Paso 3: Condición de convergencia, se repite hasta que cumpla.
                break

        # Tiempo que toma para que la evaluación de la política sea estable 
        end_time = time.time()
        print(f"El tiempo total de evaluación es: {(end_time - start_time) / 60.0:.2f} minutes")

        # Grafico los valores de delta e  iteraciones
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.figure()
        plt.plot(valores_delta)
        plt.xlabel('Iteraciones')
        plt.ylabel('Delta')
        plt.title('Convergecnia de Delta vs. Iteraciones')

        filename = f'Delta_Convergencia_{timestamp}.png'
        save_dir = 'delta'
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)

        plt.close()
        print(f"Imagen guardada en: {image_path}")

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def policy_improvement(self):
        """Evaluado policy_evaluation, el siguiente paso es mejorar la PI a PI', respondiendo la siguiente pregunta:
            ¿Cómo puedo mejorar mi PI para obtener > R_t?.

            El objetivo es encontrar una mejor PI eligiendo una A_t que lleve a un > R_t usando la EC. BELLMAN
            Paso 1: REviso todas las A_t posibles en cada S_t
            Paso 2: Selecciono la A-t que maximice el R_t esperador.
            Paso 3: Si, la nueva PI' se actualiza de manera que sea mejor o = que la PI.
        """
        policy_stable = True
        recompen = []
        ac_tomada = []
        print('Mejorando la politica')
        for s in range(self.nS):    # Paso 1: Recorro todas los S_t
            old_a = self.policy[s]
            action_values = np.zeros(self.nA)

            for a in range(self.nA):    # Paso 1: Recorro todas las A_t
                prob = 1
                reward, Y_reg_end = self.eval_state_action(self.V[s], a)
                sig_state = self.next_state(Y_reg_end)
                action_values[a] = prob * (reward + self.gamma * self.V[sig_state])
                ac_tomada.append(a)
                recompen.append(reward)
                print(f"Estado: {s}, Acción: {a}, Reward: {reward}")  # Monitor reward

            best_action = np.argmax(action_values)  # Paso 2: Selecciono A_t q max el R_t esperada--> A_(t+1)
            print(f"State {s} Valor de la Acción: {action_values}, Mejor Acción: {best_action}")  # Monitor action values

            self.policy[s] = best_action    #Paso 3: Nueva PI' que sea mejor o = que PI.
            #Paso 3: Nueva PI' que sea mejor o = que PI. Si la A_(t+1)-->best_action
            if old_a != best_action:   #Si nunguna de las A_t mejora PI, entonces PI es estable =True
                print(f"Política cambio al estado {s}: Acción Anterior: {old_a}, Acción Siguiente: {best_action}")  # Monitor policy changes
                policy_stable = False   #Si la A_t cambia, significa que PI no era estble

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Gráfico de las acciones tomadas
        plt.figure(figsize=(10,6))
        plt.plot(range(len(ac_tomada)), ac_tomada, label='Acciones tomadas')
        plt.xlabel('Iteraciones')
        plt.ylabel('Acción')
        plt.title('Evolución de las Acciones durante policy_improvement')
        plt.legend()

        filename = f'Acciones_policy_improvement_{timestamp}.png'
        save_dir = 'mejorPolitica'
        archivo_acciones = os.path.join(save_dir, filename)
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)

        plt.close() # Si no cierro, la simulación se para

        # Gráfico de las recompensas obtenidas
        plt.figure(figsize=(10,6))
        plt.plot(range(len(recompen)), recompen, label='Recompensas')
        plt.xlabel('Iteraciones')
        plt.ylabel('Recompensa')
        plt.title('Evolución de las Recompensas durante policy_improvement')
        plt.legend()

        filename = f'Recompensas_policy_improvement_{timestamp}.png'
        save_dir = 'mejorPolitica'
        archivo_recompensas = os.path.join(save_dir, filename)
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)

        plt.close() # Si no cierro, la simulación se para

        print(f"Gráficas guardadas: {archivo_acciones} y {archivo_recompensas}")

        return policy_stable            #Cuando todo se cumpla

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
            max_wait_time = 5
            time_waited = 0
            while True:
                try:
                    with h5py.File(file_path, 'r') as mat_file:
                        #print("Claves en el archivo:", list(mat_file.keys()))
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
                mi_variable_np = np.array(vreg_data)
                # Verificar si la variable contiene datos
                if mi_variable_np.size > 0:
                    # Obtener el último valor de la última fila y columna
                    Y_reg_end = mi_variable_np[-1][-1]/self.V_base_fase
                    print(f"Último valor de 'Y_reg_end' en pu: {Y_reg_end}")
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
    def get_tap_position(self, Y_reg, a):
        """Recorro todos los posibles posiciones del TAP

        Args:
            Y_reg (_type_): _description_
            a (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Obtener la posición actual del TAP desde el workspace
        pos_actual_tap = eng.workspace['tap']

        if Y_reg is None:
            print("Y_reg = None, regresar la posición actual del tap.")
            return pos_actual_tap

        # Factor de amortiguación: si estamos dentro del rango deseado
        if self.desired_min <= Y_reg <= self.desired_max:
            # Si estamos cerca del valor ideal (dentro del rango 0.98 - 1.02)
            if 0.98 <= Y_reg <= 1.02:
                f_amorti = 0.5  # Reducir el cambio a la mitad
            else:
                f_amorti = 0.75  # Reducir el cambio en un 25%
        else:
            f_amorti = 1.0  # Sin amortiguación si está fuera del rango

        if a == None:
            if Y_reg < self.desired_min:
                a = 1   # Subir TAP
            elif Y_reg > self.desired_max:
                a = 0   # Bajar TAP
            else:
                a = np.random.choice([0, 1])

        if a == 0:
            cambio_tap = np.random.choice([-2, -1, 0]) * f_amorti
        elif a == 1:
            cambio_tap = np.random.choice([0, 1, 2]) * f_amorti
        else:
            cambio_tap = 0

        nueva_pos_tap = pos_actual_tap + cambio_tap
        nueva_pos_tap = max(self.pos_min_tap, min(self.pos_max_tap, nueva_pos_tap))

        return nueva_pos_tap
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def next_state(self, Y_reg_end):
        """_summary: Mapea la salida output en el S_t

        Args:
            Y_reg (float): El voltaje regulado de salida para la simulación

        Returns:
            int: S_(t+1) corresponde al dado por Y_reg
        """
        print(f"Y_reg_end de transición, permite calcular en S_(t+1): {Y_reg_end}")

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
    def calculo_reward(self, Y_reg_end):
        """_summary_: Esta función es directa, revisa si los valores del Vreg('Y_reg') están entre los valores
        máximos y mínimos aceptables para el sistema. Se la llama en eval_state_action.
        """

        print(f"Y_reg_end: {Y_reg_end}, Rango requerido: ({self.desired_min}, {self.desired_max})")

        if Y_reg_end is None or Y_reg_end < 0.95:
            desviacion = self.desired_min - Y_reg_end
            penalty = -5 * desviacion
            reward = max(penalty, -0.5)
            print(f"Y_reg_end {Y_reg_end} está fuera del rango deseado. Penalty: {reward}")

        elif Y_reg_end is None or Y_reg_end > 1.05: #Penalización por encima del límite máximo.
            desviacion = Y_reg_end - self.desired_max
            penalty = -5 * desviacion
            reward = max(penalty, -0.5)
            print(f"Y_reg_end {Y_reg_end} está fuera del rango deseado. Penalty: {reward}")

        else:
            #Recompenza escalonada basada en que tan cerca esta el voltaje del valor nominal
            if 0.98 <= Y_reg_end <= 1.02:
                reward = 2
                print(f"Y_reg_end {Y_reg_end} está dentro del rango deseado. Reward: {reward}")
            elif 0.95 <= Y_reg_end <= 0.98 or 1.02 < Y_reg_end <= 1.05:
                reward = 1  #Rewar < si esta en el rango aceptable.
                print(f"Y_reg_end {Y_reg_end} está dentro del rango deseado. Reward: {reward}")
        return reward
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
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        plt.figure(figsize=(12,6))
        plt.subplot(2, 1, 1)
        plt.plot(self.Y_reg_val, label = 'Y_reg_val')
        plt.xlabel('Pasos Tiempo')
        plt.ylabel('Y_reg')
        plt.title('Regulación del Voltaje en el Tiempo')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.tap_pos_val, label = 'Posición TAP', color='orange')
        plt.xlabel('Pasos Tiempo')
        plt.ylabel('Posición TAP')
        plt.title('Posición del TAP en el tiempo')
        plt.legend()

        plt.tight_layout()

        filename = f'Resultados_Simulacion_{timestamp}.png'
        save_dir = 'resultados'
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)
        plt.close()
        print(f"Imagen guardada en: {image_path}")
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def run_episodes(self, num_games, Y_reg_init, max_steps):
        tot_rew = 0
        action_taken = []
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a unique timestamp
        for game_num in range(num_games):  # Use game_num as part of the filename
            Y_reg = Y_reg_init
            state = 0
            done = False
            step = 0  # Initialize a step counter

            # Ajusto el epsilon al inicio de cada episodio
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            while not done or step < max_steps:
                action = self.choose_action(state, self.epsilon)
                #action = self.policy[state]
                action_taken.append(action)
                reward, Y_reg_end = self.eval_state_action(Y_reg, action)  # Execute action, get reward and Y_reg
                next_state = self.get_next_state_from_simulation_output(Y_reg_end)  # Determine next state
                done = self.is_terminal_state(next_state)  # Check if the next state is terminal
                state = next_state  # Update the current state
                tot_rew += reward  # Accumulate total reward
                step += 1  # Increment the step counter
            # If the episode ends due to reaching max_steps
            if step >= max_steps:
                print(f"Episodeos han terminado despues de alcanzar {max_steps} steps.")
            # Generate a unique filename using the game number and timestamp
            filename = f"Resultado_de_la simulacion_juegos_{game_num}_{timestamp}.png"
            # Plotting the results for this episode (example plot)
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(action_taken)), action_taken, label=f'Actions for game {game_num}')
            plt.xlabel('Episode Step')
            plt.ylabel('Acción tomada')
            plt.title(f'Actions over Time - Game {game_num}')
            plt.legend()

            filename = f'Actions over Time - Game_{timestamp}.png'
            save_dir = 'juegos'
            image_path = os.path.join(save_dir, filename)
            plt.savefig(image_path)
            plt.close()  # Close the figure to prevent it from displaying

        print(f'Completado {num_games} episodeos, total reward: {tot_rew}')
        return action_taken


#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    eng = matlab.engine.start_matlab()   # Inicio el motor de Matlab
    eng = matlab.engine.connect_matlab() # Quita el # de la ventana de MATLAB si ya estamos compartiendo Matlab
    #eng = veri_compartir_matlab_engine()



    # Cargo el modelo y parámetros de Simulink
    eng.load_system('AC_Feeder_Control')
    eng.run('AC_Feeder_Control_Param_02.m', nargout=0)

    agent = PolicyIterationAgent(nS=5, nA=2, gamma=0.8, eps=0.001, eng=eng)  # Inicializo el agent

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

    print("\n La matriz de la Funcion del Valor Vpi: ",agent.V.reshape((1, 5)))
    print("\n La matriz de la politica PI es: ", agent.policy.reshape((1, 5)))

    # Para evaluar con valores randomicos de mi Y_reg
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a unique timestamp
    max_steps = 10  # Setear el máximo numero de pasos
    num_episod = 10
    random_Y_reg_init_values = np.random.uniform(0.9, 1.1, 5)
    actions_by_Y_reg = []
    for Y_reg_init in random_Y_reg_init_values:
        print(f"\nEsta corriendo el episodio con Y_reg_init = {Y_reg_init}")
        actions = agent.run_episodes(num_episod, Y_reg_init=Y_reg_init, max_steps=max_steps)
        actions_by_Y_reg.append(actions)

    plt.figure(figsize=(10,6))
    for i, Y_reg_init in enumerate(random_Y_reg_init_values):
        plt.plot(range(len(actions_by_Y_reg[i])), actions_by_Y_reg[i], label = f'Y_reg_init = {Y_reg_init:.2f}')

    plt.xlabel('Episode Step')
    plt.ylabel('Actions Taken')
    plt.title('Y_reg_init vs Actions')
    plt.legend()

    filename = f'Y_reg_vs_Actions__{timestamp}.png'
    save_dir = 'final'
    image_path = os.path.join(save_dir, filename)

    plt.close()


    #Cierro Matlab
    eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'stop')", nargout=0)
    eng.quit()
