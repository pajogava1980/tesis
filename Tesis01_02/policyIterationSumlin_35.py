# -*- coding: utf-8 -*-
#START
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matlab.engine
import numpy as np
import socket
import struct
import random
import time
import os   #Me ayuda a garantizar que la lectura del simulink sea correcta
'''
Creado 14-01-2025 10:22 p.m.

@author: pablo.gamboa

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

    El State Value Function V_(PI)(s) y de forma similar el Action Value Function Q_(PI)(s,a) lo realizan.
    ----------------------------------
    | 1.3.1 State Value Function V(s)|
    ----------------------------------
    Estima la calidad en terminos del valor esperado, cuando el AGENTE esta en un (S_t) y sigue una PI a partir
    de ese (S_t). ¿Cuándo es un valor que me saque de ese estado?
    V_(pi)(s) = E_(pi)[R_t|s_0=s]

    Valor esperado: Valor promedio ponderado de todos los posibles valores de una variable aleatoria--> S_t+1 
    al que transita el agente despues de tomar una A_t.
    -----------------------------------------------------------------
    | 1.3.1 Action Value Function Q_(PI)(S_t, A_t)--> Q-Function(s,a)|
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

    --------------------------------------------------------
    | 1.4.1 Bellman Equation para State Value Function V(s)|
    --------------------------------------------------------
    Describe el valor de un S_t como la recompensa inmediata que el agente Rx al estar en ese S_t, más
    el valor descontado de los estados futuros S_(t+1).

    V_(pi)(s) = E(pi)[R_t + factor * V(pi)(S_(t+1))] S_t = S, A_t ~ PI(S_t)

    ------------------------------------------------------------
    | 1.4.2 Bellman Equation para Action Value Function Q(s,a) |
    ------------------------------------------------------------
    Describe el valor de tomar una A_t en un S_t y luego seguir la PI, como la R_t imediata más
    el valor descontado de las futuras A_(t+1) y S_(t+1).

    ------------------------------------------------------------
    | 2.0 Agente: class PolicyIterationAgent                   |
    ------------------------------------------------------------
    Es el cerebro del sistema de control.

    Toma desiciones sobre como actua el TAP

    Aprende una política óptima con def policy_improvement(self):

    Interactúa con Simulink.

'''
class PolicyIterationAgent:
    """
    Agente de Iteración de Políticas para el control del TAP en un sistema eléctrico.
    Interactúa con Simulink para evaluar políticas y mejorar la estabilidad del voltaje
    """
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def __init__(self, nS, nA, gamma, eps, eng, prob_satis = 0.9, prob_falla = 0.7,  ip = '127.0.0.1', port = 9096):
        """
        Inicializa el agente de iteración de políticas.

        Args:
            nS (int): Número de estados posibles.
            nA (int): Número de acciones disponibles.
            gamma (float): Factor de descuento.
            eps (float): Tolerancia para la convergencia.
            eng (matlab.engine): Motor de MATLAB para interactuar con Simulink.
            prob_satis (float): Probabilidad de éxito en la transición de estado.
            prob_falla (float): Probabilidad de falla en la transición de estado.
            ip (str): Dirección IP para comunicación UDP.
            port (int): Puerto para comunicación UDP.
        """

        self.nS = nS
        self.nA = nA                                    #Por las acciones binarias... 0/1 siempre multiplo de 2

        self.gamma = gamma                              #Factor de descuento
        self.eps = eps
        self.eng = eng

        self.V = np.zeros(nS)
        self.policy = np.zeros(nS)

        self.Y_reg_val = []
        self.tap_pos_val = []
        self.V_nominal = 13.8e3
        self.initial_Y_reg = 1.0                        #Valor asumido inicialmente
        self.V_base_fase = self.V_nominal/np.sqrt(3)

        self.pausa = 1

        self.pos_max_tap = 16
        self.pos_min_tap = -16

        self.last_tap = 0
        self.tap_action = 0                             # Inicializa la variable para almacenar el TAP
        self.tap_initialized = False                    # Variable de control para saber si ya se usó el 0

        self.desired_min = 0.95
        self.desired_max = 1.05

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line1 = None
        self.line2 = None

        self.prob_satis = prob_satis
        self.prob_falla = prob_falla
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        #self.P = {}

        self.ip = ip
        self.port = port
        self.udp_socket = None                          # Se inicializa con None para la verificación en la clase marObj:
    #--------------------------------------------------------------------------------------------------------------------
    def policy_evaluation (self):
        """
        Evalúa la política actual y actualiza los valores de estado `V(s)` hasta la convergencia.
        Utiliza la ecuación de Bellman para calcular el valor esperado de cada estado.
        """
        iteracion = 0
        valores_delta = []
        start_time =time.time()

        while True:                         # Formula 3.8 pag. 62, pseudocode pg. 64 "miesntras PI no es estable"
            delta = 0                       # la V_función es etable cuando delta sea < que eps
            for s in range(self.nS):        # Para cada estado

                # Obtengo el tap correspondiente a 's'
                tap_position = self.get_tap_desde_state(s)

                        # Verifico el estado de la simulacion en SIMULINK
                sim_status = eng.get_param('AC_Feeder_Control', 'SimulationStatus')
                if sim_status == 'stopped':
                    eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
                elif sim_status == 'compiled':
                    eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
                elif sim_status == 'paused':
                    eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')", nargout=0)
                elif sim_status == 'running':
                    print("La simulación esta corriendo.")

                # Actualizo la posición del TAP aleatoria en SIMULINK
                tap_position = self.get_tap_position(self.tap_action)
                eng.workspace['tap'] = float(tap_position)
                eng.eval("set_param('AC_Feeder_Control/Tap','Value','tap')", nargout=0)

                # Cambio el clk
                clk = eng.workspace['clk']          # Leo lo que tengo en el workspace
                nuevo_clk = not clk
                eng.workspace['clk'] = nuevo_clk    # Cambio en el Workspace el valor
                eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)

                time.sleep(self.pausa) #Tiempo para esperar que se puede obtener datos desde simulink a python.

                Y_reg_end = self.matObj()
                correc_s = self.next_state(Y_reg_end)

                # Asegurar que `s` corresponde a Simulink antes de aplicar la política
                if s != correc_s:
                    print(f"Advertencia: Simulink reporta estado {correc_s}, corrigiendo `s={s}` → `s={correc_s}`")
                    s = correc_s


                old_v = self.V[s]
                self.V[s] = self.eval_state_action(s, self.policy[s])
                delta = max(delta, np.abs(old_v - self.V[s]))
                print(f"State: {s}, Old V[s]: {old_v}, New V[s]: {self.V[s]},  Delta: {delta}") #Reward: {reward},

            valores_delta.append(delta)
            print(f"Iteration: {iteracion}, Delta: {delta}")
            iteracion += 1

            if delta < self.eps:
                """
                Paso 3: Condición de convergencia, se repite hasta que cumpla.
                La convergencia V(s) significa que el agente ha aprendido la calidad de cada estado, dado el control actual del TAP. 
                El sistema ha alcanzado una represetnación estable de los efecto del contrl del TAP sobre el voltaje en Yreg
                Las transiciones y r imediatas estan correctamente integradas en el valor esperado de cada estado.
                """
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
        plt.title('Convergencia Delta vs. Iteraciones')

        filename = f'Delta_Convergencia_{timestamp}.png'
        save_dir = 'delta'
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)

        plt.close()
        print(f"Imagen guardada en: {image_path}")
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def eval_state_action(self, s, a):
        """
        Evalúa una acción en un estado dado utilizando Simulink y calcula la recompensa.

        Args:
            s (int): Estado actual.
            a (int): Acción a evaluar.

        Returns:
            float: Valor esperado de la acción en el estado dado.
        """

        # Verifico el estado de la simulacion en SIMULINK
        sim_status = eng.get_param('AC_Feeder_Control', 'SimulationStatus')
        if sim_status == 'stopped':
            eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
        elif sim_status == 'compiled':
            eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
        elif sim_status == 'paused':
            eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')", nargout=0)
        elif sim_status == 'running':
            print("La simulación esta corriendo.")

        # Actualizo la posición del TAP aleatoria en SIMULINK
        tap_position = self.get_tap_position(self.tap_action)
        eng.workspace['tap'] = float(tap_position)
        eng.eval("set_param('AC_Feeder_Control/Tap','Value','tap')", nargout=0)

        # Cambio el clk
        clk = eng.workspace['clk']          # Leo lo que tengo en el workspace
        nuevo_clk = not clk
        eng.workspace['clk'] = nuevo_clk    # Cambio en el Workspace el valor
        eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)

        time.sleep(self.pausa) #Tiempo para esperar que se puede obtener datos desde simulink a python.

        Y_reg_end = self.matObj()

        correc_s = self.next_state(Y_reg_end)
        if s != Y_reg_end:
            print(f"Advertencia: Corrigiendo el estado 's' de {s} a {correc_s}.")
            s = correc_s

        # Actualizo los valores almacenados
        self.tap_pos_val.append(tap_position)
        self.Y_reg_val.append(Y_reg_end)

        #transicion = self.P[s][a]
        mat_tran = self.mat_tran_gen(s, a, Y_reg_end)
        print(mat_tran)

        v_fun = 0   #Inicializo el valor esperado

        # Tupla (p, next_s, rew, done)
        for p, next_s, rew, done in mat_tran:
            if done:
                v_fun += p * rew
            else:
                # Bellma: El valor de un 's' V(s), es = a la 'r' imediata q se obtiene al estar en 's', más el valor esperado de los estados 's´'
                v_fun += p * (rew + self.gamma * self.V[next_s])
        return v_fun
    #--------------------------------------------------------------------------------------------------------------------
    def get_tap_desde_state(self, s):
        """
                Convierte un estado `s` en la posición del TAP correspondiente en Simulink.
            Se asume que `s=0` corresponde a `TAP=0`, los estados `s=1` a `s=16` aumentan 
            el TAP hasta `+16`, y `s=17` a `s=32` disminuyen el TAP hasta `-16`.

            Args:
                s (int): Estado actual.

            Returns:
                int: Posición del TAP asociada al estado `s`.
        """
        if s <= 16:
            return s
        else:
            return -(s-16)
    #--------------------------------------------------------------------------------------------------------------------
    def get_tap_position(self, tap_action):
        """
        Obtiene la posición del TAP basada en la acción recibida.
        La primera vez devuelve 0, luego genera valores aleatorios dentro de los límites.
        """
        if not hasattr(self, 'tap_initialized') or not self.tap_initialized:
            # Primer valor del TAP = 0
            self.tap_initialized = True
            self.tap_action = 0  # Almacenar el valor inicial en la variable de la clase
            return 0

        # Genera un valor aleatorio dentro de los límites, excluyendo el 0
        nueva_pos_tap = random.choice([i for i in range(self.pos_min_tap, self.pos_max_tap + 1) if i != 0])

        # Almacenar el nuevo valor en 'self.tap_action' para su reutilización
        self.tap_action = nueva_pos_tap

        print(f"Acción del TAP: {tap_action}, Nuevo Tap generado: {nueva_pos_tap}")

        return nueva_pos_tap  # Devuelve solo el valor
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def matObj(self):
        # Check if the UDP socket is already created
        if not hasattr(self, 'udp_socket') or self.udp_socket is None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind((self.ip, self.port))
            self.udp_socket.settimeout(5)  # Timeout for receiving data
            print(f"Socket creado y enlazado a {self.ip}:{self.port}")
        try:
            # Limpiar el buffer del socket antes de leer
            while True:
                try:
                    self.udp_socket.recvfrom(4)
                except socket.timeout:
                    print("Buffer del socket vacío, listo para leer nuevos datos.")
                    break
            #print(f"Esperando por datos en {self.ip}:{self.port}...")
            data, addr = self.udp_socket.recvfrom(4)
            #print(f"Datos sin procesar: {data} desde {addr}")

            if len(data) != 4:
                print(f"Error: El tamaño de los datos recibidos es Incorrecto: {len(data)} bytes.")
                return None

            # Decode the received 4 bytes into a float
            vreg_actual = struct.unpack('<f', data)[0]

            if not (-1e3 <= vreg_actual <= 9e3):
                raise ValueError(f"Valor fuera de rango: {vreg_actual}")

            Y_reg_end = round(vreg_actual / self.V_base_fase, 2)
            #print(f"Valor procesado: {vreg_actual}, p.u Y_reg_end: {Y_reg_end}")
            return Y_reg_end

        except socket.timeout:
            print("Timeout: No hay datos recibidos en 3 segundos.")
            return None

        except OSError as e:
            print(f"Socket error: {e}")
            return None
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def mat_tran_gen(self, s, a, Y_reg_end):
        """
                Genera la matriz de transición `self.P[s][a]` ajustando las probabilidades 
            de transición en función de la acción aplicada al TAP y su impacto en el voltaje `Y_reg_end`.

            Este método:
            1. Aplica una acción al TAP (`subir`, `bajar` o `mantener`).
            2. Lee el nuevo valor de `Y_reg_end` desde Simulink tras el ajuste del TAP.
            3. Calcula el siguiente estado (`next_s`), si es un estado terminal (`done`), y la recompensa (`rew`).
            4. Asigna probabilidades favoreciendo la acción que más acerque el voltaje al rango ideal (`0.99 ≤ Y_reg_end < 1.00`).
            5. Si `Y_reg_end_nuevo` ya está en el rango ideal, asigna **100% de probabilidad** a la acción "mantener" (`0`).

            Args:
                s (int): Estado actual antes de aplicar la acción.
                a (int): Acción aplicada al TAP (`-1`: bajar, `0`: mantener, `+1`: subir).
                Y_reg_end (float): Valor del voltaje antes de aplicar la acción.

            Returns:
                dict: `self.P[s][a]` con la estructura:
                    - `(probabilidad, next_s, rew, done)`: Probabilidad de transición, estado siguiente, recompensa y flag de estado terminal.
                    - `(1 - probabilidad, s, -1, False)`: Probabilidad alternativa en caso de error.

            Ejemplo de salida (`self.P[s][a]`):
            ```
            {
                0: [(0.8, 2, 10, False), (0.2, 0, -1, False)],
                1: [(0.1, 3, 15, False), (0.9, 1, -1, False)],
                2: [(1.0, 2, 10, False), (0.0, 2, -1, False)]  # Si ya está en 0.99-1.00
            }
            ```
        """

        correc_s = self.next_state(Y_reg_end)

        if s != correc_s:
            print(f"Advertencia: El estado proporcionado 's' ({s}) no coincide con el calculado 
                  ({correc_s}). Usando el estado calculado.")
            s = correc_s

        self.P[correc_s] = {}
        tap_actual = float(eng.workspace['tap'])

        acciones_tap = {0: -1, 1: 0, 2: +1} #bajar (-1), mantener (0), subir (+1)
        objetivo = 0.995     #Punto medio entre 0.99-1.00

        # Diccionario donde voy a almacenar la diferencia de las distancias de cada acción
        distancia_acciones = {}
        #
        #for s in range(self.nS):
            #self.P[s] = {}
        for accion in range(self.nA):
            # Calcula el nuevo TAP sin exceder los límites
            nuevo_tap = max(self.pos_min_tap, min(self.pos_max_tap, tap_actual + acciones_tap[accion]))

            eng.workspace['tap'] = float(nuevo_tap)
            eng.eval("set_param('AC_Feeder_Control/Tap','Value','tap')", nargout=0)

            # Cambio el clk
            clk = eng.workspace['clk']          # Leo lo que tengo en el workspace
            nuevo_clk = not clk
            eng.workspace['clk'] = nuevo_clk    # Cambio en el Workspace el valor
            eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)

            time.sleep(self.pausa) #Tiempo para esperar que se puede obtener datos desde simulink a python.

            Y_reg_end_nuevo = self.matObj()

            distancia_acciones[accion] = abs(Y_reg_end_nuevo - objetivo)

            next_s = self.next_state(Y_reg_end_nuevo)
            done = self.is_terminal_state(next_s)
            rew = self.calculo_reward(Y_reg_end_nuevo)

            # Almacena temporalmente la información
            self.P[correc_s][accion] = (next_s, rew, done, Y_reg_end_nuevo)

            # **Asignación de probabilidades basada en cercanía al objetivo**
        mejor_accion = min(distancia_acciones, key=distancia_acciones.get)

        if 0.99 <= Y_reg_end_nuevo < 1.00:
            probabilidad = {0:1.0, 1: 0.0, 2: 0.0}
        else:
            probabilidad = {action: (0.1 if accion != mejor_accion else 0.8) for action in range(self.nA)}

            #prob_min = self.prob_falla if rew < 0 else self.prob_satis
            #prob_alt = 1.0 - prob_min           #alt = alternativa entre 0.3 o 0.1 según el rewad
        for accion in range(self.nA):
            next_s, rew, done, Y_reg_end_nuevo = self.P[correc_s][accion]

            self.P[correc_s][accion] = [
                    (probabilidad[accion], next_s, rew, done),
                    (1 - probabilidad[accion], s, -1, False)
                ]
            print(f"Acción: {accion}, TAP: {tap_actual + acciones_tap[accion]}, Y_reg_end_nuevo: {Y_reg_end_nuevo},
                  "f"Distancia al objetivo: {distancia_acciones[accion]:.4f}, Probabilidad: {probabilidad[accion]:.2f}")

        return self.P[correc_s]
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
        print('Mejorando la politica')
        policy_stable = True
        ac_tomada = []

        for s in range(self.nS):    # Paso 1: Recorro todas los S_t
            old_a = self.policy[s]
            action_values = []

            for a in range(self.nA):    # Paso 1: Recorro todas las A_t
                action_values.append(self.eval_state_action(s,a))

            best_action = np.argmax(action_values)  # Paso 2: Selecciono A_t q max el R_t esperada--> A_(t+1)
            ac_tomada.append(best_action)
            print(f"State {s} Valor de la Acción: {action_values}, Mejor Acción: {best_action}")  # Monitor action values

            self.policy[s] = best_action            #Paso 3: Nueva PI' que sea mejor o = que PI.

            #Paso 3: Nueva PI' que sea mejor o = que PI. Si la A_(t+1)-->best_action
            if old_a != best_action:   #Si nunguna de las A_t mejora PI, entonces PI es estable =True
                print(f"Política cambio al estado {s}: Acción Anterior: {old_a}, Acción Siguiente: {best_action}")  # Monitor policy changes

                policy_stable = False   #Si la A_t cambia, significa que PI no era estable

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
        print(f"Gráficas guardadas: {archivo_acciones}")

        return policy_stable

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def plot_dynamic_results(self):
        # Configuración de la figura y los ejes
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Configuración inicial de los gráficos
        self.ax1.set_title("Regulación de Voltaje en el Tiempo")
        self.ax1.set_xlabel("Pasos de Tiempo")
        self.ax1.set_ylabel("Y_reg")
        self.ax1.grid(True)

        self.ax2.set_title("Posición del TAP en el Tiempo")
        self.ax2.set_xlabel("Pasos de Tiempo")
        self.ax2.set_ylabel("Posición TAP")
        self.ax2.grid(True)

        # Inicializa las líneas (plots vacíos)
        self.line1, = self.ax1.plot([], [], label="Y_reg_val", color='blue')
        self.line2, = self.ax2.plot([], [], label="Tap Posición", color='orange')
        self.ax1.legend()
        self.ax2.legend()

    # Función de actualización
    def update(self, frame):
        self.line1.set_data(range(len(self.Y_reg_val)), self.Y_reg_val)
        self.line2.set_data(range(len(self.tap_pos_val)), self.tap_pos_val)

        # Ajusta los límites dinámicamente
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        return self.line1, self.line2

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def next_state(self, Y_reg_end):
        """
        Si el voltaje es < 0.85 la estrategia de control debe cambiar
        """
        print(f"Y_reg_end de transición: {Y_reg_end}")

        if Y_reg_end is None:
            print("Error: Y_reg_end es 'None', Regresa al estado inicial")
            return 0
        else:
            #Y_reg_end = round(Y_reg_end, 3)
            if Y_reg_end < 0.927:                # Estado 0
                return 0
            elif 0.927 <= Y_reg_end < 0.936:      # Estado 1
                return 1
            elif 0.936 <= Y_reg_end < 0.946:      # Estado 2
                return 2
            elif 0.946 <= Y_reg_end < 0.955:      # Estado 3
                return 3
            elif 0.955 <= Y_reg_end < 0.964:      # Estado 4
                return 4
            elif 0.964 <= Y_reg_end < 0.973:      # Estado 5
                return 5
            elif 0.973 <= Y_reg_end < 0.982:      # Estado 6
                return 6
            elif 0.982 <= Y_reg_end < 0.991:      # Estado 7
                return 7
            elif 0.991 <= Y_reg_end < 1.001:      # Estado 8
                return 8
            elif 1.001 <= Y_reg_end < 1.010:      # Estado 9
                return 9
            elif 1.010 <= Y_reg_end < 1.020:      # Estado 10
                return 10
            elif 1.020 <= Y_reg_end < 1.029:      # Estado 11
                return 11
            elif 1.029 <= Y_reg_end < 1.038:      # Estado 12
                return 12
            elif 1.038 <= Y_reg_end < 1.047:      # Estado 13
                return 13
            elif 1.047 <= Y_reg_end < 1.056:      # Estado 14
                return 14
            elif 1.056 <= Y_reg_end < 1.066:      # Estado 15
                return 15
            elif 1.066 <= Y_reg_end < 1.075:      # Estado 16
                return 16
            elif 0.918 < Y_reg_end <= 1.075:      # Estado 17
                return 17
            elif 0.908 < Y_reg_end <= 0.918:      # Estado 18
                return 18
            elif 0.900 < Y_reg_end <= 0.908:      # Estado 19
                return 19
            elif 0.890 < Y_reg_end <= 0.900:      # Estado 20
                return 20
            elif 0.880 < Y_reg_end <= 0.890:      # Estado 21
                return 21
            elif 0.872 < Y_reg_end <= 0.880:      # Estado 22
                return 22
            elif 0.863 < Y_reg_end <= 0.872:      # Estado 23
                return 23
            elif 0.853 < Y_reg_end <= 0.863:      # Estado 24
                return 24
            elif 0.844 < Y_reg_end <= 0.853:      # Estado 25
                return 25
            elif 0.834 < Y_reg_end <= 0.844:      # Estado 26
                return 26
            elif 0.825 < Y_reg_end <= 0.834:      # Estado 27
                return 27
            elif 0.816 < Y_reg_end <= 0.825:      # Estado 28
                return 28
            elif 0.806 < Y_reg_end <= 0.816:      # Estado 29
                return 29
            elif 0.797 < Y_reg_end <= 0.806:      # Estado 30
                return 30
            elif 0.788 < Y_reg_end <= 0.797:      # Estado 31
                return 31
            elif 0.779 < Y_reg_end <= 0.788:      # Estado 32
                return 32
            else:
                return 33                         # Estado 31
            # Voltaje extremadamente alto, físicamente significa una mala regulación de voltaje, en SE, el TAP debe funcionar, A=0, baja el TAP
        #Solo es una mala regulación... I no son extremas

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def calculo_reward(self, Y_reg_end):
        """_Resumen_: Esta función es directa, revisa si los valores del Vreg('Y_reg') están entre los valores
        máximos y mínimos aceptables para el sistema. Se la llama en eval_state_action.

        Cual es el mecanismo para salir del rew = -1, debo buscar la forma de salir de ahi, cómo??? Obligar al TAP que se mueva de ahí para arriba
        Mientras mas se aleje del valor nominal una mayuor penalidad... OJO!!!!!

        Comparar con controladores lineales y no lineales, verificar su comportamientoe implemntar en el algoritomo...!!!!
        """

        print(f"Y_reg_end: {Y_reg_end}, Rango requerido: ({self.desired_min}, {self.desired_max})")

        if Y_reg_end is None:
            print("Error: Y_reg_end es None, asignando penalización máxima.")
            return -10
        else:
            # Asignación de recompensas para los 31 estados, positivas en el rango [0.95, 1.05]
            if Y_reg_end < 0.79:
                return -10
            elif 0.79 <= Y_reg_end < 0.8:
                return -9
            elif 0.8 <= Y_reg_end < 0.81:
                return -8
            elif 0.81 <= Y_reg_end < 0.82:
                return -7
            elif 0.82 <= Y_reg_end < 0.83:
                return -6
            elif 0.83 <= Y_reg_end < 0.84:
                return -5
            elif 0.84 <= Y_reg_end < 0.85:
                return -4
            elif 0.85 <= Y_reg_end < 0.86:
                return -3
            elif 0.86 <= Y_reg_end < 0.87:
                return -2
            elif 0.87 <= Y_reg_end < 0.88:
                return -1
            elif 0.88 <= Y_reg_end < 0.89:
                return -1
            elif 0.89 <= Y_reg_end < 0.9:
                return -1
            elif 0.9 <= Y_reg_end < 0.91:
                return -1
            elif 0.91 <= Y_reg_end < 0.92:
                return -1
            elif 0.92 <= Y_reg_end < 0.93:
                return -1
            elif 0.93 <= Y_reg_end < 0.94:
                return -1
            elif 0.94 <= Y_reg_end < 0.95:
                return 0
            elif 0.95 <= Y_reg_end < 0.96:
                return 5
            elif 0.96 <= Y_reg_end < 0.97:
                return 10
            elif 0.97 <= Y_reg_end < 0.98:
                return 10
            elif 0.98 <= Y_reg_end < 0.99:
                return 10
            elif 0.99 <= Y_reg_end < 1.0:
                return 15
            elif 1.0 <= Y_reg_end < 1.01:
                return 15
            elif 1.01 <= Y_reg_end < 1.02:
                return 10
            elif 1.02 <= Y_reg_end < 1.03:
                return 10
            elif 1.03 <= Y_reg_end < 1.04:
                return 10
            elif 1.04 <= Y_reg_end < 1.05:
                return 5
            elif 1.05 <= Y_reg_end < 1.06:
                return 0
            elif 1.06 <= Y_reg_end < 1.07:
                return -1
            elif 1.07 <= Y_reg_end < 1.08:
                return -2
            elif 1.08 <= Y_reg_end:
                return -10
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def is_terminal_state(self, next_state):
        """
        Determina si un estado es terminal.

            Un estado terminal ocurre en tres situaciones:
            1. Cuando el voltaje `Y_reg_end` es menor a `0.85` (baja tensión peligrosa).
            2. Cuando el voltaje `Y_reg_end` es mayor a `1.8` (sobretensión peligrosa).
            3. Cuando el voltaje está dentro del rango nominal `0.99 ≤ Y_reg_end < 1.01` (objetivo alcanzado).

            Args:
                next_state (float): Voltaje `Y_reg_end` en p.u.

            Returns:
                bool: `True` si el estado es terminal, `False` en caso contrario.
        """
        #terminal_state = [0, 1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31]

        return next_state < 0.85 or next_state > 1.8 or (0.99 <= next_state < 1.01)
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

    # Cargo el modelo y parámetros de Simulink
    eng.load_system('AC_Feeder_Control', nargout = 0)
    eng.run('AC_Feeder_Control_Param_02.m', nargout=0)

    agent = PolicyIterationAgent(nS = 33, nA = 3, gamma = 0.88, eps = 0.01, eng = eng, ip = '127.0.0.1', port = 9096)  # Inicializo el agent

    agent.plot_dynamic_results()

    # Animación en tiempo real
    ani = FuncAnimation(agent.fig, agent.update, interval=500, save_count=200)  # Guarda hasta 200 cuadros

    plt.tight_layout()
    plt.show()

    # Ciclo principal
    try:
        policy_stable = False                               # Inicializo la Política PI FALSE = no es etable
        it = 0
        while not policy_stable:                            # Mientras que policy_stable no cambie a True..
            agent.policy_evaluation ()                      # Evaluación de las policy
            policy_stable = agent.policy_improvement()
            it +=1
        print('Convergencia despues de %i  interaciones --> policy (Politicas)' % (it))

        # Llamo a mi función para graficar
        agent.plot_simulation_results()

        print("\n La matriz de la Funcion del Valor Vpi: ",agent.V.reshape((1, 10)))
        print("\n La matriz de la politica PI es: ", agent.policy.reshape((1, 10)))

    # Para evaluar con valores randomicos de mi Y_reg
    # timestamp = time.strftime("%Y%m%d-%H%M%S")  # Genera una estampa de tiempo unica.
    # max_steps = 10  # Setear el máximo numero de pasos
    # num_episod = 10
    # random_Y_reg_init_values = np.random.uniform(0.9, 1.1, 5)
    # actions_by_Y_reg = []
    # for Y_reg_init in random_Y_reg_init_values:
    #     print(f"\nEsta corriendo el episodio con Y_reg_init = {Y_reg_init}")
    #     actions = agent.run_episodes(num_episod, Y_reg_init=Y_reg_init, max_steps=max_steps)
    #     actions_by_Y_reg.append(actions)

    # plt.figure(figsize=(10,6))
    # for i, Y_reg_init in enumerate(random_Y_reg_init_values):
    #     plt.plot(range(len(actions_by_Y_reg[i])), actions_by_Y_reg[i], label = f'Y_reg_init = {Y_reg_init:.2f}')

    # plt.xlabel('Episode Step')
    # plt.ylabel('Actions Taken')
    # plt.title('Y_reg_init vs Actions')
    # plt.legend()

    # filename = f'Y_reg_vs_Actions__{timestamp}.png'
    # save_dir = 'final'
    # image_path = os.path.join(save_dir, filename)

    # plt.close()


    #Cierro Matlab
    finally:

        #estado_sim = eng.get_param('AC_Feeder_Control', 'SimulationCommand')
        #if estado_sim in ['running', 'paused', 'compiled']:
        eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'stop')", nargout=0)

        #eng.set_param('AC_Feeder_Control', 'SimulationMode', 'normal', nargout = 0)
        #eng.set_param('AC_Feeder_Control', 'FastRestart', 'off', nargout=0)
        eng.quit()
