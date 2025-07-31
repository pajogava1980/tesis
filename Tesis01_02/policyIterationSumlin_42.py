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
import os
# Me ayuda a garantizar que la lectura del simulink sea correcta
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

        self.pausa = 3

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

        self.ip = ip
        self.port = port
        self.udp_socket = None                          # Se inicializa con None para la verificación en la clase marObj:

        # Sincronizar estado inicial con Simulink
        self.estado_actual = self.sincronizar_estado_inicial()
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def obtener_estado_rango(self, s):
        Y_reg_ranges = [
            (16, (0.927, 0.936)), (17, (0.936, 0.945)), (18, (0.945, 0.955)),
            (19, (0.955, 0.964)), (20, (0.964, 0.973)), (21, (0.973, 0.982)),
            (22, (0.982, 0.992)), (23, (0.992, 1.001)), (24, (1.001, 1.010)),
            (25, (1.010, 1.019)), (26, (1.019, 1.029)), (27, (1.029, 1.039)),
            (28, (1.039, 1.048)), (29, (1.048, 1.057)), (30, (1.057, 1.066)),
            (31, (1.066, 1.075)), (15, (0.917, 0.927)), (14, (0.908, 0.917)),
            (13, (0.899, 0.908)), (12, (0.890, 0.899)), (11, (0.880, 0.890)),
            (10, (0.871, 0.880)), (9, (0.862, 0.871)),  (8, (0.852, 0.862)),
            (7,  (0.844, 0.852)), (6, (0.834, 0.844)),  (5, (0.825, 0.834)),
            (4,  (0.815, 0.825)), (3, (0.806, 0.815)),  (2, (0.797, 0.806)),
            (1,  (0.788, 0.797)), (0, (0.779, 0.788))
        ]
        return Y_reg_ranges[s]
    #--------------------------------------------------------------------------------------------------------------------
    def policy_evaluation (self):
        """
        Evalúa la política actual y actualiza los valores de estado `V(s)` hasta la convergencia.
        Utiliza la ecuación de Bellman para calcular el valor esperado de cada estado.
        """
        iteracion = 0
        valores_delta = []
        start_time =time.time()
        estados_a_actualizar = {}

        for s in range(self.nS):
            correc_s, Y_reg_end_range = self.obtener_estado_rango(s)
            estados_a_actualizar[s] = (correc_s, Y_reg_end_range)

        while True:
            delta = 0
            # Paso 2: Actualizar los valores de estado de manera mas eficiente
            for s, (correc_s, Y_reg_end_range) in estados_a_actualizar.items():
                old_v = self.V[correc_s]
                self.V[correc_s] = self.eval_state_action(s, self.policy[correc_s], estados_a_actualizar)
                delta = max(delta, np.abs(old_v - self.V[correc_s]))
                print(f"State: {correc_s}, Old V[s]: {old_v}, New V[s]: {self.V[correc_s]},  Delta: {delta}") #Reward: {reward},

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
    def eval_state_action(self, s, a, estados_a_actualizar):
        """
        Evalúa una acción en un estado dado utilizando Simulink y calcula la recompensa.
        Args:
            s (int): Estado actual.
            a (int): Acción a evaluar.
        Returns:
            float: Valor esperado de la acción en el estado dado.
        """
        correc_s, Y_reg_end_range = estados_a_actualizar[s]
        Y_reg_end =round((Y_reg_end_range[0] + Y_reg_end_range[1])/2,3)

        # Actualizo los valores almacenados
        self.tap_pos_val.append(self.get_tap_desde_state(correc_s))
        self.Y_reg_val.append(Y_reg_end)

        mat_tran = self.mat_tran_gen(s, a, Y_reg_end, estados_a_actualizar)
        print(mat_tran)

        v_fun = sum(p * (rew + (0 if done else self.gamma * self.V[next_s])) for p, next_s, rew, done in mat_tran)

        return v_fun
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
                self.policy[s] = np.argmax([self.eval_state_action(self.V, s, a) for a in range(self.nA)])

                ac_tomada.append(self.policy[s])

                #Paso 3: Nueva PI' que sea mejor o = que PI. Si la A_(t+1)-->best_action
                if old_a != self.policy[s]:   #Si nunguna de las A_t mejora PI, entonces PI es estable =True
                    print(f"Política cambio al estado {s}: Acción Anterior: {old_a}, Acción Siguiente: {self.policy[s]}")  # Monitor policy changes
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
    def int_simple_simulink(self):
        # Verifico el estado de la simulación en SIMULINK
        sim_status = self.eng.get_param('AC_Feeder_Control', 'SimulationStatus')
        if sim_status == 'stopped':
            self.eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
            time.sleep(0.5)
        elif sim_status == 'compiled':
            self.eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
            time.sleep(0.5)
        elif sim_status == 'paused':
            self.eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')", nargout=0)
            time.sleep(0.5)
        elif sim_status == 'running':
            print("La simulación está corriendo.")
        else:
            print(f"Estado desconocido de Simulink: {sim_status}. Intentando iniciar la simulación...")
            self.eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
            time.sleep(0.5)

#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
    def sincronizar_estado_inicial(self):
        #Leo el tap de Simulink
        tap_inicial = float(eng.workspace['tap'])
        s_inicial = self.get_state_desde_tap(tap_inicial)

        # Sincronizar el estado inicial de Python con Simulink
        print(f"[SYNC] Estado Inicial: TAP Simulink={tap_inicial}, Estado Python={s_inicial}")

        return s_inicial
#--------------------------------------------------------------------------------------------------------------------
    def get_state_desde_tap(self, tap):
        """
            Convierte una posición de TAP en su estado `s` equivalente en Python.
            Args:
                tap (int): Posición del TAP en Simulink.
            Returns:
                int: Estado `s` correspondiente.
        """
        if tap >= 0:
            return tap
        else:
            return 16 - abs(tap)
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
            return -(s - 16)
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def matObj(self):
        # Verificar si el socket ya está creado
        if not hasattr(self, 'udp_socket') or self.udp_socket is None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind((self.ip, self.port))
            self.udp_socket.settimeout(3)  # Timeout para recibir datos
            print(f"Socket creado y enlazado a {self.ip}:{self.port}")

        try:
            # Limpiar el buffer del socket
            self.limpiar_buffer()
            # Esperar datos nuevos
            start_time = time.time()
            data, addr = self.udp_socket.recvfrom(4)
            end_time = time.time()
            print(f"Y_reg_end recibido: {data}, Intervalo entre lecturas: {end_time - start_time:.6f} segundos")

            # Decodificar los datos
            if len(data) != 4:
                raise ValueError(f"Tamaño de datos recibido inválido: {len(data)} bytes.")

            vreg_actual = round(struct.unpack('<f', data)[0], 3)

            # Validar rango de datos
            if not (-1e3 <= vreg_actual <= 9e3):
                raise ValueError(f"Valor fuera de rango: {vreg_actual}")

            # Convertir a p.u. y redondear
            Y_reg_end = round(vreg_actual / self.V_base_fase, 3)
            print(f"Valor procesado: {vreg_actual}, p.u Y_reg_end: {Y_reg_end}")
            return Y_reg_end

        except socket.timeout:
            print("Timeout: No hay datos recibidos en 3 segundos.")
            return 0.99  # Valor por defecto en caso de error
            #continue

        except OSError as e:
            print(f"Error de socket: {e}")
            return 0.99  # Valor por defecto en caso de error
            #continue
    #--------------------------------------------------------------------------------------------------------------------
    def limpiar_buffer(self):
        """
        Limpia el buffer del socket UDP para evitar datos residuales antes de leer nuevos datos.
        """
        # Limpiar el buffer del socket
        intento =0
        while True:
            try:
                self.udp_socket.recvfrom(4)
                print(f"Intento {intento + 1}: Dato eliminado del buffer.")
                intento += 1
            except socket.timeout:
                print(f"Buffer vacío después de {intento + 1} intentos.")
                break
            #intento += 1
    #--------------------------------------------------------------------------------------------------------------------
    def mat_tran_gen(self, s, a, Y_reg_end, estados_a_actualizar ):
        """
    Genera la matriz de transición `self.P[s][a]` ajustando las probabilidades 
    de transición en función de la acción aplicada al TAP y su impacto en el voltaje `Y_reg_end`.
    Args:
        s (int): Estado actual antes de aplicar la acción.
        a (int): Acción aplicada al TAP (`-1`: bajar, `0`: mantener, `+1`: subir).
        Y_reg_end (float): Voltaje antes de aplicar la acción.
    Returns:
        list: Lista de tuplas con la estructura [(probabilidad, next_s, rew, done)].

            Ejemplo de salida (`self.P[s][a]`):
            ```
            {
                0: [(0.8, 2, 10, False), (0.2, 0, -1, False)],
                1: [(0.1, 3, 15, False), (0.9, 1, -1, False)],
                2: [(1.0, 2, 10, False), (0.0, 2, -1, False)]  # Si ya está en 0.99-1.00
            }
            ```
        """
        #Paso 1 Incialización 
        correc_s, Y_reg_end = estados_a_actualizar [s]
        acciones_tap = {0: [-1, 0], 1: [1, 0], 2: [-1, 1]}            #0:bajar (-1),  1:subir (+1), 2:mantener (0),
        tap_actual = self.get_tap_desde_state(s)
        acciones_permitidas = acciones_tap[a]

        transiciones = []

        self.int_simple_simulink()

        for accion in acciones_permitidas:
            nuevo_tap = max(self.pos_min_tap, min(self.pos_max_tap, tap_actual + accion))
            self.eng.workspace['tap'] = float(nuevo_tap)
            self.eng.eval("set_param('AC_Feeder_Control/Tap','Value','tap')", nargout=0)
            clk = eng.workspace['clk']
            nuevo_clk = not clk
            self.eng.workspace['clk'] = nuevo_clk
            self.eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)
            time.sleep(self.pausa)

            Y_reg_end_nuevo = self.matObj()
            next_s = self.next_state(Y_reg_end_nuevo)
            done = self.is_terminal_state(Y_reg_end_nuevo)
            rew = self.calculo_reward(Y_reg_end_nuevo)

            transiciones.append((0.5, next_s, rew, done))

        return transiciones
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def next_state(self, Y_reg_end):
        """
        Determina el próximo estado basado en el valor de Y_reg_end.
        Si Y_reg_end no está dentro del rango esperado, regresa al estado inicial.
        """
        print(f"Y_reg_end de transición: {Y_reg_end}")
        if Y_reg_end is None:
            print("Error: Y_reg_end es 'None', Regresa al estado inicial")
            return 0
        else:
            # Rango y asignación de estados
            if 0.779 <= Y_reg_end < 0.788:        # Estado 0
                return 0
            elif 0.788 <= Y_reg_end < 0.797:      # Estado 1
                return 1
            elif 0.797 <= Y_reg_end < 0.806:      # Estado 2
                return 2
            elif 0.806 <= Y_reg_end < 0.815:      # Estado 3
                return 3
            elif 0.815 <= Y_reg_end < 0.825:      # Estado 4
                return 4
            elif 0.825 <= Y_reg_end < 0.834:      # Estado 5
                return 5
            elif 0.834 <= Y_reg_end < 0.844:      # Estado 6
                return 6
            elif 0.844 <= Y_reg_end < 0.852:      # Estado 7
                return 7
            elif 0.852 <= Y_reg_end < 0.862:      # Estado 8
                return 8
            elif 0.862 <= Y_reg_end < 0.871:      # Estado 9
                return 9
            elif 0.871 <= Y_reg_end < 0.880:      # Estado 10
                return 10
            elif 0.880 <= Y_reg_end < 0.890:      # Estado 11
                return 11
            elif 0.890 <= Y_reg_end < 0.899:      # Estado 12
                return 12
            elif 0.899 <= Y_reg_end < 0.908:      # Estado 13
                return 13
            elif 0.908 <= Y_reg_end < 0.917:      # Estado 14
                return 14
            elif 0.917 <= Y_reg_end < 0.927:      # Estado 15
                return 15
            elif 0.927 <= Y_reg_end < 0.936:      # Estado 16
                return 16
            elif 0.936 <= Y_reg_end < 0.945:      # Estado 17
                return 17
            elif 0.945 <= Y_reg_end < 0.955:      # Estado 18
                return 18
            elif 0.955 <= Y_reg_end < 0.964:      # Estado 19
                return 19
            elif 0.964 <= Y_reg_end < 0.973:      # Estado 20
                return 20
            elif 0.973 <= Y_reg_end < 0.982:      # Estado 21
                return 21
            elif 0.982 <= Y_reg_end < 0.992:      # Estado 22
                return 22
            elif 0.992 <= Y_reg_end < 1.001:      # Estado 23
                return 23
            elif 1.001 <= Y_reg_end < 1.010:      # Estado 24
                return 24
            elif 1.010 <= Y_reg_end < 1.019:      # Estado 25
                return 25
            elif 1.019 <= Y_reg_end < 1.029:      # Estado 26
                return 26
            elif 1.029 <= Y_reg_end < 1.039:      # Estado 27
                return 27
            elif 1.039 <= Y_reg_end < 1.048:      # Estado 28
                return 28
            elif 1.048 <= Y_reg_end < 1.057:      # Estado 29
                return 29
            elif 1.057 <= Y_reg_end < 1.066:      # Estado 30
                return 30
            elif 1.066 <= Y_reg_end < 1.075:      # Estado 31
                return 31
            else:
                return 32                         # Estado 32
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def calculo_reward(self, Y_reg_end_nuevo):
        """_Resumen_: Esta función es directa, revisa si los valores del Vreg('Y_reg') están entre los valores
        máximos y mínimos aceptables para el sistema. Se la llama en eval_state_action.
        Cual es el mecanismo para salir del rew = -1, debo buscar la forma de salir de ahi, cómo??? Obligar al TAP que se mueva de ahí para arriba
        Mientras mas se aleje del valor nominal una mayuor penalidad... OJO!!!!!
        Comparar con controladores lineales y no lineales, verificar su comportamientoe implemntar en el algoritomo...!!!!
        """
        print(f"Y_reg_end: {Y_reg_end_nuevo}, Rango requerido: ({self.desired_min}, {self.desired_max})")
        if Y_reg_end_nuevo is None:
            return -10  # Penalización alta si no se recibe un valor válido

        if 0.99 <= Y_reg_end_nuevo <= 1.01:
            return 10  # Máxima recompensa dentro del rango óptimo
        elif 0.97 <= Y_reg_end_nuevo < 0.99 or 1.01 < Y_reg_end_nuevo <= 1.03:
            return 5  # Recompensa media
        elif 0.95 <= Y_reg_end_nuevo < 0.97 or 1.03 < Y_reg_end_nuevo <= 1.05:
            return 2  # Recompensa baja
        else:
            return -5  # Penalización para valores fuera del rango aceptable
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def is_terminal_state(self, Y_reg_end_nuevo):
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
            #terminal_state = [0, 1, 2, 3, 4, 5, 6, 7] [23, 32] = True
            #terminal_state = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] [24, 25, 26, 27, 28, 29, 30, 31] = False
        """
        return bool(Y_reg_end_nuevo <= 0.853 or Y_reg_end_nuevo >= 1.075 or (0.992 <= Y_reg_end_nuevo < 1.002))
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

    agent = PolicyIterationAgent(nS = 32, nA = 3, gamma = 0.88, eps = 1, eng = eng, ip = '127.0.0.1', port = 9096)  # Inicializo el agenteps = 0.01

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
        #agent.plot_simulation_results()

        print("\n La matriz de la Funcion del Valor Vpi: ",agent.V.reshape((1, 10)))
        print("\n La matriz de la politica PI es: ", agent.policy.reshape((1, 10)))

    #Cierro Matlab
    finally:
        eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'stop')", nargout=0)
        eng.quit()
