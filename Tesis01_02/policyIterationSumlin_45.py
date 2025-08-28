# -*- coding: utf-8 -*-
#START
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matlab.engine
import numpy as np
import struct, socket, time
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
    Value Functions son herramientas fundamentales para evaluar cuán "bueno" es un estado (S_t)
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
        git add .
        git commit -m "Actualización de scripts y archivos .mat para control Sumlink"
        git push origin nueva-version

        git checkout -b Capitulo-03-v45 (Para crear una nueva rama)

'''
class PolicyIterationAgent:
    """
    Agente de Iteración de Políticas para el control del TAP en un sistema eléctrico.
    Interactúa con Simulink para evaluar políticas y mejorar la estabilidad del voltaje
    """
    #--------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 nS,
                 nA,
                 gamma,
                 eps, #Tolerancia 1e-3
                 eng,
                 prob_satis = 0.8,
                 pausa = None,
                 host = '127.0.0.1',
                 udp_packet_size: int = 4,
                 port = 9096):
        """_summary_

        Args:
            nS (int): Número de estados posibles
            nA (int): Número de acciones disponibles.
            gamma (float): Factor de descuento.
            eps (float): Tolerancia para la convergencia.
            eng (matlab.engine): Motor de MATLAB para interactuar con Simulink.
            prob_satis (float): Probabilidad de éxito en la transición de estado, por defaults a 0.8.
            pausa (float): Defaults to None.
            host (str): Dirección IP para comunicación UDP. Defaults to '127.0.0.1'.
            port (int): Puerto para comunicación UDP. Defaults to 9096.
        Raises:
            ValueError: nS debe ser > 0
            ValueError: nA debe ser > 0
            ValueError: gamma debe estar entre 0 y 1
            ValueError: Numero de acciones no soportadas
        Inicializa el agente de iteración de políticas.

        """
        # Validación de parámetros
        if not isinstance(nS, int) or nS <= 0:
            raise ValueError(f"nS debe ser un entero positivo, recibido: {nS}")
        if not isinstance(nA, int) or nA <= 0:
            raise ValueError(f"nA debe ser un entero positivo, recibido: {nA}")
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma debe estar entre 0 y 1, recibido: {gamma}")

        # Inicialización básica
        self.nS = nS

        self.nA = nA                                    #Por las acciones binarias... 0/1 siempre multiplo de 2
        if self.nA == 3:
            self.acciones = {0: -1, 1: 0, 2: +1}
        elif self.nA == 2:
            self.acciones = {0: -1, 1: +1}
        else:
            raise ValueError("Numero de acciones no soportadas")

        self.gamma = gamma                                #Factor de descuento
        self.eps = eps
        self.eng = eng

        self.Ts_step   = getattr(self, 'Ts_step', 0.02)   # sample time del modelo
        self.tx_guard  = getattr(self, 'tx_guard', 0.005) # 5 ms de guarda post-toggle
        self.rx_wait   = getattr(self, 'rx_wait', 1.0)    # espera máx. lectura UDP

        self.udp_packet_size = udp_packet_size

       # Estados de Valor y Politica
        self.V = np.zeros(nS)
        self.policy = np.zeros(nS, dtype=int)

        # Probabilidad de transiciones
        self.prob_satis = prob_satis

        # Pausa basada en Simulink o valor por defecto
        self.pausa = pausa
        if self.pausa is None:
            try:
                self.pausa = float(self.eng.workspace['T'])
            except Exception:
                self.pausa = 3.0
        else:
            self.pausa = float(pausa)

        # Configuración red y socket
        self.udp_host = host
        self.udp_port = port
        self.udp_socket = self._init_socket()

        self.pos_max_tap = 16
        self.pos_min_tap = -16

        self.V_nominal = 13.8e3
        self.initial_Y_reg = 1.0                        #Valor asumido inicialmente
        self.V_base_fase = self.V_nominal/np.sqrt(3)

        self.Y_reg_val = []
        self.tap_pos_val = []

        # Matriz de transición vacía
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        # Sincronizar estado inicial con Simulink
        self.estado_actual = self.sincronizar_estado_inicial()

        self.last_tap = 0
        self.tap_action = 0                             # Inicializa la variable para almacenar el TAP
        self.tap_initialized = False                    # Variable de control para saber si ya se usó el 0

        self.desired_min = 0.95
        self.desired_max = 1.05

    #--------------------------------------------------------------------------------------------------------------------
    def _init_socket(self):
        """
        Crea y retorna un socket UDP configurado.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)  # timeout ajustable
        sock.bind((self.udp_host, self.udp_port))
        return sock
    #--------------------------------------------------------------------------------------------------------------------
    def policy_evaluation (self):
        """
        Evalúa la política actual y actualiza los valores de estado 'V(s)' hasta la convergencia.
        Utiliza la ecuación de Bellman para calcular el valor esperado de cada estado.
        V(s) = sum_a pi(a|s) * sum_{s',r} P[s][a] (r + gamma * V(s'))
        """
        import time
        iteracion = 0
        valores_delta = []
        start_time =time.time()

        while True:
            delta = 0.0
            # Paso 2: Actualizar los valores de estado de manera mas eficiente
            for s in range(self.nS):
                old_v = self.V[s]
                # Evalúo la acción que indíca la política en s
                self.V[s] = self.eval_state_action(s, self.policy[s])
                # Máxima diferencia para el critério de parada
                delta = max(delta, np.abs(old_v - self.V[s]))
                print(f"State: {s}, Old V[s]: {old_v}, New V[s]: {self.V[s]},  Delta: {delta:.4f}") #Reward: {reward},

            valores_delta.append(delta)
            iteracion += 1
            print(f"Iteration {iteracion}: max delta={delta:.4f}")

            if delta < self.eps:
                """
                Paso 3: Condición de convergencia, se repite hasta que cumpla.
                La convergencia V(s) significa que el agente ha aprendido la calidad de cada estado, dado el control actual del TAP.
                El sistema ha alcanzado una represetnación estable de los efecto del control del TAP sobre el voltaje en Yreg
                Las transiciones y r imediatas estan correctamente integradas en el valor esperado de cada estado.
                """
                break
        # Tiempo que toma para que la evaluación de la política sea estable
        end_time = time.time()
        print(f"El tiempo total de evaluación es: {(end_time - start_time) / 60.0:.2f} minutes")
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        plt.figure()
        plt.plot(valores_delta)
        plt.xlabel('Iteraciones')
        plt.ylabel('Delta')
        plt.title('Convergencia Delta vs. Iteraciones')

        filename = f'Delta_Convergencia_{timestamp}.png'
        save_dir = 'delta'
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)
        plt.close()
        print(f"Imagen guardada en: {image_path}")
    #--------------------------------------------------------------------------------------------------------------------
    def eval_state_action(self, s: int, a: int) -> float:
        """
        Devuelve Q(s,a) =  E[ r + γ·V(s') ] considerando que una misma
        acción puede derivar en varios desenlaces (éxito, fallo, etc.).

        1) Obtiene la lista de transiciones con self.mat_tran_gen(s, a):
        (p, next_s, r, done)
        2) Suma p·(r + γ·V[next_s]) en cada rama
        • Si 'done' es True, no se añade el término futuro.
        Returns:
            float: EL valor esperado, que usa policy_evaluation y policy_improvemnet.

        """
        return sum(
            p * (r + (0.0 if done else self.gamma * self.V[next_s]))
            for p, next_s, r, done in self.mat_tran_gen(s,a)
        )
    #--------------------------------------------------------------------------------------------------------------------
    def policy_improvement(self)-> bool:
        """_summary_:
            Mejora la política π(s) seleccionando en cada estado la acción que maximiza Q(s,a),
            usando un caché temporal para evitar simulaciones redundantes.
            Retorna True si la política ya no cambia (es estable).
        """
        print('Mejorando la politica')
        policy_stable = True
        ac_tomada = []
        q_cache = {} # (s,a) -> Q(s,a), evita llamdas repetidas a Simulink

        for s in range(self.nS):    # Paso 1: Recorro todas los S_t
            old_a = self.policy[s]
            q_values = []

            for a in range(self.nA):
                if (s, a) not in q_cache:
                    q_sa = self.eval_state_action(s, a)
                    q_cache[(s, a)] = q_sa
                q_values.append(q_cache[(s, a)])

            #Calculo Q(s,a) para todas las acciones
            #q_values = [self.eval_state_action(s,a) for a in range(self.nA)]
            best_action = int(np.argmax(q_values))
            ac_tomada.append(best_action)

            # Actualizar la política
            self.policy[s] = best_action
            #Paso 3: Nueva PI' que sea mejor o = que PI. Si la A_(t+1)-->best_action
            if best_action != old_a:   #Si nunguna de las A_t mejora PI, entonces PI es estable =True
                print(f"[PI]  Estado {s:2d}:  acción {old_a} → {best_action}")
                policy_stable = False   #Si la A_t cambia, significa que PI no era estable

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Gráfico de las acciones tomadas
        plt.figure(figsize=(10,6))
        plt.plot(range(len(ac_tomada)), ac_tomada, label='Acciones tomadas')
        plt.xlabel('Estados')
        plt.ylabel('Acción')
        plt.title('Evolución de las Acciones durante policy_improvement')
        plt.legend()

        filename = f'Acciones_policy_improvement_{timestamp}.png'
        save_dir = 'mejorPolitica'
        os.makedirs(save_dir, exist_ok=True)
        archivo_acciones = os.path.join(save_dir, filename)
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)
        plt.close() # Si no cierro, la simulación se para
        print(f"Gráficas guardadas: {archivo_acciones}")

        return policy_stable
    #--------------------------------------------------------------------------------------------------------------------
    def choose_action (self, s: int, epsilon: float = 0.05)->int:
        """
        Devuelve una acción siguiendo una política ε-greedy sobre Q(s,a).

        • Con prob. ε    → elige una acción aleatoria  (exploración)
        Acción aleatoria, sin importar que sea buena o mala.
        • Con prob. 1-ε  → elige argmax_a Q(s,a)       (explotación)
        El agente elige la mejor acción que conoce hasata ahora
        """
        if np.random.rand() < epsilon:
            # Exploración
            return np.random.randint(self.nA)

        # Explotación: calcular Q(s,a) para cada acción
        q_values = [self.eval_state_action(s, a) for a in range(self.nA)]
        return int(np.argmax(q_values))
    #--------------------------------------------------------------------------------------------------------------------
    def int_simple_simulink(self, Ts: float = 0.02, max_wait: float = 2.0) -> None:
        # Verifico el estado de la simulación en SIMULINK
        mdl = 'AC_Feeder_Control'
        sim_status = self.eng.get_param(mdl, 'SimulationStatus')

        if sim_status in ('stopped', 'compiled', 'terminating'):
            self.eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)
        elif sim_status == 'paused':
            self.eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'continue')", nargout=0)
        elif sim_status == 'running':
            print("La simulación está corriendo.")
            return
        else:
            print(f"Estado desconocido de Simulink: {sim_status}. Intentando iniciar la simulación...")
            self.eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'start')", nargout=0)

        t0 = float(self.eng.get_param(mdl, 'SimulationStatus'))
        t1 = t0 + Ts
        self.eng.get_param(mdl, 'StopTime', str(t1), nargout = 0)
        self.eng.eval("set_param('AC_Feeder_Control','SimulationCommand','continue')", nargout=0)

        t_start = time.time()
        while time.time() - t_start < max_wait:
            t_sim = float(self.eng.get_param(mdl, 'SimulationTime'))
            if t_sim >= t1:
                return
            time.sleep(0.005)
        print("[int_simple_simulink_step] Aviso: no alcanzó t1 dentro de max_wait.")
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
    def sincronizar_estado_inicial(self):
        #Leo el tap de Simulink
        tap_inicial = float(self.eng.workspace['tap'])
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
            return 16 + abs(tap)
#--------------------------------------------------------------------------------------------------------------------
    def get_tap_desde_state(self, s):
        """
            Convierte un estado 's' en la posición del TAP correspondiente en Simulink.
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
    def matObj(self, drain: bool = False, max_wait: float | None = None):
        # Verificar si el socket ya está creado
        #if not hasattr(self, 'udp_socket') or self.udp_socket is None:
        #    self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #    self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #    self.udp_socket.bind((self.udp_host, self.udp_port)) #es la parte que puede tener problemas 
        #    self.udp_socket.settimeout(3)  # Timeout para recibir datos
        #   print(f"Socket creado y enlazado a {self.udp_host}:{self.udp_port}")

        if drain:
            self.limpiar_buffer()
        # Esperar de forma consistente
        if max_wait is None:
            max_wait = self.rx_wait
        deadline = time.time() + max_wait
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                print("[matObj] Timeout: no se recibieron datos en ventana de espera.")
                return 0.99
            self.udp_socket.settimeout(remaining)
            try:
                data, _ = self.udp_socket.recvfrom(self.udp_packet_size)
                if len(data) < 4:
                    continue

                vreg = round(struct.unpack('<f', data)[0], 3)

                # Convertir a p.u. y redondear
                Y = round(vreg / self.V_base_fase, 3)
                print(f"[matObj] V_reg = {vreg} V → Y_reg_end = {Y} p.u.")
                return Y
            except socket.timeout:
               continue
            except OSError as e:
                print(f"[matObj] Error de socket: {e}")
                return 0.99  # Valor por defecto en caso de error
    #--------------------------------------------------------------------------------------------------------------------
    def limpiar_buffer(self, max_drain: int = 200):
        """
        Limpia el buffer del socket UDP para evitar datos residuales antes de leer nuevos datos.
        """
        # Limpiar el buffer del socket
        intento = 0
        curt_to = None

        try:
            cur_to = self.udp_socket.gettimeout()
        except Exception:
            pass

        self.udp_socket.setblocking(False)
        try:
            for _ in range(max_drain):
                try:
                    self.udp_socket.recvfrom(self.udp_packet_size)
                    intento += 1
                except (BlockingIOError, InterruptedError, OSError):
                    break
        finally:
            self.udp_socket.setblocking(True)
            if cur_to is not None:
                self.udp_socket.settimeout(cur_to)
        if intento:
            print(f"[UDP] Buffer drenado: {intento} paquetes descartados.")
    # ------------------------------------------------------------------
    # Paso 5 · Generador de transiciones estocásticas P(s,a)
    # ------------------------------------------------------------------
    def mat_tran_gen(self, s: int, a: int):
        """
    Genera la matriz de transición `self.P[s][a]` ajustando las probabilidades 
    de transición en función de la acción aplicada al TAP y su impacto en el voltaje `Y_reg_end`.
    Devuelve una lista de tuplas (p, next_s, r, done) que describe
        todas las ramificaciones posibles al ejecutar la acción `a`
        desde el estado `s`.

        • Rama 1  (éxito)  -> prob = self.prob_satis
        • Rama 2  (falla)  -> prob = 1 - self.prob_satis

        Cada rama:
          p         → probabilidad de ocurrir
          next_s    → estado discreto alcanzado
          r         → recompensa inmediata
          done      → True si el episodio termina en esa rama
        """
        transiciones = []

        # 1) Asegurar simulación corriendo ANTES de aplicar cambios
        self.int_simple_simulink()

        # 2) Preparar TAPs
        # Calcula el tap destino  aplicando delta y acotando
        delta_tap   = self.acciones[a]    # {0: -1, 1: 0, 2: +1} o {-1, +1}
        tap_actual  = self.get_tap_desde_state(s)

        tap_ok      = int(np.clip(tap_actual + delta_tap,
                                    self.pos_min_tap,
                                    self.pos_max_tap))
#--------------------------------------------------------------------------------------------------------------------
        # Fijo el TAP en Simulink y se simula
        def _aplicar_rama(prob: float, tap_destino: int):
            self.limpiar_buffer()

            # 2.1) Fijar el TAP vía wokspace-Matlab
            self.eng.workspace['tap'] = float(tap_destino)
            self.eng.eval("set_param('AC_Feeder_Control/Tap','Value','tap')", nargout=0)

            # 2.2) Toggle de clk (0<->), si no existe, inicaliza en False
            try:
                clk_actual = bool(self.eng.workspace['clk'])
            except Exception:
                clk_actual = False
                self.eng.workspace['clk'] = clk_actual
            self.eng.workspace['clk'] = (not clk_actual)
            self.eng.eval("set_param('AC_Feeder_Control/Clk','Value','clk')", nargout=0)
            self.int_simple_simulink_step(self.Ts_step)
            time.sleep(self.tx_guard)  # 1–5 ms suele bastar
            # 2.3) Leer medición (UDP bloqueate con limpiar_buffer interno)
            Y = self.matObj(drain = False, max_wait = self.rx_wait)

            # 2.4) Armar transición
            n_s  = self.next_state(Y)
            r       = self.calculo_reward(Y)
            done    = self.is_terminal_state(Y)

            transiciones.append((prob, n_s, r, done))
#--------------------------------------------------------------------------------------------------------------------
        # Rama ÉXITO (aplicando delta)
        _aplicar_rama(self.prob_satis, tap_ok)

        # Rama FALLA (amantener tap)
        _aplicar_rama(1.0 - self.prob_satis, tap_actual)

        #Verificaicón de la normalización de 'p'
        total_prob = sum(p for p, *_ in transiciones)
        assert abs(total_prob - 1.0) < 1e-6, "Las probabilidades no suman 1."

        return transiciones
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

        if 0.992 <= Y_reg_end_nuevo < 1.002:
            return 10  # Máxima recompensa dentro del rango óptimo
        elif 0.97 <= Y_reg_end_nuevo < 0.992 or 1.002 <= Y_reg_end_nuevo <= 1.03:
            return 5  # Recompensa media
        elif 0.95 <= Y_reg_end_nuevo < 0.97 or 1.03 < Y_reg_end_nuevo <= 1.05:
            return 2  # Recompensa baja
        else:
            return -5  # Penalización para valores fuera del rango aceptable
    #--------------------------------------------------------------------------------------------------------------------
    def is_terminal_state(self, Y_reg_end_nuevo):
        """
        Determina si un estado es terminal.
        """
        if Y_reg_end_nuevo is None:
            return True

        return (
            Y_reg_end_nuevo < 0.95 or
            Y_reg_end_nuevo > 1.05 or
            (0.992 <= Y_reg_end_nuevo < 1.002)
            )
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
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, filename)
        plt.savefig(image_path)
        plt.close()
        print(f"Imagen guardada en: {image_path}")
#--------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    eng = matlab.engine.start_matlab()   # Inicio el motor de Matlab
    eng = matlab.engine.connect_matlab() # Quita el # de la ventana de MATLAB si ya estamos compartiendo Matlab

    # Cargo el modelo y parámetros de Simulink
    eng.load_system('AC_Feeder_Control', nargout = 0)
    eng.run('AC_Feeder_Control_Param_02.m', nargout=0)

    agent = PolicyIterationAgent(
        nS = 33,
        nA = 3,
        gamma = 0.88,
        eps = 7, # Tolerancia 1e-3
        eng = eng,
        host = '127.0.0.1',
        port = 9096
        )

    # Ciclo principal
    try:
        policy_stable = False                               # Inicializo la Política PI FALSE = no es etable
        it = 0
        while not policy_stable:                            # Mientras que policy_stable no cambie a True..
            agent.policy_evaluation ()                      # Evaluación de las policy
            policy_stable = agent.policy_improvement()
            it +=1
        print('Convergencia despues de %i  interaciones --> policy (Politicas)' % (it))
        print("\nVπ:", agent.V)
        print("\nπ:", agent.policy)

    #Cierro Matlab
    finally:
        eng.eval("set_param('AC_Feeder_Control', 'SimulationCommand', 'stop')", nargout=0)
        eng.quit()