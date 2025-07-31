# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:30:56 2023

@author: pablo.gamboa

Capitulo 3
RL-Puede formularce como un Markov Decision Process (MDP).
    MDP--> provee los elementos claves del RL:
        1.- Value functions V-Function
        2.- Expected Reward E-Reward
            RL algorithms pueden ser creados usando estos elementos.
        Diferencia entre 1 y 2:
            Combinacion entre ellas
            Asumciones que se realiza mientras se las diseña.
                Por estas razones RL, se categorizan en 3, y se pueden convinar unas con otras.
                    1.- MDP
                    2.- Categorizing RL algorithms
                    3.- Dynamic Programming
    -------------------------------------------------------------------
    |1.- MDP:- Representan los problemas 'sequential decision-making'. |
    -------------------------------------------------------------------
        Las A_t influyen en el siguiente S_(t+1) y en el RESULTS.
        Porque RL es compatible con--> MDP?:
            - MDP es flexible, y se puede formalizar un problema de aprendizaje a trevés de 'iteraciones'.
            - RL tambien puede solucionar este tipo de problemas con 'iteraciones'.
        MDP-->tupla (S_t,A_t,P,R)
            S: State space--> finite set of space
            A: Action space--> finite set of action
            P: Transition function
                p(s'|s,a): probabilidad de alcanzar S_(t+1) dado que el sistema esta en S_t(actual) y se toma A_t
            R: Reward function--> determina el valor recibido por la transición S_(t+1) después de tomar A_t del S_t
        El MDP se lo controla por una secuencia de pasos discretos de tiempo --> crea una trayectoria
        de S_t y A_t (S_0, A_0, S_1, A_1).
        S_t sige la dinámica del MDP con la 'función state transition'-->p(s'|s,a)
            De esta forma la 'transition function' caracteriza el Env dinámico.

        Por definición: 'transition function S_(t+1) y Reward Function R_t' --> por el S_t--> estado actual.
            Se lo conoce como una 'Markov property'-->proceso es-->memory-less
                S_(t+1), depende solo de S_t, NO de su historial.
                    El --> S_t contiene toda la Info.
                        A este sistema se lo llama .'FULLY OBSERVABLE'
                    Si solo uso un numero finito de S_(t-1)-->'PARTIALLY OBSERVABLE'
                        S_t --> son llamados observaciones
        El objetivo final del MDP--> es encontrar la 'Policy (PI)' (A_t y S_t) que maximice la recompenza acumulada
        'SUM R_t', como 'argmax_(PI)E_(PI)[G(Tau)]
        La solución del MDP es encontrada--> CUANDO una POLICY toma la mejor 'A_t' en cada 'S_t' del MDP.
        Se la conoce como 'optimal policy'
    ----------------------------------------------
    | 1.1. POLICY | OR DECISION MAKER | OR AGENT |
    ----------------------------------------------
    Es una estrategia o regla que define el comportamiento del AGENTE. Una PI especifica la A_t
    que el agente debe tomar en cada S_t del ENV para Maximizar alguna medida de rendimiento, tipicamente
    el REWARD R_t
            - Politicas Deterministicas: a_t = u(S_t), variables fíjas y predecibles.
            - Políticas Estocastica: a_t ~PI(.|S_t)-->~ 'has distribution'.
                Se las usa cuando se considera una 'distribución A_t', A_t son fijas y predecibles.
                Las A_t pueden ser:
                    - Categoricas-->Variables discretas--> un solo numero contable-->Problema de clasificación.
                        #de hijos
                        #de libros vendidos
                    - Gausiana (normal)-->varibles continuas-->Toma un valor dentro de un intervalo
                        descrito por:
                            - Media -->función de S_t
                            - Desviación estándar o varianza -->función de S_t
    -----------------------------------------------------------------------------------------------
    | 1.2. RETURN G(tau)--> TRAJECTORY OR ROLLOUT proporciona un buen valor interno NO en calidad |
    -----------------------------------------------------------------------------------------------
    Es la recompensa total acumulada que un AGENTE Rx a lo largo del tiempo (t) a partir de un S_t o A_t.
    Este es el valor que el AGENTE trata de maximizar mientras interactua con el ENV. 


    Con la definición de RETURN, se puede definir el Objetivo de encontrar la PI óptima que MAX RETURN.
        PI Optima, es una A_t óptima para cada S_t.
        Al correr una politica 'PI' en MDP se genera una SECUENCIA --> S_t y A_t (S_0, A_0, S_1, A_1)
        --> 'TRAYECTORY or ROLLOUT'
        En cada 'TRAYECTORY or ROLLOUT':
            Se collecta --> Secuencia de R_t que se genera como --> resultado A_t
                Se lo llama 'F-RETURN'
                Puede ser analizado por trayectorias:
                    - Infinitas soluciones-->Env no determinado, la suma del R_t siempre sera Inf.
                        Situación péligrosa, pq no se da ninguna información.
                            Estas tareas son llamadas: TAREAS CONTINUAS, necesitan otra formulación.
                    - Finitas Soluciones
                        - Se le da > peso a las R_t a corto plazo
                        - < peso a las R_t a las mas lejanas.
                        Se usa un 'DISCOUNT FACTOR'-->'lammda'
                        El uso -->lammda incrementa la estabilidad del algorítmo
                            La R_t muy futuras, se las considera parcialmente.

                            lammda[0.9-0.999]
                            lammda=--> 1
                                Si no se tiene apuro--> Más número de iteraciones.
                            lammda=--> 0
                                Se quiere un resultado ya --> Menos número de iteraciones.
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
    Si se calcula de forma separada V-function and Q-function, consume muchos recursos, considerando que el
    RETURN requiere R_t de toda la trayector de S_t y A_t

    BELLMAN EQUATION define Q y V function recursivamente
        Recursivo--> se lo llama a si mismo en la ejecución
            Simplifica problemas complejos.
                Usa los R_t obtenida el el S_t y el valor del S_(t+1)
        Con esta recursividad de BELLMAN V_(PI) y Q_(PI) se actualizan solo con S_(t+1)
            No necesita recorrer todos las trayctorias hasta el END.

    -----------------------------------------
    | 2. CATEGORIZING RL ALGORITHMS       |
    -----------------------------------------
    2.1. Model-based algorithms
        - Requiere un Env.
            -Env. es muy valorado, la información proporcionada ayuda a encontrar la PI deseada.
                PI--> determinar la A_t y S_t
        El tener un Env. signifíca que el S_(t+1) y los R_t puede ser predecidos por cada tupla (S_t, A_t)
            No requiere interacción con el Env. real
            Se lo usa en aplicaciones del modelo para A_t futuras.
                A_t futuras: Planificación  de movimientos futuros anticipando las consecuencias de los movimeintos
                siguientes.
        Un modelo tambien puede ser aprendido a través de interacciones con el Env. asimilando las consecuencia
            En terminos del S_t y R_t por una A_t
            NO es la mejor solución, en el mundo real, es muy costoso.
                Solo se tiene una aproximación rustica del Env.
                    Da un resultado DESASTROSO.
        Un modelo conocido o aprendido, Se lo usa para planificar y mejorar la política en distintas fases del algorítmo.
            Casos comunes incluyen:
                Planificación pura
                Planificaicón integrada
                Generación de muestras de un modelo aproximado.
        Se usa Dynamic Programming (DP)

    2.2 Model-free algorithms MF
        - No requiere un Env.
        Se descompone en:
            - Policy gradient algorithms.
            - Value-based algorithms.
                HYBRIDS
                    Combinan caracteristicas importantes de PI gradienet and V-based
        Al no tener Env. se corre trayectorias dentro de una PI para ganar experiencia y mejorar al AGENT.
        Se debe seguir 3 pasos, con la combinación de estos se puede generar diferentes algoritmos tales como:
            1.- Value-based algorithm
            2.- Policy gradient algorithm
            Pasos:
                1.- Generación de nuevas muestras para correr la PI en el Env.
                    Las trayectorias corren hasta que se alcance el S_t final o se cumpla un número de PASOS.
                2.- La estimación de la funcion RETURN
                3.- El mejoramiento de la PI usando por la recolección de muestras y la estimación del RETURN-->paso 2

    2.2.1. Value-based algorithm--> OFF_POLICY
        Conocido como Value-based algorithms
            Usan la ecuación de Bellman 1.4 para aprender la Q-function
                Con la Q-function va ha aprender la PI
                Usan DNN como una función de aproximación.
                Trucos para tratar:
                    Varianza
                    Inestabilidades
                V_function--> son similares a un algoritmo de regresión.
        No requiere optimizar la misma PI que se uso para generar los datos
            Estos métodos pueden aprender de la experiencia previa.
            Pueden almacenar los datos de muestra en un buffer
                El poder usar los datos previos, hacen que el V_function sea más eficiente que otro modelo FREE

    2.2.2. Policy gradient algorithm
        Interpretación directa y ovia del RL.
        Aprenden directamente de la POLICY parametrizada
            Se actualizan los parametros en la dirección de las mejoras.
            Se premia las buenas A_t
            Se desalienta una mala A_t
        A diferencia del Value-fuction:
            Se requiere datos de POLICY
                Algoritmo ineficiente.
                Son algoritmos inestables.
                    Solución:
                        Optimización de la PI--> regiones confiables
                        Optimización de la Función Objetivo
                            Para limitar los cambios en la PI
        POLICY GRADIENT METHODS
            Gestiona ENV con espacios de A_t continuos.

    2.2.3. Actor Critic algorithms
        Combinan un ACTOR que toma decisiones y un CRITICO que evalúa esa desición para mejorar la PI.
        Se optimiza la PI incluso sin alcanzar el Objetivo FInal.

    2.2.4. Algorimos Hybridos
            Combinan V-function y las políticas de Gradiente, para ser eficiente y robusto.

    Resumen:
            Policy Gradient Algorithm: Más estable
            V-function: Más eficientes en el uso de muestras
            V-function methods: Eficientes, pero consumen mas recursos computacionales

    -----------------------------------------
    | 3. DYNAMIC PROGRAMMING (DP)         |
    -----------------------------------------
    Secciona un problema en TROZOS pequeños
        Con estos trozos encuentra soluciones a problemas complejos.
            Combinando soluciones de subproblemas.
    DP--> para RL tiene uno de los enfoques mas simples:
        - Calcula PI óptimas (S_t, A_t)--> óptimo
            Al contar con un Env. perfecto del entorno.
        - Es computacionalmente muy costoso.
        - Trabaja con MDP
            Limitado # de S_t y A_t
    Almacena V-function en una tabla
        Acceso rápido y sin perdida de informacion
            Pero, requiere mucho espacio de almacenameinto.
            Metodo tabular
            Approximating Learning: Utiliza NN par agestionar las V-fuction en tamaños fijos.
    Usa BOOTSTRAPPING
        Mejora la estimación de los Valores de un S_t por el uso de un Valor esperado del S_(t+1)
        Se usa en la ecuaciópn de BELLMAN
            V-Fuction
            Q-Fuction

    3.1 Policy Evaluation (PI) and Policy Improvement (PI')
    3.1.1. Policy Evaluation (PI)
                Primero Encuentro la V-function óptima  V_(k+1)(S_t) uso la Eq 3.8
                        Creando una secuencia {V0...Vk}
                            En cada iteración mejora el V-function para una política PI
                                Usa el S_t-->transición, S_(t+1) y el R_t-->inmediato
                                Se crean una secuancia de V-functions MEJORADOS-->BELLMAN equation
                                    La puedo encontrar solo si tengo:
                                        El 'state transition function-->p'
                                        'reward function-->r'
                                    Para cada S-t y A_t que se conoce
                                    SOLO SI EL MODELO DEL ENV ES CONOCIDO COMPLETAMENTE
                                    Se utiliza Políticas DETERMINISTICAS-->línea 47
    3.1.2. Policy Improvement (PI'): Eq. 3.9
            Se crea una política PI'--> con base V_(PI), con la Policy Evaluation (PI)
            PI' es siempre mejor que PI
            PI* es óptimo si V_function es óptimo.
            Combinar PI'-PI da como resultado 2 algorítmos:
                1.- POLICY ITERATION PI'
                    Dos pasos:
                        1.- Evalauación PI: Actualizo V_Funcion de PI usando BELLMAN
                        2.- Mejora de PI: Se calcula una nueva PI'--> EQ3.9
                            PI' es mejor o 0 que la PI anterior
                                Pq elije las A_t que maximizan el valor esperado.
                    Se repite de forma cíclica durante 'n' iteraciones
                    Converge en la PI óptima--> PI*

                    Computacionalmente es costoso, ya que debe evaluar la PI actual en cada iteración.
                2.- VALUE ITERATION
            1 y 2 usan PI para mejorar el V_Function y PI'
            DIFERENCIA: 1 ejecuta las dos faces cyclicamente--> V-Fuction y PI' para estimar el nuevo PI
                        2 combina V-Function y PI' en una sola actualización.
#########################################################################################################################
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
---------------------------
|  B   | HOLE |  B  | HOLE|
---------------------------
|  B   |  B   |  B  | HOLE|
---------------------------
| HOLE |  B   |  B  | END |
---------------------------

PlicyIteration.py corre de forma iterativa, la polítca de evalacuación y mejora

"""
import gym
import numpy as np

#----------------------------------------------------------------------------

#Primer paso-->Policy evaluation
def eval_state_action(V, s, a, gamma = 0.99):
    """_summary: Función para evaluar la action-value esperado, que sera usado en la función policy_improvement.
    P --> accede a una estructura de datos q continen info de la policy para s y a-->Diccionario.

    Args:
        V (_type_): value-function
        s (_type_): estado actual
        a (_type_): action
        p = la probabilidad de transitar el estado 's' al siguiente estado 'next_s' al tomar la acción 'a'
        next_s = El siguiente estado al toamr una acción 'a' en el estado 's'
        rew = La recompensa asociada con la transición del estado 's' al 'next_s'
        rew = función reward
        gamma (float, optional): factor de descuento, 0.99
        np.sum([p * (rew + gamma*V[next_s]) Ecuación 3.8 pagina 62

    Returns:
        _type_:  policy evaluation and policy improvement-->pepi
    """
    print(env.P[s][a])
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

#----------------EVALUO MI POLÍTICA-------------------------------------------

def policy_evaluation (V, policy, eps):
    """_summary: Defino mi política de evaluación que ha sido calculada desde la formula 8, bajo la actual policy
    para cada estado 's', hasta alcanzar el estado estable. Solo se evalúa 1 acción, ya que la POLICY es deterministica.

    La función es estable cuando el 'delta' es < que 'eps

    Args:
        V (_type_): value-fuction
        policy (_type_): Toma una a_t en un estado s_t
        eps (float, optional): limite-->threshold
    """
    while True:             #Formula 3.8 pag. 62, pseudocode pg. 64 miesntras PI no es estable
        delta = 0           # la función es etable cuando delta sea < que eps
        for s in range(nS): #Lazo para todos los estados nS = 16-->cuadriculas estados de observación
            #print("Valor de s:", s)
            old_v = V[s]
            # Actualizo V[s] usando la ecuación de Bellman
            V[s] = eval_state_action(V, s, policy[s])   #policy[s] = action en 'eval_state_action'
            #print("El valor V[s]", V)
            delta = max(delta, np.abs(old_v - V[s]))
            #print("El valor de delta p_ev", delta)

        if delta < eps:
            break
#----------------EVALUO MI POLÍTICA-------------------------------------------
#-----------------------------------------------------------------------------

def policy_improvement(V, policy):
    """PI = Policy improvement, toma el V(Value-function) y la policy y las itera a lo largo de todos los 
    estados para actualizar la policy con base en el valor de la nueva función. Eq. 3.9, pag. 63 policy = PI

    Args:
        V (_type_): Value function
        policy (_type_): policy heredada
        nA = acciones-->Diccionario 4 acciones
            Policy iteration applied to frozenLake, 4 acciones 

            0 = left
            1 = down
            2 = right
            3 = up

    Returns:
        _type_: Plicy estable, despues de haber iterado todos los 16 estados 
    """

    #print('V: ', V, 'policy: ', policy)
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]   #Acción anterior es tomada del array de la policy en la ubicación 's'
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(nA)])
        if old_a != policy[s]:
            policy_stable = False #Significa que la policy[s] no es estable
    return policy_stable #Cuando todo se cumpla

#----------------------------------------------------------------------------

def run_episodes(env, policy, num_games):
    """_summary: Para probar la nueva policy e imprime el número de juegos ganados 

    Args:
        env (_type_): _description_
        policy (_type_): _description_
        num_games (_type_): _description_
    """
    tot_rew = 0
    for _ in range(num_games):
        state = env.reset()
        # Verificar si state es una tupla y extraer el estado si es necesario
        if isinstance(state, tuple):
            state = state[0]
        done = False
        while not done:
            if isinstance(state, int) and 0 <= state < len(policy):
                action = policy[state]
                result = env.step(action)
                if isinstance(result, tuple) and len(result) == 4:
                    next_state, reward, done, _ = result
                else:
                    next_state, reward, done = result[:3]
                    _ = None
                # Verificar si next_state es una tupla y extraer el estado si es necesario
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                state = next_state
                tot_rew += reward
            else:
                raise IndexError(f"Invalid state index: {state}")
            if done:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
    print('Gano %i de %i juegos!'%(tot_rew, num_games))


#-------------------------INICIALIZO el ENVIROMENT-----------------------------

if __name__ == '__main__':
    #Creo mi env e inicializo los valores de la función y la POLICY
    env = gym.make("FrozenLake-v1")
    env = env.unwrapped                                 # Cargar  esto si se tiene información adicional
    nA = env.action_space.n                             # Espacio de acción 4 discretos
    nS = env.observation_space.n                        # Espacio de observación-estados 16
    V = np.zeros(nS)                                    # V-Function
    policy = np.zeros(nS)                               # PI

    #Ciclo principal
    policy_stable = False                               #Inicializo la Política PI FALSE = no es etable
    it = 0                                              #it = iteración
    while not policy_stable:                            #Mientras que plicy_stable no cambie a True..
        policy_evaluation (V, policy, eps = 0.0001)     #Primer paso PI
        policy_stable = policy_improvement(V, policy)   #Segundo paso PI'
        it += 1
    print('Convergencia despues de %i  interaciones --> policy (Politicas)'%(it))
    run_episodes(env, policy,100)
    print("\n La matriz de la Funcion del Valor Vpi: ",V.reshape((4,4)))
    print("\n La matriz de la politica PI es: ", policy.reshape((4, 4)))