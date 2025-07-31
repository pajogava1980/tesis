
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:45:10 2023

@author: pablo.gamboa
Capitulo 4

SARSA TaxiV2

- Se aplica a environments no conocidos
- Environments dinámicos 
- Es tabular y la escalabilidad es limitada 
- SARSA se puede aplicar solo a enviroments con una pequeña y discreta at y st
- Rewar = +20 y una penalidad de -10
: = localidad vacia 
| = una pared que el taxi no puede atravezar 
R, G, B y Y son cuatro localidades 

El taxi recoge al pasajero en R y lo deja en B

SARSA necesita otros hyper parametros como argumentos para trabajar 

learning rate = lr = alfa, para controlar la cantidad de aprendizage en cada 
actualización 

num_episodes 

eps = valor inicial de egreedy policy 

gamma = factor de descuento usado para dar menos importancia a mas acciones en 
el futuro 

eps_decay = es el decremento lineal de eps a traves de los episodios S

"""
#-----------------------------------------------------------------------------
# import sys
# sys.path.append("c:\\users\\pagav\\anaconda3\\envs\\gymenv02\\lib\\site-packages")
#import gymnasium as gym
import gym
import numpy as np
from matplotlib import pyplot as plt 



def SARSA(env,lr = 0.001,num_episodes = 10000, eps = 0.3, gamma = 0.95,eps_decay = 0.00005):
    '''
    SARSA mejora el uso de montecarlo. Evita esperar hasta el final de la trayectoria
    SARSA-->Recolecta experiencias on-policy:
        on-policy es usada para recolectar experiencas de la interacción con el
        env. 
    

    Parameters
    ----------
    env : TYPE
        DESCRIPTION.
    lr : TYPE, optional, es el valor de alfa, [0.5-0.001]-->learning rate 
        DESCRIPTION. 1--> el state va a ser TD target 0--> State no cambia 
    num_episodes : TYPE, optional
        DESCRIPTION. The default is 10000.
    eps : TYPE, optional-->E-greedy valor inicial
        DESCRIPTION. The default is 0.3.
    gamma : TYPE, discount factor 
        DESCRIPTION. The default is 0.95.-->Factor de descuento.
    eps_decay : Estrategia utilizada para evitar explorar mucho, por lo tanto
    va decreciendo el valor de E, con esto la política va ac onverger gradualmente.
    Otra tecnica de exploración es BOLTZMANN--> mas complicada.
        DESCRIPTION. The default is 0.00005.

    Returns
    -------
    Q : TYPE
        DESCRIPTION.

    '''
    #-----------------------Inicialización Q(s,a) apha (0,1] y gamma (0,1] ----
    #-----------------------Inicialización ------------------------------------
    #Implementación del seudo código de SARSA Pg. 83
    #Valor que decrece en cada estado 's'
    nA = env.action_space.n         #Número de acciones del entorno
    nS = env.observation_space.n    #Número de observaciones del entorno 
    Q = np.zeros((nS, nA))          #Creo la matriz de nS Filas x nA Columnas
    test_rewards = []               #Almaceno las recompenzas de las pruebas
    games_rewards = []              #Almaceno las recompenzas de los juegos 
    episodios = []                  #Almaceno los episodios 
    #-----------------------Inicialización Q(s,a) apha (0,1] y gamma (0,1] ----
    #-----------------------Inicialización ------------------------------------
    
    #Implementación del lazo principal que aprenderá los Q-valores, Q es el 
    #valor que toma la acción a en un estado s
    for ep in range(num_episodes):  #ep--> episodios 
        state = env.reset()         #Reset de cada env en cada nuevo s st = env_start()
        #print(state)
        done = False                #Cambiara a True cuando el env este s final 
        tot_rew = 0                 #Inicializo la recompenza en 0
        
        if eps > 0.01:              #Actualizar eps hasta que eps sea > 0.01
            eps -= eps_decay
        
        action = eps_greedy(Q,state,eps) #Action basada en state= estado actual de Q-matrix(nS=6,nA=500)
        
        #Lazo hasta que los episodios S del juego terminen hasta que St no sea el último estado 
        while not done: #El agente sige está aún interactuando con el env, 
            next_state, rew, done, _ = env.step(action)  #Toma un step del env         
            next_action = eps_greedy(Q, next_state, eps)
            Q[state][action] = Q[state][action] + lr*(rew + 
                                                      gamma*Q[next_state][next_action]-
                                                      Q[state][action]) #Eq 4.5  pg.82
            state = next_state
            action = next_action
            tot_rew += rew
            if done:
                games_rewards.append(tot_rew)
        
        # Comporbar la policy cada 300 episodios epoch e imprimir los resultado
        if(ep % 200) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episodio:{:5d} Eps: {:2.4f} Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            episodios.append(ep)
    plt.plot(episodios, test_rewards, color="orange")
    plt.ylabel('Media de las Recompenzas Q_SARSA', )
    plt.xlabel('Episodios')
    plt.show()
    return Q
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def Q_learning(env, lr=0.001, num_episodes=10, eps=0.3, gamma=0.95, eps_decay=0.00005):
    '''
    La idea de Q-learning es aproximar la función q-function usando el optimal
    action value. Q-learning es muy similar a SARSA, solo cambia que 
    Q-learning toma el maximo state-action value.En nuestro caso con np.max.
    
    Dieferencias:
        SARSA: la actualización se realizara en la policy E-greedy. Las dos actions
        at y at+1 vienen de la misma policy
        
        Q-update: La actualizaciuón se realizara en greedy target policy:
            Maximo action value. En este caso at+1 se la elije con base en el 
            máximo valor del state-action Figura: 4.7 pag. 91
            Usa: 
                Un target greedy policy que constantemente mejora
                Un policy de comportamiento E-greedy para interactuar y explorar
                el environment

    '''
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []
    episodios = []
    

    for ep in range(num_episodes):
        #state = env.reset()[0]
        state = env.reset()
        done = False
        tot_rew = 0
        
        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop the main body until the environment stops
        while not done:
            # select an action following the eps-greedy policy
            action = eps_greedy(Q, state, eps)

            next_state, rew, done, _ = env.step(action) # Take one step in the environment

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action] = Q[state][action] + lr*(rew + gamma*np.max(Q[next_state]) - Q[state][action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # Test the policy every 300 episodes and print the results
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            episodios.append(ep)
    plt.plot(episodios, test_rewards, color="blue")
    plt.ylabel('Media de las Recompenzas Q_Learning')
    plt.xlabel('Episodios')
    plt.show()
            
    return Q     
#Función eps_greedy
def eps_greedy(Q, s, eps=0.1):
    '''
    Se la usa para escoger una acción randómica de las que esta permitida con 
    la probabilidad eps. 
    
    Es la forma de explorar las posibles soluciones. Se asegura que la accion
    randomica y la acción deseada greedy action son elejidas asegurando la 
    exploración y explotación del environment.

    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    eps : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    TYPE
        DESCRIPTION. Llama a la política greedy

    '''
    #print(Q)
    if np.random.uniform(0,1) < eps:
        #Se elige una acción randómica
        return np.random.randint(Q.shape[1])
    else:
        #Elijo una accción de la política greedy en el estado actual. 
        return greedy(Q,s)

#-----------------------------------------------------------------------------
#policy greedy
def greedy(Q,s):
    '''
    Política: Elige la acción que maximice la comparación acumulativa del 
    estado actual. NO con la mayor recompenza inmediata. 
    
    Regresa el índice que corresponde al máximo valor de la matriz action-state
    Q(nS,nA)

    '''
    return np.argmax(Q[s])
#-----------------------------------------------------------------------------

#Función run_episodes
def run_episodes(env, Q, num_episodes = 10, to_print = False):
    '''
    Corre algunos episodias para realizar un test de la policy. Para no explorar 
    mientras se prueba. 
    
    Función de programación dinámica DP

    Parameters
    ----------
    env : TYPE
        DESCRIPTION.
    Q : TYPE
        DESCRIPTION.
    num_episodes : TYPE, optional
        DESCRIPTION. The default is 10.
    to_print : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    tot_rew = []
    #state = env.reset()[0]
    state = env.reset()
    for _ in range(num_episodes):
        done = False
        game_rew = 0
        while not done:
            #next_state, rew, done, _, a = env.step(greedy(Q, state))
            next_state, rew, done, _= env.step(greedy(Q, state))
            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew.append(game_rew)
    if to_print:
        print('Mean score: %.3f of %i games!'%(np.mean(tot_rew), num_episodes))
    
    return np.mean(tot_rew)   

#games_rewards = []

 
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    #env = gym.make("Taxi-v2", render_mode="human")
    env = gym.make("Taxi-v3")
    #env.reset()
    Q_sarsa = SARSA(env, 
                    lr=0.1, 
                    num_episodes=5000, 
                    eps=0.4,              #Probabilidad de que la primera acción va a 
              #ser randomica con una probabilidad del 0.4 y va a decreser hasta
              # eps_decay=0.001
                    gamma=0.95, 
                    eps_decay=0.001)
    Q_qlearning = Q_learning(env, 
                              lr=0.1, 
                              num_episodes=5000, 
                              eps=0.4, 
                              gamma=0.999, 
                              eps_decay=0.001)

    
#-----------------------------------------------------------------------------
