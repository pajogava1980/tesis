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
import gymnasium as gym
#import gym
import numpy as np
#from matplotlib import pyplot as plt 

#-----------------------------------------------------------------------------
def SARSA(env, 
          lr = 0.01,                #leraning rate--> alfa 
          num_episodes = 10000, 
          eps = 0.3,                #Valor inicial de la politica agreedy
          gamma = 0.95,             #Factor de descuento 
          eps_decay = 0.00005):     #Valor que decrece en cada estado 's'
    nA = env.action_space.n         #Número de acciones del entorno
    nS = env.observation_space.n    #Número de observaciones del entorno 
    Q = np.zeros((nS, nA))          #Creo la matriz de nS Filas x nA Columnas
    test_rewards = []               #Almaceno las recompenzas de las pruebas
    games_rewards = []              #Almaceno las recompenzas de los juegos 
    
    
    #Implementación del lazo principal que aprenderá los Q-valores, Q es el 
    #valor que toma la acción a en un estado s
    for ep in range(num_episodes):  #ep--> episodios 
        #state = env.reset()[0]     #Reset de cada env en cada nuevo s 
        state = env.reset()         #Reset de cada env en cada nuevo s 
        #print(state)
        done = False                #Cambiara a True cuando el env este s final 
        tot_rew = 0
        
        if eps > 0.01:              #Actualizar eps hasta que sea >0.01
            eps -= eps_decay
        
        action = eps_greedy(Q,state,eps) #Action basada en s de Q-matrix(nS=6,nA=500)
        
        #Lazo hasta que los episodios S del juego terminen 
        while not done: #El agente sige está aún interactuando con el env
            #test = []
            #next_state, rew, done, _, a = env.step(action) #Toma 1 s en env
            next_state, rew, done, _, a = env.step(action) #Toma 1 s en env
            next_action = eps_greedy(Q, next_state, eps) #Toma la sigiente accción con base en el siguiente s y Q
            #state [0].np.append(test)
            #Es la límea mas importante del SARSA, aprende la tasa y el gamma
            # y son usasdos para obtener la ganacia en los últimos pasos y 
            # almacenarlos en el arreglo Q
            Q[state][action] = Q[state][action] + lr*(rew + gamma*Q[next_state][next_action]-Q[state][action]) #Eq 4.5
            state = next_state
            action = next_action
            tot_rew += rew
            if done:
                games_rewards.append(tot_rew)
        
        # Comporbar la policy cada 300 episodios epoch e imprimir los resultado
        if(ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episodio:{:5d} Eps: {:2.4f} Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
    return Q
#-----------------------------------------------------------------------------
def eps_greedy(Q, s, eps=0.1):
    '''
    Parameters
    ----------
    Epsilon greedy policy, escoje una acción randómica de las qiue están 
    permitidas con probabilidad eps=0.1
    '''
    #print(Q)
    if np.random.uniform(0,1) < eps:
        #Se elige una acción randómica
        return np.random.randint(Q.shape[1])
    else:
        #Elijo una accción que deseada 
        return greedy(Q,s)
#-----------------------------------------------------------------------------
def greedy(Q,s):
    '''
    Política.
    
    Regresa el índice que corresponde al máximo valor de la matriz action-state
    Q(nS,nA)

    '''
    return np.argmax(Q[s])
#-----------------------------------------------------------------------------
def run_episodes(env, Q, num_episodes = 10, to_print = False):
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

#-----------------------------------------------------------------------------
def Q_learning(env, lr=0.01, num_episodes=10, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []

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

            next_state, rew, done, _, a = env.step(action) # Take one step in the environment

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
            
    return Q     
        
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    env = gym.make("Taxi-v3", render_mode="human")
    #env = gym.make("Taxi-v3")
    env.reset()
    Q = SARSA(env, 
              lr=0.1, 
              num_episodes=5, 
              eps=0.4,              #Probabilidad de que la primera acción va a 
              #ser randomica con una probabilidad del 0.4 y va a decreser hasta
              # eps_decay=0.001
              gamma=0.95, 
              eps_decay=0.001)
    Q_qlearning = Q_learning(env, 
                             lr=.1, 
                             num_episodes=5, 
                             eps=0.4, 
                             gamma=0.95, 
                             eps_decay=0.001)
    
#-----------------------------------------------------------------------------