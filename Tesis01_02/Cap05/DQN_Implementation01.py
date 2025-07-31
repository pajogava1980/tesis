# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:31:25 2023

@author: pablo.gamboa
Contiene:
    DNN
    Un buffer de experiencias 
    Un grafico comp| mutacional 
    Un lazo de entrenamiento  y evalucación 
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    tf.keras.layers.Conv2D(
    filter: Int, la dimensión del espacio de salida.
    kernel_size: Un Int o Tuple de 2 dimensiones height y width
    strides=(1, 1), pasos de la convolución (height, width)
    padding='valid', Relleno = valid = no relleno, pading = 'same'= padding con
    0s
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None, 
    https://www.tensorflow.org/api_docs/python/tf/keras/activations
    activation = 'relu' = Aplica la función de activación de la unidad lineal 
    rectificada. 
    Ver link para otras activaciones
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
"""


#------------------------------------------------------------------------------ 
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1 
import gym
from datetime import datetime
from collections import deque
import time 
import sys 
from WrapperClass import make_env #Estoy importando desde otro WrapperClass.py


#------------------------------------------------------------------------------   
gym.logger.set_level(40)
current_milli_time = lambda: int(round(time.time() * 1000)) #Función  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def cnn(x): #Función para definir las 3 capas convolutional 
    '''
    Convolución: operación matemática que combina dos funciones para describir
    la superposición entre ambas. 
    
    Convolutional neural network, se define las 3 primeras layers:
        CNN
        FNN
        Salidas de cada acción 
    '''

    x = tf1.keras.layers.Conv2D(filters=16, 
                                kernel_size=8, 
                                strides=4,     #Primera capa
                                padding='valid',
                                activation='relu')(x)
   
    x = tf1.keras.layers.Conv2D(filters=32, 
                                kernel_size=4, 
                                strides=2,     #Segunda capa
                                padding='valid', 
                                activation='relu')(x)
    return tf1.keras.layers.Conv2D(filters=32, 
                                   kernel_size=3, 
                                   strides=1,  #Tercera capa
                                   padding='valid', 
                                   activation='relu')(x)

#------------------------------------------------------------------------------
def fnn(x,                                      #Dense layer 
        hidden_layers, #La lista de valores integers, es =[128]
        output_layer, #El número de agentes de acción 
        activation=tf1.nn.relu, 
        last_activation=None):
    '''
    Feed-forward neural network, defino las 2 últimas layers 
    '''
    for l in hidden_layers: #valores integers, es =[128] Cuarta capa densa
        x = tf1.keras.layers.Dense(x, 
                             units=l, 
                             activation=activation)
    return tf1.keras.layers.Dense(x, 
                                 units=output_layer,
                                 activation=last_activation)
#------------------------------------------------------------------------------
'''
qnet Conectan las layers CNN y FNN con una layer que se llana (flattens) de 
2D de salida de CNN

SE FINALIZA DE DEFINIR LA DNN, todo lo que se requiere conectar en el gráfico
computacional principal.

'''
def qnet(x, 
         hidden_layers,   #La lista de valores integers, es =[128]
         output_size, 
         fnn_activation = tf1.nn.relu, 
         last_activation = None):
    '''
    Deep Q network: CNN followed by FNN, Une CNN y FNN
    '''
    x = cnn(x)
    x = tf1.keras.layers.Flatten(data_format='channels_last')(x)
    #x = tf1.layers.flatten(x)

    return fnn(x, 
               hidden_layers, 
               output_size, 
               fnn_activation, 
               last_activation)
#------------------------------------------------------------------------------
'''
El buffer de experiencia de tipo y almacenamiento ciclico tipo FIFO para cada
uno de los siguientes componentes:
    observation
    reward
    action 
    next observation
    done
    
FIFO= especificado con maxlen
La capacidad del buffer= buffer_size:
'''
#------------------------------------------------------------------------------
class ExperienceBuffer():#Clase usada para gestionar el muestreo de mini-batches
    '''
    Experience Replay Buffer
    maxlen = Máxima capacidad especificada
    buffer_size = la capacidad 
    mini-batches o mini-lotes son usados para entrenar la NN
    '''
    def __init__(self, buffer_size):                #Metodo constructor 
        self.obs_buf = deque(maxlen=buffer_size)    #Propiedad Observaciones
        self.rew_buf = deque(maxlen=buffer_size)    #Propiedad Reward  
        self.act_buf = deque(maxlen=buffer_size)    #Propiedad Action
        self.obs2_buf = deque(maxlen=buffer_size)   #Propiedad Next observation
        self.done_buf = deque(maxlen=buffer_size)   #Propiedad Done
#------------------------------------------------------------------------------
    def add(self, obs, rew, act, obs2, done):       #Metodo add
        # Add a new transition to the buffers
        self.obs_buf.append(obs)                    #Propiedad Observaciones
        self.rew_buf.append(rew)                    #Propiedad Reward
        self.act_buf.append(act)                    #Propiedad Action
        self.obs2_buf.append(obs2)                  #Propiedad Next observation
        self.done_buf.append(done)                  #Propiedad Done     
#------------------------------------------------------------------------------  
    def sample_minibatch(self, batch_size):         #Metodo sample_minibatch
        '''
            DESCRIPTION. Gestiona los saltos de los minibatchs, que son usados
            para entrenar la red neuronal NN. Hay muestras uniformes desde el 
            buffer y se tiene un tamaño predefinido batch_size
        '''
        #Se hace unos pequeños lotes de obs, rew, act, obs2, done
        mb_indices = np.random.randint(len(self.obs_buf), 
                                       size=batch_size)#Tamaño predefinido 

        mb_obs = scale_frames([self.obs_buf[i] 
                               for i in mb_indices])
        mb_rew = [self.rew_buf[i] 
                  for i in mb_indices]
        mb_act = [self.act_buf[i] 
                  for i in mb_indices]
        mb_obs2 = scale_frames([self.obs2_buf[i] 
                                for i in mb_indices])
        mb_done = [self.done_buf[i] 
                   for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done
    
# Se sobreescribe el método _len del buffer, solo se retrona un buffer pq 
# los buffer son de = tamaño
    def __len__(self):  
        return len(self.obs_buf)

   
#------------------------------------------------------------------------------
def q_target_values(mini_batch_rw, mini_batch_done, av, discounted_value):   
    '''
    Calculate the target value y for each transition
    '''
    max_av = np.max(av, axis=1)
    
    # if episode terminate, y take value r
    # otherwise, q-learning step
    
    ys = []
    for r, d, av in zip(mini_batch_rw, mini_batch_done, max_av):
        if d:
            ys.append(r)
        else:
            q_step = r + discounted_value * av
            ys.append(q_step)
    
    assert len(ys) == len(mini_batch_rw)
    return ys
#------------------------------------------------------------------------------
def greedy(action_values):
    '''
    Greedy policy
    '''
    return np.argmax(action_values)

#------------------------------------------------------------------------------
def eps_greedy(action_values, eps=0.1):
    '''
    Eps-greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a uniform random action
        return np.random.randint(len(action_values))
    else:
        # Choose the greedy action
        return np.argmax(action_values)
    
#------------------------------------------------------------------------------
def test_agent(env_test, agent_op, num_games=20):
    '''
    Test an agent
    '''
    games_r = []

    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            # Use an eps-greedy policy with eps=0.05 (to add stochasticity to the policy)
            # Needed because Atari envs are deterministic
            # If you would use a greedy policy, the results will be always the same
            a = eps_greedy(np.squeeze(agent_op(o)), eps=0.05)
            o, r, d, _ = env_test.step(a)
            game_r += r
        games_r.append(game_r)
    return games_r
#------------------------------------------------------------------------------
def scale_frames(frames):
    '''
    Scale the frame with number between 0 and 1
    '''
    return np.array(frames, dtype=np.float32) / 255.0
#------------------------------------------------------------------------------
def DQN(env_name, 
        hidden_sizes=[32], 
        lr=1e-2, 
        num_epochs=2000, 
        buffer_size=100000, 
        discount=0.99, 
        render_cycle=100, 
        update_target_net=1000, 
        batch_size=64, 
        update_freq=4, 
        frames_num=2, 
        min_buffer_size=5000, 
        test_frequency=20, 
        start_explor=1, 
        end_explor=0.1, 
        explor_steps=100000):
    '''
    El nucleo del algoritomo, función DQN, toma el nombre de los env y todos 
    los hyperparametros como argumentos 

    
    '''

    # Create the environment both for train and test
    env = make_env(env_name,                #Env de entrenamiento creado 
                   frames_num=frames_num, 
                   skip_frames=True,
                   noop_num=20)             #Se almacena cada 20 eventos 
    
    env_test = make_env(env_name,           #Env de pruebas creado 
                        frames_num=frames_num, 
                        skip_frames=True, 
                        noop_num=20)
    # Add a monitor to the test env to store the videos
    env_test = gym.wrappers.Monitor(env_test, 
                                    "VIDEOS/TEST_VIDEOS" +
                                    env_name +
                                    str(current_milli_time()),
                                    force=True, 
                                    video_callable=lambda x: x%20==0)
                                                            #x=20 episodios

    tf1.reset_default_graph()       #reset el TensorFlow graph

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    tf1.disable_eager_execution() #Solucionar error RuntimeError: 
        #tf.placeholder() is not compatible with eager execution

    # Create all the placeholders (punteros): obs=observaciones, 
    #act =actions, Y= target values 
    obs_ph = tf1.placeholder(shape=(None, #Puntero observaciones reutilizable
                                   obs_dim[0], 
                                   obs_dim[1], 
                                   obs_dim[2]), 
                            dtype=tf1.float32, 
                            name='obs')
    act_ph = tf1.placeholder(shape=(None,), #Puntero acciones reutilizable
                            dtype=tf1.int32, 
                            name='act')
    y_ph = tf1.placeholder(shape=(None,),   #Puntero valores objetivo Y reutil
                          dtype=tf1.float32, 
                          name='y')

    # Create the target network llamado en qnet, función utilizada para unir 
    #las capas 
    with tf1.variable_scope('target_network'):
        
    #target_network va a ser actualizado por si mismo y toma los 
    #paramentros de online_network
        target_qv = qnet(obs_ph, 
                         hidden_sizes, 
                         act_dim)
    target_vars = tf1.trainable_variables()

    # Create the online network (i.e. the behavior policy)
    with tf1.variable_scope('online_network'):
        online_qv = qnet(obs_ph, 
                         hidden_sizes, 
                         act_dim)
    train_vars = tf1.trainable_variables()

    # Update the target network by assigning to it the variables of the online network
    # Note that the target network and the online network have the same exact architecture
    #Para realizar la asignación se utiliza assign de tf
    update_target = [train_vars[i].assign(train_vars[i+len(target_vars)]) 
                     for i in range(len(train_vars) - len(target_vars))]
    
    update_target_op = tf1.group(*update_target) #Operación que asigna cada
    # variable de 'online_network' a 'target_network'
    
    '''
    Los Q-values dependen de una acción  aj, pero desde la red online de salida
    un valor para cada acción.
    
    Solo se quiere el Q-value de aj mientras se descartan las otras 
    accion-values
    
    Ej:
        5 acciones aj = 3 [0,0,0,1,0]
        suponemos una salida [3.4, 3.7, 5.4, 2.1]
        Resultado = [0,0,0,5.4,0], la sum = 5.4     
    '''
    # One hot encoding of the action
    act_onehot = tf1.one_hot(act_ph, depth=act_dim)
    # We are interested only in the Q-values of those actions
    q_values = tf1.reduce_sum(act_onehot * online_qv, axis=1)
    # MSE loss function
    v_loss = tf1.reduce_mean((y_ph - q_values)**2)
    # Adam optimize that minimize the loss v_loss
    v_opt = tf1.train.AdamOptimizer(lr).minimize(v_loss)
    
    def agent_op(o):
            '''
            Forward pass to obtain the Q-values from the online network of a 
            single observation
            '''
            # Scale the frames
            o = scale_frames(o)
            return sess.run(online_qv, feed_dict={obs_ph:[o]}) 
    #Tiempo   
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, 
                                      now.hour, 
                                      now.minute, int(now.second))
    print('Time:', clock_time)

    mr_v = tf1.Variable(0.0)
    ml_v = tf1.Variable(0.0)
    
    # TensorBoard summaries 
    tf1.summary.scalar('v_loss', v_loss)
    tf1.summary.scalar('Q-value', tf1.reduce_mean(q_values))
    tf1.summary.histogram('Q-values', q_values)
    
    scalar_summary = tf1.summary.merge_all()
    reward_summary = tf1.summary.scalar('test_rew', mr_v) 
    mean_loss_summary = tf1.summary.scalar('mean_loss', ml_v)
    
    LOG_DIR = 'C:\\Users\pablo.gamboa\Desktop\Test/SRTB_'+env_name
    hyp_str = "-lr_{}-upTN_{}-upF_{}-frms_{}" .format(lr, 
                                                      update_target_net, 
                                                      update_freq, 
                                                      frames_num)
    # initialize the File Writer for writing TensorBoard summaries
    file_writer = tf1.summary.FileWriter(LOG_DIR+
                                         '/DQN_'+
                                         clock_time+
                                         '_'+hyp_str, 
                                         tf1.get_default_graph())
    
    # Apertura de sesión 
    sess = tf1.Session()
    # and initialize all the variables
    sess.run(tf1.global_variables_initializer())
    render_the_game = False
    step_count = 0
    last_update_loss = []
    ep_time = current_milli_time()
    batch_rew = []
    old_step_count = 0

    obs = env.reset()
    # Initialize the experience buffer
    buffer = ExperienceBuffer(buffer_size)
    
    # Copy the online network in the target network
    sess.run(update_target_op)

    ########## EXPLORATION INITIALIZATION ######
    eps = start_explor
    eps_decay = (start_explor - end_explor) / explor_steps
    for ep in range(num_epochs):
        g_rew = 0
        done = False

        # Until the environment does not end..
        while not done:
                
            # Epsilon decay
            if eps > end_explor:
                eps -= eps_decay

            # Choose an eps-greedy action 
            act = eps_greedy(np.squeeze(agent_op(obs)), eps=eps)

            # execute the action in the environment
            obs2, rew, done, _ = env.step(act)

            # Render the game if you want to
            if render_the_game:
                env.render()

            # Add the transition to the replay buffer
            buffer.add(obs, rew, act, obs2, done)

            obs = obs2
            g_rew += rew
            step_count += 1

            ################ TRAINING ###############
            # If it's time to train the network:
            if len(buffer) > min_buffer_size and (step_count % update_freq == 0):
                
                # sample a minibatch from the buffer
                mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)

 
                mb_trg_qv = sess.run(target_qv, feed_dict={obs_ph:mb_obs2})
                y_r = q_target_values(mb_rew, mb_done, mb_trg_qv, discount)

                # TRAINING STEP
                # optimize, compute the loss and return the TB summary
                train_summary, train_loss, _ = sess.run([scalar_summary, v_loss, v_opt], feed_dict={obs_ph:mb_obs, y_ph:y_r, act_ph: mb_act})

                # Add the train summary to the file_writer
                file_writer.add_summary(train_summary, step_count)
                last_update_loss.append(train_loss)

            # Every update_target_net steps, update the target network
            if (len(buffer) > min_buffer_size) and (step_count % update_target_net == 0):

                # run the session to update the target network and get the mean loss sumamry 
                _, train_summary = sess.run([update_target_op, mean_loss_summary], feed_dict={ml_v:np.mean(last_update_loss)})
                file_writer.add_summary(train_summary, step_count)
                last_update_loss = []


            # If the environment is ended, reset it and initialize the variables
            if done:
                obs = env.reset()
                batch_rew.append(g_rew)
                g_rew, render_the_game = 0, False

        # every test_frequency episodes, test the agent and write some stats in TensorBoard
        if ep % test_frequency == 0:
            # Test the agent to 10 games
            test_rw = test_agent(env_test, agent_op, num_games=10)

            # Run the test stats and add them to the file_writer
            test_summary = sess.run(reward_summary, feed_dict={mr_v: np.mean(test_rw)})
            file_writer.add_summary(test_summary, step_count)

            # Print some useful stats
            ep_sec_time = int((current_milli_time()-ep_time) / 1000)
            print('Ep:%4d Rew:%4.2f, Eps:%2.2f -- Step:%5d -- Test:%4.2f %4.2f -- Time:%d -- Ep_Steps:%d' %
                        (ep,
                         np.mean(batch_rew), 
                         eps,
                         step_count, 
                         np.mean(test_rw), 
                         np.std(test_rw), 
                         ep_sec_time, 
                         (step_count-old_step_count)/test_frequency))

            ep_time = current_milli_time()
            batch_rew = []
            old_step_count = step_count
                            
        if ep % render_cycle == 0:
            render_the_game = True

    file_writer.close()
    env.close()
   
#------------------------------------------------------------------------------  
if __name__ == '__main__':

    DQN('PongNoFrameskip-v4', 
        hidden_sizes=[128], 
        lr=2e-4,
        buffer_size=100000,
        update_target_net=1000, 
        batch_size=32, 
        update_freq=2, 
        frames_num=2, 
        min_buffer_size=10000, 
        render_cycle=10000)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------   


