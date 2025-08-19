# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:50:56 2023

@author: pablo.gamboa
NoopResetEnv(n): toma no-ops y reset el env para proveer un inicio random-agente
FireResetEnv(): reset de env 
MaxAndSkipEnv(skip): Salta frames mientras con cuidado de repetir las acciones
y sumar las recompensas. 
WarpFrame(): Cambia el tamaño del frame 84x84-en una escala de grises 
FrameStack(k): SE almacena las k frames

Metodos del WRAPPER gym.Wrapper class:
    self,
    env, 
    step, 
    reset, 
    render, 
    close or seed.
"""
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from datetime import datetime
#import time
#import sys
import os 
from collections import deque 
import gymnasium as gym 
from gym import spaces 
import cv2


#------------------------------------------------------------------------------
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        '''
        Se inica el muestreo de los estados  s tomando un # random
        un inicio randomico de la posición por el agente.
        '''
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        '''
        No realizar las acciones -op  para un número de pasos en [1, noop_max]
        '''
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
            #noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, ac):
        return self.env.step(ac)
#------------------------------------------------------------------------------
class LazyFrames(object):
    def __init__(self, frames):
        
        '''
        Este objeto asegura que los frames comunes entre las observaciones 
        están solo almacenadas una vez.      
        frames : TYPE
           Es para optimizar el uso de la memori  q en DQN puede ser enorme  1M
           frames almacenandose en el buffer.      
           Este objeto puede solo ser convertido a np arrar antes de empesar
           pasar por el modelo
        '''
        self._frames = frames
        self._out = None
      
    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out 
    
    def __len__(self):
        return len(self.force())
    
    def __getitem__(self,i):
        return self.force()[i]
#------------------------------------------------------------------------------
class FireResetEnv(gym.Wrapper):                                #Hereda la clase Wraper del Gym

    def __init__(self, env):                                    # Constructor 
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE' #Verifico FIRE on
        assert len(env.unwrapped.get_action_meanings()) >= 3
        
    def reset(self, **kwargs):                                  # Método de reset 
        self.env.reset(**kwargs)                                # Reseteo la función 
        obs, _, done, _ = self.env.step(1)                      # Hago una acción del fuego 
        if done:
            self.env.reset(**kwargs)
        return obs
    
    def step(self, ac):
        return self.env.step(ac)
#------------------------------------------------------------------------------
class MaxAndSkipEnv(gym.Wrapper):
    '''
    Salta frmaes mientras toma cuidado de repetir la acción y sumar las 
    recompensas
    '''
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, 
                                    dtype=np.uint8)
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info, a = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip -1:
                self._obs_buffer[0] = obs
                
            total_reward += reward
            if done:
                break
            
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs) 
#------------------------------------------------------------------------------
#Hereda las propiedades de gym.ObservationWrapper y crea un Box
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        '''
        84x84 es del tamaño natural del papel y las cartas de trabajo      
        Crea un Box  con los valores entre 0-255 y con una forma de 84x84
        '''
        gym.ObservationWrapper.__init__(self,env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, 
                                            high=255, 
                                            shape=(self.height,
                                                   self.width, 
                                                   1), 
                                            dtype= np.uint8)
        
    def observation(self, frame):
        frame = cv2.cvtColor(frame, 
                             cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, 
                           (self.width, self.height), 
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

#------------------------------------------------------------------------------

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, 
                                            high=255, 
                                            shape=(shp[0], 
                                                   shp[1], 
                                                   shp[2] * k), 
                                            dtype=env.observation_space.dtype)

    def reset(self):
        ob= self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

#------------------------------------------------------------------------------       
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, 
                                                high=1, 
                                                shape=env.observation_space.shape, 
                                                dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
    
#------------------------------------------------------------------------------
#Esta función permite aplicar cada envoltura a un env
def make_env(env_name, fire=True, frames_num=2, noop_num=30, skip_frames=True):
    env = gym.make(env_name)
    
    if skip_frames:
        env = MaxAndSkipEnv(env) # Return only every `skip`-th frame
    if fire:
       env = FireResetEnv(env) ## Fire at the beginning
    env = NoopResetEnv(env, noop_max=noop_num)
    env = WarpFrame(env) ## Reshape image
    env = FrameStack(env, frames_num) ## Stack last 4 frames
    #env = ScaledFloatFrame(env) ## Scale frames
    return env 
        