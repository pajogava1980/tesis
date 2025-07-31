# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:33:10 2024

@author: pablo.gamboa
"""
import numpy as np
from collections import deque


class ExperienceBuffer():
    
    
    def __init__(self, buffer_size):
        '''
        Metodo que se llama automáticamente cuandose crea un nuevo objeto de
        la clase ExperienceBuffer().
        
        deque: una estructura de cola doble extremo, buffer de experiencia.
        maxlen: el tamaño máximo de la cola sera igual al tamaño del bufer
        de experiencia. 

        Parameters
        ----------
        buffer_size : TYPE
            DESCRIPTION.Tamaño máximo del búfer de experiencia

        Returns
        -------
        None.

        '''
        self.obs_buf = deque(maxlen = buffer_size)
        self.rew_buf = deque(maxlen = buffer_size)
        self.act_buf = deque(maxlen = buffer_size)
        self.obs2_buf = deque(maxlen = buffer_size)
        self.done_buf = deque(maxlen = buffer_size)
    
    def add(self, obs, rew, act, obs2, done):
        '''
        Adicion de una nueva transición al buffer de experiencia

        Parameters/argumentos
        ----------
        obs : TYPE
            DESCRIPTION.observación actual
        rew : TYPE
            DESCRIPTION.Recompensa
        act : TYPE
            DESCRIPTION.Acción tomada
        obs2 : TYPE
            DESCRIPTION. Siguiente observación 
        done : TYPE
            DESCRIPTION.Si el episodia a terminado. 

        Returns
        -------
        None.

        '''
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)
    
    def sample_minibatch(self, batch_size):
        mb_indices = np.random.randint(len(self.obs2_buf), size=batch_size)
        
        mb_obs = scale_frames([self.obs_buf[i] for i in mb_indices])
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = scale_frames([self.obs2_buf[i] for i in mb_indices])
        mb_done = [self.done_buf[i] for i in mb_indices]
        
        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done
    
    def __len__(self):
        return len(self.obs_buf)

def scale_frames(frames):
    '''
    Scale the frame with number between 0 and 1
    '''
    return np.array(frames, dtype=np.float32) / 255.0
        
    