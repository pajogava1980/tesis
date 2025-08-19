# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:33:00 2023

@author: pablo.gamboa
Los nombres con prefijo _spam no son públicas 
"""
class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)
        
    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)
            
    __update = update   # Copia privada del metodo original de update()
    
class MappigSubclass(Mapping):
    def update(self, keys, values):
        for item in zip(keys, values):
            self.items_list.append(item)
