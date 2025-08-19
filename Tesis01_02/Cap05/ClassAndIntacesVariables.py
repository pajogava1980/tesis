# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:44:50 2023

@author: pablo.gamboa
"""

class Dog:
    tipo = 'canino'             #Variable compartida para todas las instancias 
    
    def __init__(self, name):
        self.name = name        #Instancia única para cada instancia 
        self.tricks = []
    
    def add_trick(self, trick):
        self.tricks.append(trick)

d = Dog('Fido')
e = Dog('Buddy')
d.add_trick('roll over')
e.add_trick('play dead')
d.tricks
print(d.tipo, e.tipo, d.name, e.name, d.tricks, e.tricks)