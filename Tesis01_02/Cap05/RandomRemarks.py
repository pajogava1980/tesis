# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 07:58:14 2023

@author: pablo.gamboa
f, g y h = datos de los atributos de la case C, se refieren a los obj de la 
función 
"""


class WareHouse:
    proposito = 'almacenamiento'
    region = 'este'

w1 = WareHouse()
print(w1.proposito, w1.region)

w2 = WareHouse()
w2.region = 'oeste'
print(w2.proposito, w2.region)
#-----------------------------------------------------------------------------
#No se debe realizar de esta forma 
def f1(self, x, y):
    return min(x, x+y)

class C:
    f = f1 
    def g(self):
        return 'hola mundo'
    
    h = g
#-----------------------------------------------------------------------------
class Bag: 
    def __init__(self):
        self.data = []
    
    def add(self,x):
        self.data.append(x)
    
    def addtwice(self,x):
        self.add(x)
        self.add(x)