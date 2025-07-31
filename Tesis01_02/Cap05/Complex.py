# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:22:42 2023

@author: pablo.gamboa

Instances Object: las unicas operaciones entendidas por Instances Objects son:
    Atributos 
        Atrubutes names: 
            Data atributes: son como las variables locales 
            Methods: Una función q pertenece a un objeto:
                append
                insert
                remove
                sort
    Referencias
"""

class Complex:
    def __init__(self, partereal, parteimag):
        self.i = parteimag
        self.r = partereal

x = Complex(3.0, -4.5)
print(x.i, x.r)

x.counter = 1
while x.counter < 10:
    x.counter = x.counter*2
print(x.counter)
del x.counter