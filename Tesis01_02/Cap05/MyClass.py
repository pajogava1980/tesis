# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:56:48 2023

@author: pablo.gamboa
Clase de objetos soporta 2 tipos de operaciones:
Atributos Referencias e instancias 

Atributos referencia: sintax obj.name.

Los nombres de los atributos validos son todos namespaces->mapa de los nombres
de los objetos.

Namespaces: son ceados en diferentes momentos y tienes diferentes tiempos de vida
tiene una lista de nombres y objetos 
X = 3, x= nombre 3= objeto 


"""

class MyClass:
    """
    Una clase simple
    """
    i = 12345
    def f(self):
        return 'Hola Mundo'