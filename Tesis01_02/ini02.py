import numpy as np
from scipy.io import savemat

pi_value = np.pi
e_value = np.e

#Diccionario
constantes = {
    'pi' : pi_value,
    'e' : e_value
}

#Guardo en
savemat('constants.mat', constantes)