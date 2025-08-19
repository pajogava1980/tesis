import matlab.engine
import numpy as np
import random
import time
#import numpy
#from scipy.io import savemat

# Inicio el motor de Matlab
eng = matlab.engine.start_matlab()
eng = matlab.engine.connect_matlab()

eng.eval("clear", nargout=0)
eng.eval("clc", nargout=0)

# Creo las los parametros de la red AC
eng.workspace['VLLrms'] = 13.8e3
eng.workspace['VLNrms'] = eng.workspace['VLLrms']/np.sqrt(3)
eng.workspace['VLNpk'] = eng.workspace['VLNrms']/np.sqrt(2)

eng.workspace['f1'] = 60.0
eng.workspace['T1'] = 1/eng.workspace['f1']
eng.workspace['w1'] = 2*np.pi*eng.workspace['f1']

eng.workspace['Rl'] = 10
eng.workspace['Ll'] = 10e-3
eng.workspace['R'] = 10
eng.workspace['C'] = 100e-6

# Parametros de los BREAKER
eng.workspace['Ron'] = 1e-3
eng.workspace['Rsnubber'] = 10e6

# Parametros PLL
eng.gppll(nargout = 0)

# Parametros de simulación
eng.workspace['Ts'] = 50e-6
eng.workspace['Tsim'] = 1.0

# Creo  objeto .mat
eng.mat(nargout = 0)

# Intercambio de datos con Simulink
n = 10
eng.workspace['N'] = 10
eng.workspace['T'] = 0.5
eng.workspace['k'] = 1

vy = np.zeros(eng.workspace['N'])
eng.workspace['Y'] = vy

vt = np.zeros(eng.workspace['N'])
eng.workspace['t'] = vt

eng.workspace['tap'] = 0
clk = False
eng.workspace['clk'] = False

# Corro Simulink
eng.start(nargout = 0)
time.sleep(eng.workspace['T']/2) #La sincronización sistema en tiempo Real. Es encesario considerarlo, OPAL-RT o RTU, es encesario analizarlo 


# Lo que se debería remplazar con DRL

while eng.workspace['k'] <= eng.workspace['N']:
    Vreg = []
    eng.workspace['clk'] = clk
    eng.clk(nargout = 0)
    time.sleep(eng.workspace['T']/2)

    clk = not clk

    if clk == False:
        print('Hola mundo')

    time.sleep(eng.workspace['T']/2)

    if clk == True:
        tap = random.randint(-20, 20) #Valor generado por DNN
        eng.workspace['tap'] = tap
        eng.tap(nargout=0)

    eng.workspace['k'] = eng.workspace['k'] + 1

eng.stop(nargout = 0)

eng.quit()