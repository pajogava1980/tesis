import matlab.engine

# Iniciar la sesión de MATLAB
eng = matlab.engine.start_matlab()

# Dato que se desea enviar a MATLAB
dato = 42

# Asignar el dato al workspace de MATLAB
eng.workspace['miDato'] = dato

# Mostrar el dato en MATLAB
eng.eval("disp('El valor de miDato es:'); disp(miDato)", nargout=0)

# Cerrar la sesión de MATLAB
eng.quit()