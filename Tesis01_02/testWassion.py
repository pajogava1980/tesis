
import serial

# Configuración del puerto serial
ser = serial.Serial(
    port='COM15',          # Cambia esto por el puerto correcto en tu computadora
    baudrate=19200,        # Velocidad de transmisión
    parity=serial.PARITY_NONE,  # Paridad par
    stopbits=serial.STOPBITS_ONE, # 1 bit de parada
    bytesize=serial.EIGHTBITS,    # 7 bits de datos
    timeout=1              # Tiempo de espera para lectura (ajústalo según sea necesario)
)

ascii_values = [65,84,43,65,65,80,61,34,48,50,48,48,48,48 ,49, 48, 48, 56, 51 ,55, 34 ,44 ,57, 50, 50 ,55 ,48 ,48 ,48 ,48 ,48, 44, 55, 44, 48, 44, 48 ,44 ,50 ,48 ,44, 49,13,10]
ser.write(bytearray(ascii_values))
response3 = ser.read(100)  # Leer hasta 100 bytes (ajusta según el tamaño esperado de la respuesta)
print(response3)
response_ascii3 = response3.decode('ascii', errors='replace')
print(f'respuesta 3: {response_ascii3}')

serie=[65, 84, 43, 86, 69, 82, 83, 73, 79, 78, 63, 13, 10]
ser.write(bytearray(serie))
response4=ser.read(200)
response_ascii4 = response4.decode('ascii', errors='replace')
print(f'respuesta 4: {response_ascii4}')

energia=[47,63,48,50,48,48,48,48,49,48,48,56,51,55,33,13,10]
ser.write(bytearray(energia))
response5=ser.read(200)
response_ascii5 = response5.decode('ascii', errors='replace')
print(f'Energía: {response_ascii5}')

energia=[1, 82, 49 ,2, 49, 53 ,49, 46, 56, 46 ,48, 40, 41, 3 ,94]
ser.write(bytearray(energia))
response5=ser.read(200)
response_ascii5 = response5.decode('ascii', errors='replace')
print(f'Energía: {response_ascii5}')

# Cerramos la conexión serial
ser.close()

# Ejemplo de uso




