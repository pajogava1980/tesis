
import re

def b(paramString):
    b = 0
    b1 = 1
    while b1 < len(paramString):
        b ^= ord(paramString[b1])
        b1 += 1
    return b

def checksum(paramString, target_checksum=101):
    # Calcula el valor acumulado de b sin el último caracter
    b_value = b(paramString)  # Cálculo sin el último carácter

    # Encuentra el último caracter tal que b_value ^ ord(last_char) == target_checksum
    last_char_value = target_checksum ^ b_value

    # Devuelve el carácter correspondiente a ese valor
    return chr(last_char_value)

def generate_result(paramString2):
    paramString1 = "22222222"  # paramString1 es siempre "22222222"
    str_result = ""

    # Generar la parte numérica
    for b in range(8):
        b1 = ord(paramString2[7 - b]) - 48  # Convertimos el carácter de paramString2 en número
        c = ord(paramString1[b]) - 48  # Convertimos el carácter de paramString1 en número (siempre '2')

        if b % 2 == 0:
            # Si el índice es par, sumamos los números y tomamos el resultado módulo 10
            str_result += chr((c + b1) % 10 + 48)
        else:
            # Si el índice es impar, sumamos c y 10, y luego restamos b1
            str_result += chr((c + 10 - b1) % 10 + 48)


    # Retornar el resultado final con el nuevo checksum
    return f"({str_result})."



import serial

# Configuración del puerto serial
ser = serial.Serial(
        port='COM18',          # Cambia 'COM15' por el puerto correcto
        baudrate=19200,        # Baud rate
        bytesize=serial.SEVENBITS,  # 7 bits de datos
        parity=serial.PARITY_EVEN,  # Paridad par
        stopbits=serial.STOPBITS_ONE,  # 1 bit de parada
        xonxoff=True,          # Habilitar control de flujo XON/XOFF
        timeout=1              # Tiempo de espera para lectura (ajústalo según sea necesario)
)
print(ser)
 #Lista de enteros a enviar
data_to_send = [47, 63, 50, 48, 50, 51, 48, 48, 55, 48, 53, 57, 56, 33, 13, 10]
# Convertimos la lista de enteros a bytes
data_bytes = bytearray(data_to_send)
# Enviamos los datos por el puerto serial
ser.write(data_bytes)# Leemos la respuesta del dispositivo
response1 = ser.read(100)  # Leer hasta 100 bytes (ajusta según el tamaño esperado de la respuesta)
response_ascii1 = response1.decode('ascii', errors='replace')
print(f'respuesta 1: {response_ascii1}')

ser.write(bytearray([6, 48, 54, 49, 13, 10]))
response2 = ser.read(100)  # Leer hasta 100 bytes (ajusta según el tamaño esperado de la respuesta)
response_ascii2 = response2.decode('ascii', errors='replace')
print(f'respuesta 2: {response_ascii2}')

match = re.search(r'\((\d+)\)', response_ascii2)

if match:
    # Extraemos el número
    extracted_number = match.group(1)
    print("Número extraído:", extracted_number)
else:
    print("No se encontró un número en la respuesta.")
#convertimos el código
paramString2=extracted_number
#paramString2='67254563'
#print(paramString2)
resultado = generate_result(paramString2)
resultado1=checksum(resultado)
resultado3='.P2.'+resultado+resultado1
print(resultado3)
ascii_values = [ord(char) for char in resultado3]
ascii_values[0]=1
ascii_values[3]=2
ascii_values[14]=3
print(ascii_values)

ser.write(bytearray(ascii_values))
response3 = ser.read(100)  # Leer hasta 100 bytes (ajusta según el tamaño esperado de la respuesta)
print(response3)
response_ascii3 = response3.decode('ascii', errors='replace')
print(f'respuesta 3: {response_ascii3}')

serie=[1, 82, 49, 2, 48, 48, 48, 49, 48, 48, 48, 48, 54, 48, 48, 49, 48, 48, 70, 70, 48, 50, 40, 41, 3, 103]
ser.write(bytearray(serie))
response4=ser.read(200)
response_ascii4 = response4.decode('ascii', errors='replace')
print(f'respuesta 4: {response_ascii4}')

energia=[1, 82, 49, 2, 48, 48, 48, 51, 48, 49, 48, 48, 48, 50, 48, 56, 48, 48, 70, 70, 48, 50, 40, 41, 3, 105]
ser.write(bytearray(energia))
response5=ser.read(200)
response_ascii5 = response5.decode('ascii', errors='replace')
print(f'Energía: {response_ascii5}')

# Cerramos la conexión serial
ser.close()

# Convertir la respuesta a una lista de enteros
response_integers = list(response2)
# Convertir la respuesta a ASCII
response_ascii = response2.decode('ascii', errors='replace')  # Usamos 'replace' para manejar caracteres no ASCII

# Ejemplo de uso




