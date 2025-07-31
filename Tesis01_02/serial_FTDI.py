import serial
import serial.serialutil
import time
import re

def manejo_xon_xoff(data):
    """
    Maneja los caracteres XON (17) y XOFF (19) recibidos en el flujo de datos.
    """
    XON = 17  # Código ASCII para XON
    XOFF = 19  # Código ASCII para XOFF
    for byte in data:
        if byte == XON:
            print("XON recibido: reanudando transmisión.")
        elif byte == XOFF:
            print("XOFF recibido: pausando transmisión.")

def detectar_protocolo(response):
    """
    Detecta el protocolo basado en la respuesta del dispositivo.
    """
    if ".P2." in response:
        return "Sanxing"
    elif "+VERSION" in response or "OK" in response:
        return "Wassion"
    return None

def identificar_dispositivo(ser):
    """
    Intenta identificar el dispositivo conectado al puerto serial.
    """
    print("Intentando identificar dispositivo...")

    # Prueba comandos Sanxing
    print("Probando comandos para Sanxing...")
    ser.bytesize = serial.SEVENBITS
    ser.parity = serial.PARITY_EVEN
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = True
    trama_sanxing = bytearray([47, 63, 50, 48, 50, 51, 48, 48, 55, 48, 53, 57, 56, 33, 13, 10])
    data_bytes = bytearray(trama_sanxing)
    ser.write(data_bytes)
    time.sleep(0.5)
    
    response1 = ser.read(100)
    response_ascii1 = response1.decode('ascii', errors='replace')
    print(f'respuesta 1: {response_ascii1}')

    response_bytes = ser.read(ser.in_waiting)
    print(f"Respuesta en bruto Sanxing: {response_bytes}")
    response = response_bytes.decode(errors='replace')
    print(f"Respuesta decodificada Sanxing: {response}")
    if ".P2." in response:
        return "Sanxing"

    # Prueba comandos Wassion
    print("Probando comandos para Wassion...")
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = False
    ser.write("AT+VERSION?\r\n".encode())
    time.sleep(0.5)
    response_bytes = ser.read(ser.in_waiting)
    print(f"Respuesta en bruto Wassion: {response_bytes}")
    response = response_bytes.decode(errors='replace')
    print(f"Respuesta decodificada Wassion: {response}")
    if "+VERSION" in response or "OK" in response:
        return "Wassion"

    return None


def ejecutar_comandos_sanxing(ser):
    """
    Ejecuta las configuraciones y comandos específicos para Sanxing.
    """
    # Configuración para Sanxing
    ser.bytesize = serial.SEVENBITS
    ser.parity = serial.PARITY_EVEN
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = True

    # Trama de ejemplo Sanxing
    trama_sanxing = bytearray([47, 63, 50, 48, 50, 51, 48, 48, 55, 48, 53, 57, 56, 33, 13, 10])
    ser.write(trama_sanxing)
    time.sleep(0.5)
    response_sanxing = ser.read(ser.in_waiting).decode()
    print(f"Respuesta Sanxing: {response_sanxing.strip()}\n")

    ser.write(bytearray([6, 48, 54, 49, 13, 10]))
    response2 = ser.read(100)
    response_ascii2 = response2.decode('ascii', errors='replace')
    print(f"Respuesta 2: {response_ascii2}\n")

    match = re.search(r'\((\d+)\)', response_ascii2)
    if match:
        extracted_number = match.group(1)
        print("Número extraído:", extracted_number)
        paramString2 = extracted_number
    else:
        print("No se encontró un número en la respuesta.")
        return

    def b(paramString):
        b = 0
        b1 = 1
        while b1 < len(paramString):
            b ^= ord(paramString[b1])
            b1 += 1
        return b

    def checksum(paramString, target_checksum=101):
        b_value = b(paramString)
        last_char_value = target_checksum ^ b_value
        return chr(last_char_value)

    def generate_result(paramString2):
        paramString1 = "22222222"
        str_result = ""
        for b in range(8):
            b1 = ord(paramString2[7 - b]) - 48
            c = ord(paramString1[b]) - 48
            if b % 2 == 0:
                str_result += chr((c + b1) % 10 + 48)
            else:
                str_result += chr((c + 10 - b1) % 10 + 48)
        return f"({str_result})."

    resultado = generate_result(paramString2)
    resultado1 = checksum(resultado)
    resultado3 = ".P2." + resultado + resultado1
    print(resultado3)
    ascii_values = [ord(char) for char in resultado3]
    ascii_values[0] = 1
    ascii_values[3] = 2
    ascii_values[14] = 3
    print(ascii_values)

    ser.write(bytearray(ascii_values))
    response3 = ser.read(100)
    response_ascii3 = response3.decode('ascii', errors='replace')
    print(f"Respuesta 3: {response_ascii3}\n")

def ejecutar_comandos_wassion(ser):
    """
    Ejecuta las configuraciones y comandos específicos para Wassion.
    """
    # Configuración para Wassion
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = False

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

try:
    # Configuración inicial del puerto serial
    ser = serial.Serial(
        port='COM15',
        baudrate=19200,
        timeout=5
    )

    print(f"Estado del puerto: {ser.is_open}")
    print(f"Configuración inicial del puerto: {ser}\n")

    # Identificar dispositivo
    protocolo = identificar_dispositivo(ser)
    print(f"Dispositivo identificado: {protocolo}\n")

    if protocolo == "Sanxing":
        ejecutar_comandos_sanxing(ser)
    elif protocolo == "Wassion":
        ejecutar_comandos_wassion(ser)
    else:
        print("No se pudo identificar el dispositivo.\n")

except serial.serialutil.SerialException as e:
    print(f"Error al acceder al puerto serial: {e}\n")

except KeyboardInterrupt:
    print("Ejecución interrumpida por el usuario.\n")

finally:
    try:
        ser.close()
        print("Puerto serial cerrado.\n")
    except NameError:
        print("El puerto no fue abierto correctamente.\n")
