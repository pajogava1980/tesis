import socket
import struct

# Configuración del servidor UDP
ip = "127.0.0.1"
puerto = 9091
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, puerto))
sock.settimeout(5)

print(f"Esperando datos en {ip}:{puerto}...")

# Variable para habilitar la captura
capturar = True

while True:
    try:
        if capturar:
            # Leer datos del servidor UDP
            data, addr = sock.recvfrom(4)  # Leer 4 bytes
            vreg_actual = struct.unpack('<f', data)[0]  # Decodificar float

            # Redondear el valor a 3 dígitos
            vreg_redondeado = round(vreg_actual, 3)
            print(f"Valor flotante recibido (Vreg): {vreg_actual} -> Redondeado: {vreg_redondeado}")

            # Cambiar el estado de captura para la siguiente iteración
            capturar = False

        # Aquí puedes agregar tu lógica para habilitar la captura nuevamente
        continuar = input("Presiona 'Enter' para capturar el siguiente dato... (o escribe 'exit' para salir): ")
        if continuar.lower() == 'exit':
            print("Finalizando el script.")
            break
        else:
            capturar = True  # Habilitar la captura para la siguiente iteración

    except socket.timeout:
        print("Timeout: No se recibieron datos en 5 segundos.")
    except Exception as e:
        print(f"Error: {e}")
        break

