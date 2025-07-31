import socket
import struct
import time
import numpy as np

V_nominal = 13.8e3
V_base_fase = V_nominal/np.sqrt(3)

# Configuración del servidor UDP
ip = "127.0.0.1"
puerto = 9096
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((ip, puerto))
#sock.bind((ip, puerto))
udp_socket.settimeout(5)
#sock.settimeout(5)

print(f"Esperando datos en {ip}:{puerto}...")

while True:
    try:

        #input("Presiona Enter para capturar el siguiente dato...")
        # Leer datos continuamente
        #data, addr = sock.recvfrom(4)  # Leer 4 bytes
        pausa = 0.2
        time.sleep(pausa)
        data, addr = udp_socket.recvfrom(4)  # Leer 4 bytes  # Leer 4 bytes
        print(f"Datos crudos recibidos: {data}, desde {addr}")
        if len(data) != 4:
            print(f"Error: Tamaño incorrecto de datos recibidos: {len(data)} bytes.")
            continue

        #vreg_actual = struct.unpack('<f', data)[0]  # Decodificar float
        vreg_actual = struct.unpack('<f', data)[0]  # Para un double (8 bytes)
        Y_reg_end = round(vreg_actual / V_base_fase, 3)
        print(f"Valor entero recibido (int32): {vreg_actual} y Valor de 'Y_reg_end' en pu: {Y_reg_end}")
        #return Y_reg_end

        # Redondear el valor a 3 dígitos
        #vreg_redondeado = round(vreg_actual, 2)

        #print(f"Valor flotante recibido (Vreg): {vreg_actual} -> Redondeado: {vreg_redondeado}")

    except socket.timeout:
        print("Timeout: No se Rx datos en 5 seg...")

    except Exception as e:
        print(f"Error: {e}")
        break
