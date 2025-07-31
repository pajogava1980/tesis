from pymodbus.client import ModbusSerialClient
import time

# Configuración del cliente Modbus
client = ModbusSerialClient(
    port='COM15',           # Cambia al puerto correcto
    baudrate=19200,
    parity='E',             # Paridad
    stopbits=1,
    bytesize=7,
    timeout=1
)

if client.connect():
    print("Conexión establecida con el medidor.")
    try:
        # Lee registros y captura tráfico en bruto
        print("Enviando solicitud Modbus...")
        client.socket.flushInput()  # Limpia el buffer antes de la lectura
        response = client.read_holding_registers(address=0x0001, count=2, slave="2023007059")
        
        if not response.isError():
            print(f"Datos recibidos: {response.registers}")
        else:
            print(f"Error al leer registros: {response}")
        
        # Captura y muestra el tráfico en bruto
        while True:
            if client.socket.in_waiting > 0:
                raw_data = client.socket.read(client.socket.in_waiting)
                print(f"Datos en bruto recibidos: {raw_data.hex()}")
                time.sleep(1)  # Pausa para evitar un bucle ocupado
    except Exception as e:
        print(f"Excepción durante la lectura: {e}")
    finally:
        client.close()
        print("Conexión cerrada.")
else:
    print("No se pudo conectar al medidor.")
