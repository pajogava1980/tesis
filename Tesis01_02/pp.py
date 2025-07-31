import serial
import time

def enviar_comando_sanxing():
    """
    Configura el puerto serial y envía un comando básico al medidor Sanxing.
    """
    try:
        # Configuración del puerto serial
        ser = serial.Serial(
            port='COM15',          # Cambia si usas otro puerto
            baudrate=19200,
            bytesize=serial.SEVENBITS,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=True,
            timeout=5
        )

        print(f"Estado del puerto: {ser.is_open}")
        print(f"Configuración inicial del puerto: {ser}")

        # Trama de comando básica
        trama_sanxing = bytearray([47, 63, 50, 48, 50, 51, 48, 48, 55, 48, 53, 57, 56, 33, 13, 10])
        print(f"Enviando comando Sanxing: {trama_sanxing}")
        ser.write(trama_sanxing)

        # Esperar la respuesta
        time.sleep(1)  # Espera más tiempo para recibir la respuesta
        response = ser.read(ser.in_waiting)
        print(f"Respuesta en bruto: {response}")
        print(f"Respuesta decodificada: {response.decode(errors='replace')}")

    except serial.serialutil.SerialException as e:
        print(f"Error al acceder al puerto serial: {e}")
    finally:
        try:
            ser.close()
            print("Puerto serial cerrado.")
        except NameError:
            print("El puerto no fue abierto correctamente.")

# Ejecutar prueba
enviar_comando_sanxing()
