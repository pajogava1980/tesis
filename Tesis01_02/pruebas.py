import struct

# Ejemplo: Bytes en formato hexadecimal
hex_value = b'\x71\xcf\xdf\x45'  # Valor en formato bytes (big-endian)

# Decodificar el valor single (IEEE 754) a decimal
#decimal_value = struct.unpack('>f', hex_value)[0]  # '>f' para big-endian, '<f' para little-endian
vreg_actual = struct.unpack('<f', hex_value)[0]
#print(f"El valor decimal es: {decimal_value}")
print(f"El valor decimal es: {vreg_actual}")

