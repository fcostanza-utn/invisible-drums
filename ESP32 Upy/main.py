from machine import Pin, I2C
import time

# Dirección del sensor MPU9250
MPU9250_ADDR = 0x68

# Registros para configurar y leer datos del MPU9250
PWR_MGMT_1 = 0x6B
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
MAG_XOUT_L = 0x03
MAG_XOUT_H = 0x04

# Configuración del I2C
i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)

# Función para escribir un byte en un registro del MPU9250
def write_reg(addr, reg, data):
    i2c.writeto_mem(addr, reg, bytearray([data]))


# Función para leer 2 bytes de registro y combinarlos en un valor de 16 bits
def read_reg_16bit(addr, reg):
    data = i2c.readfrom_mem(addr, reg, 2)
    value = (data[0] << 8) | data[1]
    return value


# Configuración inicial del MPU9250
write_reg(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Despierta el MPU9250
write_reg(MPU9250_ADDR, CONFIG, 0x01)  # Configura el MPU9250 en modo de baja potencia
write_reg(
    MPU9250_ADDR, GYRO_CONFIG, 0x18
)  # Configura el rango de giroscopio a ±2000°/s
write_reg(MPU9250_ADDR, ACCEL_CONFIG, 0x18)  # Configura el rango de acelerómetro a ±16g

while True:
    # Leer datos del acelerómetro
    accel_x = read_reg_16bit(MPU9250_ADDR, ACCEL_XOUT_H)
    accel_y = read_reg_16bit(MPU9250_ADDR, ACCEL_XOUT_H + 2)
    accel_z = read_reg_16bit(MPU9250_ADDR, ACCEL_XOUT_H + 4)

    # Leer datos del giroscopio
    gyro_x = read_reg_16bit(MPU9250_ADDR, ACCEL_XOUT_H + 8)
    gyro_y = read_reg_16bit(MPU9250_ADDR, ACCEL_XOUT_H + 10)
    gyro_z = read_reg_16bit(MPU9250_ADDR, ACCEL_XOUT_H + 12)

    # Leer datos del magnetómetro
    mag_x = read_reg_16bit(MPU9250_ADDR, MAG_XOUT_L)
    mag_y = read_reg_16bit(MPU9250_ADDR, MAG_XOUT_L + 2)
    mag_z = read_reg_16bit(MPU9250_ADDR, MAG_XOUT_L + 4)

    # Convertir datos a valores reales (multiplicar por la escala correspondiente)
    accel_scale = 16 / 32768  # ±16g
    accel_x = accel_x * accel_scale
    accel_y = accel_y * accel_scale
    accel_z = accel_z * accel_scale

    gyro_scale = 2000 / 32768  # ±2000°/s
    gyro_x = gyro_x * gyro_scale
    gyro_y = gyro_y * gyro_scale
    gyro_z = gyro_z * gyro_scale

    # Mostrar los valores
    print("Acelerómetro (g): X={}, Y={}, Z={}".format(accel_x, accel_y, accel_z))
    print("Giroscopio (°/s): X={}, Y={}, Z={}".format(gyro_x, gyro_y, gyro_z))
    print("Magnetómetro: X={}, Y={}, Z={}".format(mag_x, mag_y, mag_z))

    time.sleep(1)  # Esperar un segundo antes de leer de nuevo
