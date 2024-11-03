import utime
from machine import I2C, Pin
from mpu6500 import MPU6500
from lsm303 import LSM303

i2c = I2C(0,scl=Pin(22), sda=Pin(21), freq = 400000)
sensor = MPU6500(i2c)
#mag = LSM303(i2c)

while True:
    #dataMag = mag.read_mag()
    dataAccel = sensor.acceleration
    dataGyro = sensor.gyro
    print("Accel XYZ = " + str(dataAccel[0])+","+str(dataAccel[1])+","+str(dataAccel[2]))
    print("Gyro XYZ = " + str(dataGyro[0])+","+str(dataGyro[1])+","+str(dataGyro[2]))
    #print("Mag XYZ = " + str(dataMag[0])+","+str(dataMag[1])+","+str(dataMag[2]))

    utime.sleep_ms(1000)