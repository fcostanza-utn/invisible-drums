//La ESP es la ESP32-WROOM-DA Module
//Instalar librerias LSM303 by Pololu y MPU6050_light by rfetick

#include "Wire.h"
#include "BluetoothSerial.h"
#include <MPU6050_light.h>
#include <LSM303.h>

String device_name = "ESP32-BT-Slave";

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif
// Check Serial Port Profile
#if !defined(CONFIG_BT_SPP_ENABLED)
#error Serial Port Profile for Bluetooth is not available or not enabled. It is only available for the ESP32 chip.
#endif

// Private Objects and Variables
BluetoothSerial SerialBT;
MPU6050 mpu(Wire);
LSM303 compass;

char report[40];
unsigned long timer = 0;

void setup(){
  Serial.begin(115200);
  SerialBT.begin(device_name);  //Bluetooth device name
  SerialBT.deleteAllBondedDevices(); // Uncomment this to delete paired devices; Must be called after begin
  Serial.printf("The device with name \"%s\" is started.\nNow you can pair it with Bluetooth!\n", device_name.c_str());

  Wire.begin();
  byte status = mpu.begin();
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while(status!=0){ } // stop everything if could not connect to MPU6050
  
  Serial.println(F("Calculating offsets, do not move MPU6050"));
  delay(1000);
  mpu.calcOffsets(true,true); // gyro and accelero
  Serial.println("Done!\n");

  compass.init();
  compass.enableDefault();

}

void loop() 
{
  if( (millis()-timer) > 100 )
  {
    mpu.update();
    compass.read();
    snprintf(report, sizeof(report), "%d, %d, %d",compass.m.x, compass.m.y, compass.m.z);
    Serial.print(report);
    Serial.print(F("ACCELERO  X: "));
    Serial.print(mpu.getAccX());
    Serial.print("\tY: ");
    Serial.print(mpu.getAccY());
    Serial.print("\tZ: ");
    Serial.println(mpu.getAccZ());
  
    Serial.print(F("GYRO      X: "));
    Serial.print(mpu.getGyroX());    
    Serial.print("\tY: ");
    Serial.print(mpu.getGyroY());
    Serial.print("\tZ: ");
    Serial.println(mpu.getGyroZ());
    
    // Sending Data by bluetooth as CSV
    SerialBT.print(mpu.getAccX());
    SerialBT.print(",");
    SerialBT.print(mpu.getAccY());
    SerialBT.print(",");
    SerialBT.print(mpu.getAccZ());
    SerialBT.print(",");
    SerialBT.print(mpu.getGyroX());
    SerialBT.print(",");
    SerialBT.print(mpu.getGyroY());
    SerialBT.print(",");
    SerialBT.print(mpu.getGyroZ());
    SerialBT.print(",");
    SerialBT.print(report);
    SerialBT.print("\n\r");
    timer = millis();
  }
}