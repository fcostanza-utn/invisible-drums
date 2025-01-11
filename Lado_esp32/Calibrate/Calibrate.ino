//La ESP es la ESP32-WROOM-DA Module
//Instalar librerias LSM303 by Pololu y MPU6050_light by rfetick
// OBTENGO MAX Y MIN DEL MAGNETOMETRO Y LOS ANOTO

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
LSM303 compass;
LSM303::vector<int16_t> running_min = {32767, 32767, 32767}, running_max = {-32768, -32768, -32768};

char report[80];
unsigned long timer = 0;

void setup(){
  Serial.begin(115200);
  SerialBT.begin(device_name);  //Bluetooth device name
  SerialBT.deleteAllBondedDevices(); // Uncomment this to delete paired devices; Must be called after begin
  Serial.printf("The device with name \"%s\" is started.\nNow you can pair it with Bluetooth!\n", device_name.c_str());

  Wire.begin();  
  compass.init();
  compass.enableDefault();

}

void loop() 
{
  if( (millis()-timer) > 100 )
  {
    compass.read();
    
    running_min.x = min(running_min.x, compass.m.x);
    running_min.y = min(running_min.y, compass.m.y);
    running_min.z = min(running_min.z, compass.m.z);

    running_max.x = max(running_max.x, compass.m.x);
    running_max.y = max(running_max.y, compass.m.y);
    running_max.z = max(running_max.z, compass.m.z);
  
    snprintf(report, sizeof(report), "min: {%+6d, %+6d, %+6d}    max: {%+6d, %+6d, %+6d}",
    running_min.x, running_min.y, running_min.z,
    running_max.x, running_max.y, running_max.z);

    Serial.print(report);
    Serial.print("\n\r");
    timer = millis();
  }
}