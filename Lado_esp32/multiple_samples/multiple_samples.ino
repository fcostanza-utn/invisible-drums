#include "Wire.h"
#include "BluetoothSerial.h"
#include <MPU6050_light.h>
#include <LSM303.h>

String device_name = "ESP32-BT-Slave";

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

#if !defined(CONFIG_BT_SPP_ENABLED)
#error Serial Port Profile for Bluetooth is not available or not enabled. It is only available for the ESP32 chip.
#endif

BluetoothSerial SerialBT;
MPU6050 mpu(Wire);
LSM303 compass;

const int cant_samples = 1;
const int baud_rate = 115200;
unsigned long timer = 0;
const int bufferSize = 100;
float buffer[bufferSize];
int sampleIndex = 0;
int contador = 0;

void setup(){
  Serial.begin(baud_rate);
  SerialBT.begin(device_name);
  SerialBT.deleteAllBondedDevices();
  Serial.printf("The device with name \"%s\" is started.\nNow you can pair it with Bluetooth!\n", device_name.c_str());

  Wire.begin();
  byte status = mpu.begin();
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while(status != 0){ } // Stop if failed to connect
  
  Serial.println(F("Calculating offsets, do not move MPU6050"));
  delay(1000);
  mpu.calcOffsets(true, true);
  Serial.println("Done!\n");

  compass.init();
  compass.enableDefault();
}

void loop() 
{

  if ((millis() - timer) > (100/cant_samples))
  {
    mpu.update();
    compass.read();
    float sensorData[6] = {mpu.getAccX(), mpu.getAccY(), mpu.getAccZ(), mpu.getGyroX(), mpu.getGyroY(), mpu.getGyroZ()};
    for (int i = 0; i < 6; i++) {
        buffer[sampleIndex++] = sensorData[i];
    }

    float compassData[3] = {float(compass.m.x), float(compass.m.y), float(compass.m.z)};
    for (int i = 0; i < 3; i++) {
        buffer[sampleIndex++] = compassData[i];
    }

    contador++;
    timer = millis();
  }

  if (contador == cant_samples)
  {
    String report = "";
    report += String(buffer[0]);
    for (int i = 1; i < (9*cant_samples); i++) {
        report += ",";
        report += String(buffer[i]);
    }
    unsigned long startTime = millis(); // Tiempo antes de enviar
    SerialBT.println(report);
    
    // Esperar un breve momento para permitir que se complete el envío
    delay(5); // Puedes ajustar este tiempo según sea necesario
    
    unsigned long elapsedTime = millis() - startTime; // Tiempo transcurrido

    // Asegúrate de que el tiempo transcurrido no sea cero
    if (elapsedTime > 0) {
      // Calcular la velocidad de transmisión
      size_t messageSize = report.length(); // Tamaño del mensaje en bytes
      float speed = (float)messageSize / (elapsedTime / 1000.0); // Velocidad en bytes por segundo

      // Imprimir la velocidad en el monitor serie  
      Serial.print("Tamaño del mensaje: ");
      Serial.print(messageSize);
      Serial.print(" bytes. Tiempo de envío: ");
      Serial.print(messageSize);
      Serial.print(" ms. Velocidad: ");
      Serial.print(speed);
      Serial.println(" bytes/segundo.");
    } else {
      Serial.println("Error: El tiempo transcurrido es cero.");
    }

    contador = 0;
    sampleIndex = 0;
  }
}