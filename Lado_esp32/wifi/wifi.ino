#include "Wire.h"
#include <WiFi.h>  // Incluimos la librería para WiFi
#include <MPU6050_light.h>
#include <LSM303.h>

// Configuración de WiFi
const char* ssid = "TomNet";          // Reemplaza con el nombre de tu red WiFi
const char* password = "nonealone681";  // Reemplaza con la contraseña de tu red

WiFiServer server(80); // Puerto 80 para el servidor TCP

MPU6050 mpu(Wire);
LSM303 compass;

const int cant_samples = 10;
const int baud_rate = 115200;
unsigned long timer = 0;
const int bufferSize = 100;
float buffer[bufferSize];
int sampleIndex = 0;
int contador = 0;

void setup(){
  Serial.begin(baud_rate);

  // Conectarse a la red WiFi
  Serial.print("Conectando a ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  // Esperar a que se conecte
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Conectando...");
  }

  // Una vez conectado
  Serial.println("Conectado a la red WiFi.");
  Serial.print("Dirección IP: ");
  Serial.println(WiFi.localIP());

  // Iniciar el servidor TCP
  server.begin();

  // Inicialización de los sensores
  Wire.begin();
  byte status = mpu.begin();
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while(status != 0){ } // Detener si no se puede conectar

  Serial.println(F("Calculando offsets, no mover el MPU6050"));
  delay(1000);
  mpu.calcOffsets(true, true); // Calcula los offsets
  Serial.println("¡Hecho!!\n");

  compass.init();
  compass.enableDefault();
}

void loop() 
{
  // Esperar una conexión
  WiFiClient client = server.available(); // Acepta nuevas conexiones
  if (client) {
    Serial.println("Cliente conectado.");
    
    // Mantener la conexión abierta
    while (client.connected()) {
      // Recopilar datos de los sensores
      if ((millis() - timer) > (100 / cant_samples)) {
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

      if (contador == cant_samples) {
        // Construir el mensaje
        String report = "";
        report += String(buffer[0]);
        for (int i = 1; i < (9 * cant_samples); i++) {
          report += ",";
          report += String(buffer[i]);
        }

        unsigned long startTime = millis(); // Tiempo antes de enviar

        // Enviar los datos a través del cliente TCP
        client.print(report + "\n"); // Asegúrate de agregar un salto de línea al final

        unsigned long elapsedTime = millis() - startTime; // Tiempo transcurrido

        // Asegurarse de que el tiempo transcurrido no sea cero
        if (elapsedTime > 0) {
          size_t messageSize = report.length(); // Tamaño del mensaje en bytes
          float speed = (float)messageSize / (elapsedTime / 1000.0); // Velocidad en bytes por segundo

          // Imprimir la velocidad en el monitor serie
          Serial.print("Tamaño del mensaje: ");
          Serial.print(messageSize);
          Serial.print(" bytes. Tiempo de envío: ");
          Serial.print(elapsedTime);
          Serial.print(" ms. Velocidad: ");
          Serial.print(speed);
          Serial.println(" bytes/segundo.");
        } else {
          Serial.println("Error: El tiempo transcurrido es cero.");
        }

        contador = 0;
        sampleIndex = 0;      

      }


      delay((100 / cant_samples / 10)); // Espera antes de la próxima transmisión para no saturar al cliente
    }

    // El cliente ha cerrado la conexión
    client.stop(); // Cerrar la conexión con el cliente
    Serial.println("Cliente desconectado.");
  }

  delay(1000); // Espera 1 segundo antes de la próxima comprobación de conexión
}
