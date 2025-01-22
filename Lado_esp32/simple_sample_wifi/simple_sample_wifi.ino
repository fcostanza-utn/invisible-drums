#include "Wire.h"
#include <WiFi.h>  // Incluimos la librería para WiFi
#include <MPU6050_light.h>
#include <LSM303.h>
#include <string.h>

// Configuración de WiFi
// const char* ssid = "TomNet";            // Reemplaza con el nombre de tu red WiFi
// const char* password = "nonealone681";  // Reemplaza con la contraseña de tu red
// const char* ssid = "ANANOMUERDE 2.4GHz";
// const char* password = "Dulcinea01";
const char* ssid = "Personal-Fran-2.4G";
const char* password = "Fran270894$";

WiFiServer server(80);  // Puerto 80 para el servidor TCP

MPU6050 mpu(Wire);
LSM303 compass;

const int baud_rate = 115200;
unsigned long timer = 0;
const int bufferSize = 10;
float buffer[bufferSize];
int sampleIndex = 0;
int inicio = 0;
int fin = 0;

float corrected_x = 0;
float corrected_y = 0;
float corrected_z = 0;
float offset_x, offset_y, offset_z;
float avg_delta_x, avg_delta_y, avg_delta_z;
float avg_delta;
float scale_x, scale_y, scale_z;


void setup() {
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
  byte status = mpu.begin(0,3);
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while (status != 0) {}  // Detener si no se puede conectar

  //Calibración de MPU6050, Dejar quieto en una posicion
  Serial.println(F("Calculando offsets, no mover el MPU6050"));
  delay(1000);
  mpu.calcOffsets(true, true);  // gyro and accelero

  compass.init();
  compass.enableDefault();
  //Valores previamente obtenidos para la correccion
  compass.m_min = (LSM303::vector<int16_t>){ -336, -440, -224 };
  compass.m_max = (LSM303::vector<int16_t>){ +300, +319, +277 };
  //Calculo de offset y scale para magnetometro
  offset_x = (compass.m_max.x + compass.m_min.x) / 2;
  offset_y = (compass.m_max.y + compass.m_min.y) / 2;
  offset_z = (compass.m_max.z + compass.m_min.z) / 2;

  avg_delta_x = (compass.m_max.x - compass.m_min.x) / 2;
  avg_delta_y = (compass.m_max.y - compass.m_min.y) / 2;
  avg_delta_z = (compass.m_max.z - compass.m_min.z) / 2;

  avg_delta = (avg_delta_x + avg_delta_y + avg_delta_z) / 3;

  scale_x = avg_delta / avg_delta_x;
  scale_y = avg_delta / avg_delta_y;
  scale_z = avg_delta / avg_delta_z;
}

void loop() {
  // Esperar una conexión
  WiFiClient client = server.available();  // Acepta nuevas conexiones
  if (client) {
    Serial.println("Cliente conectado.");

    int ref_time = millis();

    // Mantener la conexión abierta
    while (client.connected()) {
      if ((millis() - timer) > 15) {
        // inicio = millis();

        mpu.update();
        compass.read();
        corrected_x = (compass.m.x - offset_x) * scale_x;
        corrected_y = (compass.m.y - offset_y) * scale_y;
        corrected_z = (compass.m.z - offset_z) * scale_z;

        float sensorData[6] = { mpu.getAccX(), mpu.getAccY(), mpu.getAccZ(), mpu.getGyroX(), mpu.getGyroY(), mpu.getGyroZ() };
        for (int i = 0; i < 6; i++) {
          buffer[sampleIndex++] = sensorData[i];
        }
        float compassData[3] = { float(corrected_x), float(corrected_y), float(corrected_z) };
        for (int i = 0; i < 3; i++) {
          buffer[sampleIndex++] = compassData[i];
        }
        String report = "";
        report += String(buffer[0]);
        for (int i = 1; i < 9; i++) {
          report += ",";
          report += String(buffer[i]);
        }
        int milliseconds = millis() - ref_time;
        report += ",";
        report += String(milliseconds);
        
        // unsigned long startTime = millis();  // Tiempo antes de enviar

        // Enviar los datos a través del cliente TCP
        client.print(report + "\n");                       // Asegúrate de agregar un salto de línea al final
        // Serial.print("Mensaje: ");
        // Serial.println(report);
        // unsigned long elapsedTime = millis() - startTime;  // Tiempo transcurrido

        // // Asegurarse de que el tiempo transcurrido no sea cero
        // if (elapsedTime > 0) {
        //   size_t messageSize = report.length();                       // Tamaño del mensaje en bytes
        //   float speed = (float)messageSize / (elapsedTime / 1000.0);  // Velocidad en bytes por segundo

        //   // Imprimir la velocidad en el monitor serie
        //   Serial.print("Tamaño del mensaje: ");
        //   Serial.print(messageSize);
        //   Serial.print(" bytes. Tiempo de envío: ");
        //   Serial.print(elapsedTime);
        //   Serial.print(" ms. Velocidad: ");
        //   Serial.print(speed);
        //   Serial.println(" bytes/segundo.");
        // } else {
        //   Serial.println("Error: El tiempo transcurrido es cero.");
        // }

        sampleIndex = 0;
        // fin = millis();
        // Serial.print("Tardó en terminar el ciclo: ");
        // Serial.print(fin - inicio);
        // Serial.println(" milisegundos");
        timer = millis();
      }
    }
    // El cliente ha cerrado la conexión
    client.stop();  // Cerrar la conexión con el cliente
    Serial.println("Cliente desconectado.");
  }
  delay(1000);  // Espera 1 segundo antes de la próxima comprobación de conexión
}