#include "Wire.h"
#include <WiFi.h>  // Incluimos la librería para WiFi
#include <MPU6050_light.h>
#include <LSM303.h>
#include <string.h>
#include <esp_now.h>

#define PIN_GPIO 23  // Definir el pin que queremos leer

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

// Estructura para los datos de los sensores
typedef struct {
  unsigned long tiempo = 0;   // Tiempo de la muestra (milisegundos)
  float ax, ay, az = 0;       // Aceleración
  float gx, gy, gz = 0;       // Velocidad angular
  float mx, my, mz = 0;       // Campo magnético
  int boton = 0;              // Estado del botón
  int receptor = 1;           // Flag de sincronización de receptor
  int emisor = 0;             // Flag de sincronización de emisor
} SensorData;

SensorData sensorDataRemoto;  // Datos recibidos vía ESP‑Now

bool flagDatosRemotos = false;
bool terminate = false;
bool broadcastEnviado = false;

uint8_t mac[6];  // Dirección MAC del Emisor

uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};  // Define la dirección broadcast

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

// Callback que se ejecuta cuando se envían datos vía ESP‑Now
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("Envio ESP‑Now: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Exitoso" : "Fallido");
}

// Callback que se invoca al recibir datos vía ESP‑Now desde el Emisor
void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
  uint8_t *mac_addr = recv_info->src_addr;
  Serial.print("Datos por ESP‑Now recibidos de: ");
  for (int i = 0; i < 6; i++) {
    if (mac_addr[i] < 0x10) {
      Serial.print("0");
    }
    Serial.print(mac_addr[i], HEX);
    if (i < 5) {
      Serial.print(":");
    }
  }
  Serial.println();

  if (len == sizeof(SensorData)) {
    memcpy(mac, mac_addr, 6);
    memcpy(&sensorDataRemoto, data, sizeof(SensorData));
    flagDatosRemotos = true;
  } else {
    Serial.println("Error: tamaño de datos recibido incorrecto");
  }
}

void setup() {
  Serial.begin(baud_rate);
  pinMode(PIN_GPIO, INPUT);   // Configurar el pin como entrada

  // Conectarse a la red WiFi
  Serial.print("Conectando a ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  // Esperar a que se conecte
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Conectando...");
  }

  // Una vez conectado
  Serial.println("Conectado a la red WiFi.");
  Serial.print("Dirección IP: ");
  Serial.println(WiFi.localIP());

  // Inicializar ESP‑Now
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error iniciando ESP‑Now");
    return;
  }
  // Registrar callbacks para envío y recepción
  esp_now_register_send_cb(OnDataSent);
  esp_now_register_recv_cb(OnDataRecv);

  // Configura la información del peer broadcast
  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, broadcastAddress, sizeof(broadcastAddress));
  peerInfo.channel = 0;    // Ajusta el canal según tu configuración
  peerInfo.encrypt = false;

  // Agrega el peer broadcast
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Error al agregar el peer broadcast");
    return;
  }

  // Iniciar el servidor TCP
  server.begin();

  // Inicialización de los sensores
  Wire.begin();
  byte status = mpu.begin(0);
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
  int ref_time = millis();

  // Enviar datos locales vía ESP‑Now (modo broadcast) para definir receptor
  if (!broadcastEnviado) {
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&sensorDataRemoto, sizeof(sensorDataRemoto));
    if (result == ESP_OK) {
      Serial.println("Datos enviados vía ESP‑Now para sincronizacion...");
    } else {
      Serial.print("Error en envío ESP‑Now para sincronizacion: ");
      Serial.println(result);
    }
    broadcastEnviado = true;  // Marca que ya se envió el broadcast
  }

  delay(1000);  // Espera 1 segundo antes de la próxima comprobación de conexión

  while(sensorDataRemoto.receptor == 1){
    if(flagDatosRemotos && sensorDataRemoto.emisor == 0) {
      Serial.print("Enviando datos de sincronizacion a dispositivo con mac: ");
      for (int i = 0; i < 6; i++) {
        if (mac[i] < 0x10) {
          Serial.print("0");
        }
        Serial.print(mac[i], HEX);
        if (i < 5) {
          Serial.print(":");
        }
      }
      Serial.println();
      // Configura la información del peer emisor
      esp_now_peer_info_t peerInfo;
      memset(&peerInfo, 0, sizeof(peerInfo));
      memcpy(peerInfo.peer_addr, mac, sizeof(mac));
      peerInfo.channel = 0;    // Ajusta el canal según tu configuración
      peerInfo.encrypt = false;

      // Agrega el peer broadcast
      if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Error al agregar el peer emisor");
        return;
      }
      sensorDataRemoto.emisor = 1;
      sensorDataRemoto.receptor = 0;
      esp_err_t resultado = esp_now_send(mac, (uint8_t *) &sensorDataRemoto, sizeof(sensorDataRemoto));
      if (resultado == ESP_OK) {
        Serial.println("Datos enviados vía ESP‑Now para definir receptor");
      } else {
        Serial.print("Error en envío ESP‑Now para definir receptor: ");
        Serial.println(resultado);
      }
      sensorDataRemoto.emisor = 0;
      sensorDataRemoto.receptor = 1;
      flagDatosRemotos = false;
      delay(1000);  // Espera 1 segundo antes de la próxima comprobación de conexión
    }
    
    WiFiClient client = server.available();  // Acepta nuevas conexiones    
    // Mantener la conexión abierta
    while (client.connected()) {
      if ((millis() - timer) > 15) {

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

        if (flagDatosRemotos) {
          if(sensorDataRemoto.receptor == 1 && sensorDataRemoto.emisor == 0){
            Serial.print("Enviando datos de sincronizacion a dispositivo con mac: ");
            for (int i = 0; i < 6; i++) {
              if (mac[i] < 0x10) {
                Serial.print("0");
              }
              Serial.print(mac[i], HEX);
              if (i < 5) {
                Serial.print(":");
              }
            }
            Serial.println();
            // Configura la información del peer emisor
            esp_now_peer_info_t peerInfo;
            memset(&peerInfo, 0, sizeof(peerInfo));
            memcpy(peerInfo.peer_addr, mac, sizeof(mac));
            peerInfo.channel = 0;    // Ajusta el canal según tu configuración
            peerInfo.encrypt = false;

            // Agrega el peer broadcast
            if (esp_now_add_peer(&peerInfo) != ESP_OK) {
              Serial.println("Error al agregar el peer emisor");
              return;
            }
            sensorDataRemoto.emisor = 1;
            sensorDataRemoto.receptor = 0;
            esp_err_t resultado = esp_now_send(mac, (uint8_t *) &sensorDataRemoto, sizeof(sensorDataRemoto));
            if (resultado == ESP_OK) {
              Serial.println("Datos enviados vía ESP‑Now para definir receptor");
            } else {
              Serial.print("Error en envío ESP‑Now para definir receptor: ");
              Serial.println(resultado);
            }
            sensorDataRemoto.emisor = 0;
            sensorDataRemoto.receptor = 1;
          }
          else{
            report += ",";
            report += String(sensorDataRemoto.ax);
            report += ",";
            report += String(sensorDataRemoto.ay);
            report += ",";
            report += String(sensorDataRemoto.az);
            report += ",";
            report += String(sensorDataRemoto.gx);
            report += ",";
            report += String(sensorDataRemoto.gy);
            report += ",";
            report += String(sensorDataRemoto.gz);
            report += ",";
            report += String(sensorDataRemoto.mx);
            report += ",";
            report += String(sensorDataRemoto.my);
            report += ",";
            report += String(sensorDataRemoto.mz);
            report += ",";
            report += String(sensorDataRemoto.boton);
            report += ",";
            report += String("2");
          }
          flagDatosRemotos = false;
        }
        else {
          report += ",";
          report += String("4");
        }
        int estado = digitalRead(PIN_GPIO);  // Leer el estado del pin
        if(estado == 0){
          mpu.calcOffsets(true, true);
        }
        int milliseconds = millis() - ref_time;
        report += ",";
        report += String(milliseconds);
        report += ",";
        report += String(estado);

        // Enviar los datos a través del cliente TCP
        client.print(report + "\n");                       // Asegúrate de agregar un salto de línea al final

        sampleIndex = 0;

        timer = millis();
      }
    }
    // El cliente ha cerrado la conexión
    client.stop();  // Cerrar la conexión con el cliente
    Serial.println("Cliente desconectado.");
    delay(250);  // Espera 1 segundo antes de la próxima comprobación de conexión
  }

  while (sensorDataRemoto.emisor == 1){
    Serial.println("Este es el emisor...");
    sensorDataRemoto.receptor = 1;

    // Configura la información del peer receptor
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    memcpy(peerInfo.peer_addr, mac, sizeof(mac));
    peerInfo.channel = 0;    // Ajusta el canal según tu configuración
    peerInfo.encrypt = false;

    // Agrega el peer broadcast
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
      Serial.println("Error al agregar el peer receptor");
      return;
    }

    while(!terminate){
      if ((millis() - timer) > 10) {
        mpu.update();
        compass.read();
        sensorDataRemoto.mx = (compass.m.x - offset_x) * scale_x;
        sensorDataRemoto.my = (compass.m.y - offset_y) * scale_y;
        sensorDataRemoto.mz = (compass.m.z - offset_z) * scale_z;
        sensorDataRemoto.ax = mpu.getAccX();
        sensorDataRemoto.ay = mpu.getAccY();
        sensorDataRemoto.az = mpu.getAccZ();
        sensorDataRemoto.gx = mpu.getGyroX();
        sensorDataRemoto.gy = mpu.getGyroY();
        sensorDataRemoto.gz = mpu.getGyroZ();
    
        sensorDataRemoto.tiempo = millis() - ref_time;
        sensorDataRemoto.boton = digitalRead(PIN_GPIO);  // Leer el estado del pin

        esp_err_t resultado = esp_now_send(mac, (uint8_t *) &sensorDataRemoto, sizeof(sensorDataRemoto));
        if (resultado == ESP_OK) {
          Serial.println("Datos enviados vía ESP‑Now");
        } else {
          terminate = true;
          Serial.print("Error en envío ESP‑Now desde el emisor: ");
          Serial.println(resultado);
        }
        timer = millis();
      }
    }
  }
}