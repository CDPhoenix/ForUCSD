#include<dht11.h>
dht11 DHT11;
dht11 DHT1102;
#define DHT11PIN 2
#define DHT1102PIN 12
void setup() {
  // put your setup code here, to run once:
  pinMode(13,OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  int chk = DHT11.read(DHT11PIN);
  int chk2 = DHT1102.read(DHT1102PIN);
  int a = (int)DHT11.temperature;
  int b = (int)DHT1102.temperature;
  int c = (int)DHT11.humidity;
  Serial.print("DHT11 Temp\t");
  Serial.print("DHT11 Humd\t");
  Serial.print("DHT1102 Temp\n");
  Serial.print(a);
  Serial.print('\t');
  Serial.print('\t');
  Serial.print(c);
  Serial.print('\t');
  Serial.print('\t');
  Serial.print(b);
  Serial.print('\n');
  if (a>25 || b>25){
      digitalWrite(13, HIGH);
    }else{
      digitalWrite(13,LOW);
      }
  delay(1000);

}
