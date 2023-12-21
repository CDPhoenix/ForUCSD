/*
  Average Thermocouple

  Reads a temperature from a thermocouple based
  on the MAX6675 driver and displays it in the default Serial.

  https://github.com/YuriiSalimov/MAX6675_Thermocouple

  Created by Yurii Salimov, May, 2019.
  Released into the public domain.
*/
#include <Thermocouple.h>
#include <MAX6675_Thermocouple.h>

#define SCK_PIN 6
#define CS_PIN 5
#define SO_PIN 4
#define SCK_PIN02 10
#define CS_PIN02 9
#define SO_PIN02 8
//MAX6675_Thermocouple* thermocouple = NULL;
Thermocouple* thermocouple;
Thermocouple* thermocouple02;
// the setup function runs once when you press reset or power the board
long previousTime = 0;
long interval = 20000;
// record/5min
//long interval = 300000;
void setup() {
  Serial.begin(9600);
  thermocouple = new MAX6675_Thermocouple(SCK_PIN, CS_PIN, SO_PIN);
  thermocouple02 = new MAX6675_Thermocouple(SCK_PIN02, CS_PIN02, SO_PIN02);
}

// the loop function runs over and over again forever
void loop() {
  // Reads temperature
  const double celsius = thermocouple->readCelsius();
  const double celsius02 = thermocouple02->readCelsius();
  const double kelvin = thermocouple->readKelvin();
  const double fahrenheit = thermocouple->readFahrenheit();
  const double kelvin02 = thermocouple02->readKelvin();
  const double fahrenheit02 = thermocouple02->readFahrenheit();
  unsigned long currentTime = millis();
  //if(currentTime - previousTime>interval){
  //    previousTime = currentTime;
  //    Serial.print(celsius);
  //    Serial.print(celsius02);
  // }
  // Output of information
  //Serial.print("Temperature: ");
  Serial.print(celsius);
  Serial.print(" \t");
  Serial.println(celsius02);
  //Serial.print(kelvin);
  //Serial.print(" K, ");
  //Serial.print(fahrenheit);
  //Serial.println(" F");
  delay(interval); // optionally, only to delay the output of information in the example.
}
