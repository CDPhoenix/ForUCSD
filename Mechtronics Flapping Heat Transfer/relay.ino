int command = 1;
void setup() {
  // put your setup code here, to run once:
  pinMode(13,OUTPUT);
  Serial.begin(9600); 
}

void loop() {
  digitalWrite(13,LOW);
  Serial.print("ON\n");
  delay(100);
  digitalWrite(13,HIGH);
  Serial.print("OFF\n");
  delay(100);
}
