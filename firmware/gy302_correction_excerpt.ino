// Excerpt of the original Arduino firmware (Carvalho Neto, 2021, dissertation appendix),
// showing only the GY-302 / BH1750 reading and the calibration correction (Eq. 9).
// Reproduced for transparency. The full controller code (fuzzy logic, dimmer, serial
// link to MATLAB) is in the dissertation appendix.
//
// NOTE 1: `int lux2 = 0.7777*lux - 0.4562;` converts float -> int by TRUNCATION
//         toward zero, so on average the corrected value is about 0.5 lx below the
//         unrounded Eq. (9) result. A round-to-nearest version would be:
//             int lux2 = (int) lroundf(0.7777f * lux - 0.4562f);
// NOTE 2: The moving-average buffer (N samples) is computed, but `lux2` uses the raw
//         `lux`. Tables 5 and 6 show identical values with and without the filter.

#include <Wire.h>
#include <BH1750.h>

#define N 10
int loop_read[N];
int media;
BH1750 lightMeter(0x23);

void setup() {
  Serial.begin(9600);
  Wire.begin();
  lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
}

void loop() {
  uint16_t lux = lightMeter.readLightLevel();

  // moving average (computed, not used for lux2)
  float read_res = lux;
  for (int i = N - 1; i > 0; i--) loop_read[i] = loop_read[i - 1];
  loop_read[0] = read_res;
  long soma = 0;
  for (int i = 0; i < N; i++) soma = soma + loop_read[i];
  media = soma / N;

  int lux2 = 0.7777 * lux - 0.4562;  // Eq. (9), obtained in Microsoft Excel (Stage 1)

  Serial.println(lux2);
  delay(200);
}
