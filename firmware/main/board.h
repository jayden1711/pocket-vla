// Pin map and hardware choices. ESP32-S3-DevKitC-1; avoids strapping pins (0, 3, 45, 46),
// USB (19, 20) and the flash/PSRAM pins (26-37).
#pragma once

// Motor driver: define exactly one.
#define DRIVER_TB6612FNG 1
// #define DRIVER_DRV8833 1

#if defined(DRIVER_TB6612FNG)
// TB6612FNG: IN1/IN2 select direction, PWM sets duty, STBY low = outputs disabled.
// Put a 10k pull-down on STBY so the driver is off during reset, boot and brownout.
#define PIN_L_IN1 4
#define PIN_L_IN2 5
#define PIN_L_PWM 6
#define PIN_R_IN1 7
#define PIN_R_IN2 15
#define PIN_R_PWM 16
#define PIN_STBY 17
#elif defined(DRIVER_DRV8833)
// DRV8833: two PWM inputs per motor (slow-decay drive). nSLEEP low = outputs disabled.
#define PIN_L_A 4
#define PIN_L_B 5
#define PIN_R_A 7
#define PIN_R_B 15
#define PIN_STBY 17  // nSLEEP
#endif

// Quadrature encoders (optional; N20 motors with hall encoders). -1 to disable.
#define PIN_ENC_L_A 8
#define PIN_ENC_L_B 9
#define PIN_ENC_R_A 10
#define PIN_ENC_R_B 11

// Swap if a wheel turns the wrong way on the wheels-up test.
#define INVERT_LEFT 0
#define INVERT_RIGHT 1

#define PWM_FREQ_HZ 20000
#define PWM_RES_BITS 10

// Startup output cap (out of 255). Keep low for the first wheels-up test; raise later.
#define BOOT_MAX_OUT 120

#define TELEMETRY_PERIOD_MS 50  // "E ..." reports at 20 Hz
#define BLE_DEVICE_NAME "pocket-vla"
