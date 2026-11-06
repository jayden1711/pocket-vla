#pragma once
void motor_init(void);
// left/right in [-255, 255]; 0 on both disables the driver (STBY/nSLEEP low).
void motor_apply(int left, int right);
