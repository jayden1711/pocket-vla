#pragma once
#include <stdint.h>
void encoder_init(void);
// Cumulative x4 quadrature counts since boot.
void encoder_read(int32_t *left, int32_t *right);
