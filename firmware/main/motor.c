#include "motor.h"

#include <stdlib.h>

#include "board.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_check.h"

#define DUTY_MAX ((1u << PWM_RES_BITS) - 1u)

static uint32_t scale(int v) { return (uint32_t)abs(v) * DUTY_MAX / 255u; }

static void pwm_channel(int pin, ledc_channel_t ch) {
    ledc_channel_config_t c = {
        .gpio_num = pin,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = ch,
        .timer_sel = LEDC_TIMER_0,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&c));
}

static void set_duty(ledc_channel_t ch, uint32_t duty) {
    ledc_set_duty(LEDC_LOW_SPEED_MODE, ch, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, ch);
}

static void out_pin(int pin) {
    gpio_config_t g = {.pin_bit_mask = 1ULL << pin, .mode = GPIO_MODE_OUTPUT};
    ESP_ERROR_CHECK(gpio_config(&g));
    gpio_set_level(pin, 0);
}

void motor_init(void) {
    out_pin(PIN_STBY);  // driver disabled until the first apply
    ledc_timer_config_t t = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = PWM_RES_BITS,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&t));
#if defined(DRIVER_TB6612FNG)
    out_pin(PIN_L_IN1);
    out_pin(PIN_L_IN2);
    out_pin(PIN_R_IN1);
    out_pin(PIN_R_IN2);
    pwm_channel(PIN_L_PWM, LEDC_CHANNEL_0);
    pwm_channel(PIN_R_PWM, LEDC_CHANNEL_1);
#elif defined(DRIVER_DRV8833)
    pwm_channel(PIN_L_A, LEDC_CHANNEL_0);
    pwm_channel(PIN_L_B, LEDC_CHANNEL_1);
    pwm_channel(PIN_R_A, LEDC_CHANNEL_2);
    pwm_channel(PIN_R_B, LEDC_CHANNEL_3);
#endif
}

#if defined(DRIVER_TB6612FNG)
static void drive(int in1, int in2, ledc_channel_t ch, int v) {
    // v == 0: IN1=IN2=0 (coast) and duty 0.
    gpio_set_level(in1, v > 0);
    gpio_set_level(in2, v < 0);
    set_duty(ch, scale(v));
}
#elif defined(DRIVER_DRV8833)
static void drive(ledc_channel_t a, ledc_channel_t b, int v) {
    // Slow decay: one input held high, PWM the other inverted. v == 0: both low (coast).
    if (v == 0) {
        set_duty(a, 0);
        set_duty(b, 0);
    } else if (v > 0) {
        set_duty(a, DUTY_MAX);
        set_duty(b, DUTY_MAX - scale(v));
    } else {
        set_duty(a, DUTY_MAX - scale(v));
        set_duty(b, DUTY_MAX);
    }
}
#endif

void motor_apply(int left, int right) {
    if (INVERT_LEFT) left = -left;
    if (INVERT_RIGHT) right = -right;
    // Driver enabled only while something is commanded.
    gpio_set_level(PIN_STBY, (left != 0 || right != 0));
#if defined(DRIVER_TB6612FNG)
    drive(PIN_L_IN1, PIN_L_IN2, LEDC_CHANNEL_0, left);
    drive(PIN_R_IN1, PIN_R_IN2, LEDC_CHANNEL_1, right);
#elif defined(DRIVER_DRV8833)
    drive(LEDC_CHANNEL_0, LEDC_CHANNEL_1, left);
    drive(LEDC_CHANNEL_2, LEDC_CHANNEL_3, right);
#endif
}
