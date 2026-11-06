#include "encoder.h"

#include "board.h"
#include "driver/pulse_cnt.h"
#include "esp_check.h"

#define PCNT_LIMIT 30000

static pcnt_unit_handle_t units[2];

static pcnt_unit_handle_t make_unit(int pin_a, int pin_b) {
    pcnt_unit_config_t uc = {
        .low_limit = -PCNT_LIMIT,
        .high_limit = PCNT_LIMIT,
        .flags.accum_count = 1,  // extend the 16-bit hardware counter in software
    };
    pcnt_unit_handle_t u;
    ESP_ERROR_CHECK(pcnt_new_unit(&uc, &u));
    pcnt_glitch_filter_config_t fc = {.max_glitch_ns = 1000};
    ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(u, &fc));

    pcnt_chan_config_t ca = {.edge_gpio_num = pin_a, .level_gpio_num = pin_b};
    pcnt_chan_config_t cb = {.edge_gpio_num = pin_b, .level_gpio_num = pin_a};
    pcnt_channel_handle_t cha, chb;
    ESP_ERROR_CHECK(pcnt_new_channel(u, &ca, &cha));
    ESP_ERROR_CHECK(pcnt_new_channel(u, &cb, &chb));
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(cha, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(cha, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE));
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(chb, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(chb, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE));
    // accum_count needs watch points at the limits.
    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(u, PCNT_LIMIT));
    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(u, -PCNT_LIMIT));
    ESP_ERROR_CHECK(pcnt_unit_enable(u));
    ESP_ERROR_CHECK(pcnt_unit_clear_count(u));
    ESP_ERROR_CHECK(pcnt_unit_start(u));
    return u;
}

void encoder_init(void) {
#if PIN_ENC_L_A >= 0
    units[0] = make_unit(PIN_ENC_L_A, PIN_ENC_L_B);
    units[1] = make_unit(PIN_ENC_R_A, PIN_ENC_R_B);
#endif
}

void encoder_read(int32_t *left, int32_t *right) {
    int l = 0, r = 0;
    if (units[0]) pcnt_unit_get_count(units[0], &l);
    if (units[1]) pcnt_unit_get_count(units[1], &r);
    *left = INVERT_LEFT ? -l : l;
    *right = INVERT_RIGHT ? -r : r;
}
