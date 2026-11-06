// Platform-independent motor command logic: line protocol, watchdog, slew limit.
// No ESP-IDF includes here so it can be unit-tested on the host (see firmware/test).
//
// Protocol (app -> robot), ASCII lines terminated by '\n' ('\r' ignored):
//   "M <left> <right>"   wheel commands, integers in [-255, 255]. Feeds the watchdog.
//   "S"                  stop. Same as "M 0 0". Feeds the watchdog.
// Anything else (bad syntax, out of range, too long) is rejected, does NOT feed the
// watchdog, and leaves the current targets unchanged.
//
// Safety rules (these live here, on the ESP32, never only in the app):
//   - No valid command for CTL_WATCHDOG_MS  -> both outputs cut to 0 immediately.
//   - BLE disconnect                        -> both outputs cut to 0 immediately.
//   - "M 0 0" / "S"                         -> both outputs cut to 0 immediately.
//   - Any other change is slew limited: |out change| <= CTL_SLEW_PER_TICK per tick.
//   A one-wheel zero (e.g. "M 0 120") is a normal command and is slewed; only a
//   full stop bypasses the slew limit.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef CTL_WATCHDOG_MS
#define CTL_WATCHDOG_MS 300u
#endif
#ifndef CTL_TICK_MS
#define CTL_TICK_MS 10u  // control loop period (100 Hz)
#endif
#ifndef CTL_SLEW_PER_TICK
#define CTL_SLEW_PER_TICK 6  // 0 -> 255 takes ~430 ms at 100 Hz
#endif
#ifndef CTL_MAX_CMD
#define CTL_MAX_CMD 255
#endif
#define CTL_LINE_MAX 32

typedef enum {
    CTL_STATE_IDLE = 0,      // booted / never commanded
    CTL_STATE_RUNNING = 1,   // valid commands arriving
    CTL_STATE_STOPPED = 2,   // explicit stop command
    CTL_STATE_WATCHDOG = 3,  // watchdog tripped
    CTL_STATE_DISCONNECTED = 4,
} ctl_state_t;

typedef struct {
    int16_t target[2];  // last commanded speed, [-255, 255]
    int16_t out[2];     // slew-limited output actually applied to the motors
    uint32_t last_cmd_ms;
    bool connected;
    ctl_state_t state;
    uint32_t n_ok, n_bad, n_watchdog;
    char buf[CTL_LINE_MAX + 1];
    uint8_t len;
    bool overflow;  // current line too long, discard until '\n'
    int16_t max_out;  // runtime output cap (<= CTL_MAX_CMD), for wheels-up / low-speed tests
} ctl_t;

typedef enum { CTL_LINE_OK = 0, CTL_LINE_BAD = 1 } ctl_line_result_t;

void ctl_init(ctl_t *c, uint32_t now_ms);

// Parse one complete line (no '\n'). Applies it if valid.
ctl_line_result_t ctl_handle_line(ctl_t *c, const char *line, uint32_t now_ms);

// Feed raw bytes as they arrive from BLE (lines may be split across writes).
// Returns the number of rejected lines in this chunk.
int ctl_feed(ctl_t *c, const uint8_t *data, size_t n, uint32_t now_ms);

void ctl_set_connected(ctl_t *c, bool connected, uint32_t now_ms);

// Run once per CTL_TICK_MS. Checks the watchdog and advances out[] toward target[].
void ctl_tick(ctl_t *c, uint32_t now_ms);
