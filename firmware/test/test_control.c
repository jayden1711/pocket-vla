// Host unit tests for control.c. Run: make -C firmware/test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../main/control.h"

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                  \
        }                                                                \
    } while (0)

static void feed(ctl_t *c, const char *s, uint32_t t) { ctl_feed(c, (const uint8_t *)s, strlen(s), t); }

static ctl_t fresh(uint32_t t) {
    ctl_t c;
    ctl_init(&c, t);
    ctl_set_connected(&c, true, t);
    return c;
}

static void test_parse(void) {
    ctl_t c = fresh(0);
    CHECK(ctl_handle_line(&c, "M 120 -80", 0) == CTL_LINE_OK);
    CHECK(c.target[0] == 120 && c.target[1] == -80);
    CHECK(ctl_handle_line(&c, "M 255 -255", 0) == CTL_LINE_OK);
    CHECK(ctl_handle_line(&c, "M 256 0", 0) == CTL_LINE_BAD);
    CHECK(ctl_handle_line(&c, "M 10", 0) == CTL_LINE_BAD);
    CHECK(ctl_handle_line(&c, "M 10 20 30", 0) == CTL_LINE_BAD);
    CHECK(ctl_handle_line(&c, "M 1x 2", 0) == CTL_LINE_BAD);
    CHECK(ctl_handle_line(&c, "M10 20", 0) == CTL_LINE_BAD);
    CHECK(ctl_handle_line(&c, "X", 0) == CTL_LINE_BAD);
    CHECK(ctl_handle_line(&c, "", 0) == CTL_LINE_BAD);
    CHECK(ctl_handle_line(&c, "M 99999999999 0", 0) == CTL_LINE_BAD);
    // bad lines leave the target alone
    CHECK(c.target[0] == 255 && c.target[1] == -255);
    CHECK(ctl_handle_line(&c, "S", 0) == CTL_LINE_OK);
    CHECK(c.target[0] == 0 && c.state == CTL_STATE_STOPPED);
}

static void test_split_lines_and_overflow(void) {
    ctl_t c = fresh(0);
    feed(&c, "M 10", 0);
    CHECK(c.target[0] == 0);  // incomplete
    feed(&c, "0 -50\r\nM 5 ", 0);
    CHECK(c.target[0] == 100 && c.target[1] == -50);
    feed(&c, "6\n", 0);
    CHECK(c.target[0] == 5 && c.target[1] == 6);
    int bad = ctl_feed(&c, (const uint8_t *)"M 1 2 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nM 7 8\n", 52, 0);
    CHECK(bad == 1);
    CHECK(c.target[0] == 7 && c.target[1] == 8);
}

static void test_slew(void) {
    ctl_t c = fresh(0);
    feed(&c, "M 255 -255\n", 0);
    uint32_t t = 0;
    int ticks = 0;
    int16_t prev0 = 0;
    while (c.out[0] != 255) {
        t += CTL_TICK_MS;
        feed(&c, "M 255 -255\n", t);
        ctl_tick(&c, t);
        CHECK(abs(c.out[0] - prev0) <= CTL_SLEW_PER_TICK);
        prev0 = c.out[0];
        ticks++;
        if (ticks > 1000) break;
    }
    CHECK(ticks == (255 + CTL_SLEW_PER_TICK - 1) / CTL_SLEW_PER_TICK);
    CHECK(c.out[1] == -255);
    // reversal ramps through zero, never jumps
    for (int i = 0; i < 200; i++) {
        t += CTL_TICK_MS;
        feed(&c, "M -255 255\n", t);
        int16_t before = c.out[0];
        ctl_tick(&c, t);
        CHECK(abs(c.out[0] - before) <= CTL_SLEW_PER_TICK);
    }
    CHECK(c.out[0] == -255 && c.out[1] == 255);
}

static void ramp_up(ctl_t *c, uint32_t *t) {
    for (int i = 0; i < 100; i++) {
        *t += CTL_TICK_MS;
        feed(c, "M 200 200\n", *t);
        ctl_tick(c, *t);
    }
    CHECK(c->out[0] == 200);
}

static void test_stop_not_ramped(void) {
    ctl_t c = fresh(0);
    uint32_t t = 0;
    ramp_up(&c, &t);
    feed(&c, "M 0 0\n", t);
    CHECK(c.out[0] == 0 && c.out[1] == 0);  // immediate, even before the next tick
    ctl_tick(&c, t + CTL_TICK_MS);
    CHECK(c.out[0] == 0);
    ramp_up(&c, &t);
    feed(&c, "S\n", t);
    CHECK(c.out[0] == 0 && c.out[1] == 0);
    // one-wheel zero is an ordinary (slewed) command
    ramp_up(&c, &t);
    feed(&c, "M 0 200\n", t);
    t += CTL_TICK_MS;
    ctl_tick(&c, t);
    CHECK(c.out[0] == 200 - CTL_SLEW_PER_TICK && c.out[1] == 200);
}

static void test_watchdog(void) {
    ctl_t c = fresh(0);
    uint32_t t = 0;
    ramp_up(&c, &t);
    uint32_t last = t;
    // no command for exactly the window: still running
    t = last + CTL_WATCHDOG_MS;
    ctl_tick(&c, t);
    CHECK(c.out[0] == 200 && c.state == CTL_STATE_RUNNING);
    // one tick later: cut to zero immediately, no ramp
    t += CTL_TICK_MS;
    ctl_tick(&c, t);
    CHECK(c.out[0] == 0 && c.out[1] == 0 && c.state == CTL_STATE_WATCHDOG);
    CHECK(c.n_watchdog == 1);
    // garbage does not feed the watchdog or restart motion
    feed(&c, "hello\n", t);
    ctl_tick(&c, t + CTL_TICK_MS);
    CHECK(c.out[0] == 0);
    // a valid command restarts from zero with the slew limit
    feed(&c, "M 200 200\n", t);
    ctl_tick(&c, t + CTL_TICK_MS);
    CHECK(c.out[0] == CTL_SLEW_PER_TICK);
}

static void test_watchdog_wraparound(void) {
    uint32_t t = 0xFFFFFF00u;
    ctl_t c = fresh(t);
    ramp_up(&c, &t);  // crosses the uint32 wrap
    ctl_tick(&c, t + CTL_WATCHDOG_MS);
    CHECK(c.out[0] == 200);
    ctl_tick(&c, t + CTL_WATCHDOG_MS + 1);
    CHECK(c.out[0] == 0 && c.state == CTL_STATE_WATCHDOG);
}

static void test_disconnect(void) {
    ctl_t c = fresh(0);
    uint32_t t = 0;
    ramp_up(&c, &t);
    ctl_set_connected(&c, false, t);
    CHECK(c.out[0] == 0 && c.out[1] == 0 && c.state == CTL_STATE_DISCONNECTED);
    // commands that sneak in while disconnected don't move the motors
    feed(&c, "M 100 100\n", t);
    ctl_tick(&c, t + CTL_TICK_MS);
    CHECK(c.out[0] == 0);
    // reconnect: idle until commanded, partial line from before is dropped
    ctl_set_connected(&c, true, t);
    feed(&c, "M 5", t);
    ctl_set_connected(&c, false, t);
    ctl_set_connected(&c, true, t);
    feed(&c, "0 50\n", t);  // "0 50" alone is invalid
    ctl_tick(&c, t + CTL_TICK_MS);
    CHECK(c.out[0] == 0);
}

static void test_cap(void) {
    ctl_t c = fresh(0);
    c.max_out = 80;
    uint32_t t = 0;
    for (int i = 0; i < 100; i++) {
        t += CTL_TICK_MS;
        feed(&c, "M 200 -200\n", t);
        ctl_tick(&c, t);
    }
    CHECK(c.out[0] == 80 && c.out[1] == -80);
}

int main(void) {
    test_parse();
    test_split_lines_and_overflow();
    test_slew();
    test_stop_not_ramped();
    test_watchdog();
    test_watchdog_wraparound();
    test_disconnect();
    test_cap();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all control tests passed\n");
    return 0;
}
