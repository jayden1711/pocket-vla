#include "control.h"

#include <stdlib.h>
#include <string.h>

static void cut_to_zero(ctl_t *c, ctl_state_t why) {
    c->target[0] = c->target[1] = 0;
    c->out[0] = c->out[1] = 0;
    c->state = why;
}

void ctl_init(ctl_t *c, uint32_t now_ms) {
    memset(c, 0, sizeof(*c));
    c->last_cmd_ms = now_ms;
    c->state = CTL_STATE_IDLE;
    c->max_out = CTL_MAX_CMD;
}

// Strict signed decimal integer parse; advances *p. Returns false on no digits/overflow.
static bool parse_int(const char **p, long *v) {
    const char *s = *p;
    while (*s == ' ') s++;
    bool neg = false;
    if (*s == '+' || *s == '-') neg = (*s++ == '-');
    if (*s < '0' || *s > '9') return false;
    long x = 0;
    while (*s >= '0' && *s <= '9') {
        x = x * 10 + (*s++ - '0');
        if (x > 100000) return false;
    }
    *v = neg ? -x : x;
    *p = s;
    return true;
}

static bool rest_is_blank(const char *s) {
    while (*s == ' ') s++;
    return *s == '\0';
}

ctl_line_result_t ctl_handle_line(ctl_t *c, const char *line, uint32_t now_ms) {
    const char *p = line;
    while (*p == ' ') p++;
    long l = 0, r = 0;
    if (p[0] == 'S' && rest_is_blank(p + 1)) {
        l = r = 0;
    } else if (p[0] == 'M' && p[1] == ' ') {
        p++;
        if (!parse_int(&p, &l) || !parse_int(&p, &r) || !rest_is_blank(p)) goto bad;
        if (labs(l) > CTL_MAX_CMD || labs(r) > CTL_MAX_CMD) goto bad;
    } else {
        goto bad;
    }

    c->last_cmd_ms = now_ms;
    c->n_ok++;
    if (l == 0 && r == 0) {
        cut_to_zero(c, CTL_STATE_STOPPED);  // stops are never ramped
    } else {
        c->target[0] = (int16_t)l;
        c->target[1] = (int16_t)r;
        c->state = CTL_STATE_RUNNING;
    }
    return CTL_LINE_OK;

bad:
    c->n_bad++;
    return CTL_LINE_BAD;
}

int ctl_feed(ctl_t *c, const uint8_t *data, size_t n, uint32_t now_ms) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        char ch = (char)data[i];
        if (ch == '\r') continue;
        if (ch == '\n') {
            if (c->overflow) {
                c->n_bad++;
                bad++;
            } else if (c->len > 0) {
                c->buf[c->len] = '\0';
                if (ctl_handle_line(c, c->buf, now_ms) != CTL_LINE_OK) bad++;
            }
            c->len = 0;
            c->overflow = false;
            continue;
        }
        if (c->overflow) continue;
        if (c->len >= CTL_LINE_MAX) {
            c->overflow = true;
            continue;
        }
        c->buf[c->len++] = ch;
    }
    return bad;
}

void ctl_set_connected(ctl_t *c, bool connected, uint32_t now_ms) {
    c->connected = connected;
    c->len = 0;
    c->overflow = false;
    if (!connected) cut_to_zero(c, CTL_STATE_DISCONNECTED);
    c->last_cmd_ms = now_ms;  // a fresh connection must still send within the watchdog window
}

static int16_t clamp16(int v, int lim) { return (int16_t)(v > lim ? lim : (v < -lim ? -lim : v)); }

static int16_t slew(int16_t out, int16_t target) {
    int d = (int)target - (int)out;
    if (d > CTL_SLEW_PER_TICK) d = CTL_SLEW_PER_TICK;
    if (d < -CTL_SLEW_PER_TICK) d = -CTL_SLEW_PER_TICK;
    return (int16_t)(out + d);
}

void ctl_tick(ctl_t *c, uint32_t now_ms) {
    if (!c->connected) {
        cut_to_zero(c, CTL_STATE_DISCONNECTED);
        return;
    }
    // Unsigned subtraction is wrap-safe across the 49-day millis rollover.
    if (c->state == CTL_STATE_RUNNING && (uint32_t)(now_ms - c->last_cmd_ms) > CTL_WATCHDOG_MS) {
        cut_to_zero(c, CTL_STATE_WATCHDOG);
        c->n_watchdog++;
        return;
    }
    if (c->state != CTL_STATE_RUNNING) {
        c->out[0] = c->out[1] = 0;
        return;
    }
    for (int i = 0; i < 2; i++) {
        int16_t t = clamp16(c->target[i], c->max_out);
        c->out[i] = slew(c->out[i], t);
    }
}
