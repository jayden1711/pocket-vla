// pocket-vla ESP32-S3 BLE motor controller.
//
// BLE: Nordic-UART-style service. The app writes command lines to RX ("M 120 -80\n"),
// subscribes to TX for telemetry ("E <ms> <encL> <encR> <outL> <outR> <state> <bad>\n").
// All safety logic (watchdog, disconnect stop, slew limit) is in control.c and runs in
// the control task here, independent of anything the app does.
#include <stdio.h>
#include <string.h>

#include "board.h"
#include "control.h"
#include "encoder.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "host/ble_hs.h"
#include "motor.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char *TAG = "pvla";

static ctl_t g_ctl;
static portMUX_TYPE g_lock = portMUX_INITIALIZER_UNLOCKED;
static TaskHandle_t g_control_task;

static uint16_t g_conn = BLE_HS_CONN_HANDLE_NONE;
static uint16_t g_tx_handle;
static volatile bool g_tx_subscribed;
static uint8_t g_own_addr_type;

static uint32_t now_ms(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

// 6E400001-B5A3-F393-E0A9-E50E24DCCA9E (little-endian byte order for NimBLE)
static const ble_uuid128_t SVC_UUID =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);
static const ble_uuid128_t RX_UUID =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3, 0xb5, 0x02, 0x00, 0x40, 0x6e);
static const ble_uuid128_t TX_UUID =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0, 0x93, 0xf3, 0xa3, 0xb5, 0x03, 0x00, 0x40, 0x6e);

static int rx_access(uint16_t conn, uint16_t attr, struct ble_gatt_access_ctxt *ctxt, void *arg) {
    (void)conn, (void)attr, (void)arg;
    if (ctxt->op != BLE_GATT_ACCESS_OP_WRITE_CHR) return BLE_ATT_ERR_UNLIKELY;
    uint8_t buf[128];
    uint16_t len = 0;
    if (ble_hs_mbuf_to_flat(ctxt->om, buf, sizeof(buf), &len) != 0) return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    uint32_t t = now_ms();
    taskENTER_CRITICAL(&g_lock);
    ctl_feed(&g_ctl, buf, len, t);
    taskEXIT_CRITICAL(&g_lock);
    xTaskNotifyGive(g_control_task);  // apply a stop now, not at the next tick
    return 0;
}

static int tx_access(uint16_t conn, uint16_t attr, struct ble_gatt_access_ctxt *ctxt, void *arg) {
    (void)conn, (void)attr, (void)ctxt, (void)arg;
    return BLE_ATT_ERR_READ_NOT_PERMITTED;
}

static const struct ble_gatt_svc_def g_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &SVC_UUID.u,
        .characteristics =
            (struct ble_gatt_chr_def[]){
                {.uuid = &RX_UUID.u, .access_cb = rx_access, .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP},
                {.uuid = &TX_UUID.u, .access_cb = tx_access, .val_handle = &g_tx_handle, .flags = BLE_GATT_CHR_F_NOTIFY},
                {0},
            },
    },
    {0},
};

static void advertise(void);

static void set_connected(bool c) {
    uint32_t t = now_ms();
    taskENTER_CRITICAL(&g_lock);
    ctl_set_connected(&g_ctl, c, t);
    taskEXIT_CRITICAL(&g_lock);
    xTaskNotifyGive(g_control_task);
}

static int gap_event(struct ble_gap_event *ev, void *arg) {
    (void)arg;
    switch (ev->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (ev->connect.status == 0) {
                g_conn = ev->connect.conn_handle;
                set_connected(true);
                ESP_LOGI(TAG, "connected");
                // 15-30 ms interval, 2 s supervision timeout (inside Apple's accessory guidelines).
                // The 300 ms command watchdog, not the supervision timeout, is what stops
                // the robot if the phone goes silent.
                struct ble_gap_upd_params p = {
                    .itvl_min = 12, .itvl_max = 24, .latency = 0, .supervision_timeout = 200,
                };
                ble_gap_update_params(g_conn, &p);
            } else {
                advertise();
            }
            break;
        case BLE_GAP_EVENT_DISCONNECT:
            set_connected(false);
            g_conn = BLE_HS_CONN_HANDLE_NONE;
            g_tx_subscribed = false;
            ESP_LOGW(TAG, "disconnected (reason %d) -> stop", ev->disconnect.reason);
            advertise();
            break;
        case BLE_GAP_EVENT_SUBSCRIBE:
            if (ev->subscribe.attr_handle == g_tx_handle) g_tx_subscribed = ev->subscribe.cur_notify;
            break;
        case BLE_GAP_EVENT_ADV_COMPLETE:
            advertise();
            break;
        default:
            break;
    }
    return 0;
}

static void advertise(void) {
    struct ble_hs_adv_fields f = {0};
    f.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    f.uuids128 = (ble_uuid128_t *)&SVC_UUID;
    f.num_uuids128 = 1;
    f.uuids128_is_complete = 1;
    ble_gap_adv_set_fields(&f);
    struct ble_hs_adv_fields rsp = {0};
    rsp.name = (const uint8_t *)BLE_DEVICE_NAME;
    rsp.name_len = strlen(BLE_DEVICE_NAME);
    rsp.name_is_complete = 1;
    ble_gap_adv_rsp_set_fields(&rsp);
    struct ble_gap_adv_params ap = {.conn_mode = BLE_GAP_CONN_MODE_UND, .disc_mode = BLE_GAP_DISC_MODE_GEN};
    int rc = ble_gap_adv_start(g_own_addr_type, NULL, BLE_HS_FOREVER, &ap, gap_event, NULL);
    if (rc != 0 && rc != BLE_HS_EALREADY) ESP_LOGE(TAG, "adv start rc=%d", rc);
}

static void on_sync(void) {
    ble_hs_id_infer_auto(0, &g_own_addr_type);
    advertise();
}

static void on_reset(int reason) { ESP_LOGE(TAG, "BLE host reset %d", reason); }

static void host_task(void *arg) {
    (void)arg;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void send_telemetry(const ctl_t *snap, uint32_t t) {
    if (!g_tx_subscribed || g_conn == BLE_HS_CONN_HANDLE_NONE) return;
    int32_t el, er;
    encoder_read(&el, &er);
    char line[80];
    int n = snprintf(line, sizeof(line), "E %lu %ld %ld %d %d %d %lu\n", (unsigned long)t, (long)el, (long)er,
                     snap->out[0], snap->out[1], (int)snap->state, (unsigned long)snap->n_bad);
    struct os_mbuf *om = ble_hs_mbuf_from_flat(line, n);
    if (om) ble_gatts_notify_custom(g_conn, g_tx_handle, om);  // dropped if the stack is busy
}

static void control_task(void *arg) {
    (void)arg;
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));  // a hung control loop reboots; STBY pull-down keeps motors off
    uint32_t next_tick = now_ms() + CTL_TICK_MS;
    uint32_t next_tel = now_ms();
    for (;;) {
        uint32_t t = now_ms();
        int32_t wait = (int32_t)(next_tick - t);
        if (wait > 0) ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(wait) ? pdMS_TO_TICKS(wait) : 1);
        t = now_ms();

        ctl_t snap;
        taskENTER_CRITICAL(&g_lock);
        // Only advance the slew/watchdog on real tick boundaries; an early wake (new
        // command, disconnect) just applies the current outputs, which a stop has already zeroed.
        if ((int32_t)(t - next_tick) >= 0) {
            ctl_tick(&g_ctl, t);
            next_tick += CTL_TICK_MS;
            if ((int32_t)(t - next_tick) > 5 * (int32_t)CTL_TICK_MS) next_tick = t + CTL_TICK_MS;  // don't burst-catch-up
        }
        snap = g_ctl;
        taskEXIT_CRITICAL(&g_lock);

        motor_apply(snap.out[0], snap.out[1]);
        esp_task_wdt_reset();

        if ((int32_t)(t - next_tel) >= 0) {
            next_tel = t + TELEMETRY_PERIOD_MS;
            send_telemetry(&snap, t);
        }
    }
}

void app_main(void) {
    motor_init();
    motor_apply(0, 0);
    encoder_init();
    ctl_init(&g_ctl, now_ms());
    g_ctl.max_out = BOOT_MAX_OUT;

    xTaskCreatePinnedToCore(control_task, "control", 4096, NULL, configMAX_PRIORITIES - 2, &g_control_task, 1);

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    ESP_ERROR_CHECK(nimble_port_init());
    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.reset_cb = on_reset;
    ble_svc_gap_init();
    ble_svc_gatt_init();
    ESP_ERROR_CHECK(ble_gatts_count_cfg(g_svcs));
    ESP_ERROR_CHECK(ble_gatts_add_svcs(g_svcs));
    ble_svc_gap_device_name_set(BLE_DEVICE_NAME);
    nimble_port_freertos_init(host_task);
    ESP_LOGI(TAG, "ready: watchdog %u ms, slew %d/tick @ %u ms, cap %d", CTL_WATCHDOG_MS, CTL_SLEW_PER_TICK,
             CTL_TICK_MS, BOOT_MAX_OUT);
}
