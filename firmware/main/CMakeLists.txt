idf_component_register(
    SRCS "main.c" "control.c" "motor.c" "encoder.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES bt nvs_flash esp_driver_ledc esp_driver_pcnt esp_driver_gpio esp_timer
)
