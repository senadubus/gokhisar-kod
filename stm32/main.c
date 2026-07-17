/* 4.2.2.8 Hedef İmha — STM32F411 servo sürüş ve ateşleme katmanı.
 *
 * Görevler:
 *  - RPi'den UART ile ANG,<pan>,<tilt>*CS ve ATES_ET*CS komutlarını alma
 *  - TIM2 CH1/CH2 ile 50 Hz donanımsal PWM üretip pan-tilt servoları sürme
 *  - ATES_ET komutunda GPIO ile MOSFET sürücüyü tetikleme
 *  - İşlem sonrası DURUM,<kod> geri bildirimi (kapalı çevrim)
 *
 * STM32 HAL (CubeMX üretimi init fonksiyonları ile birlikte derlenir).
 * Deterministik çalışma: kesme tabanlı UART alımı + ana döngüde işleme.
 */
#include "main.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern TIM_HandleTypeDef htim2;    /* 50 Hz PWM: CH1 = pan, CH2 = tilt */
extern UART_HandleTypeDef huart1;  /* RPi baglantisi */

#define FIRE_GPIO_PORT   GPIOB
#define FIRE_GPIO_PIN    GPIO_PIN_0
#define FIRE_PULSE_MS    150u      /* MOSFET tetik suresi */

/* Servo PWM: 50 Hz -> 20 ms periyot. TIM2 ayarı: 1 MHz sayac (1 us tik),
 * ARR = 19999. 0 derece = 500 us, 180 derece = 2500 us. */
#define SERVO_MIN_US     500u
#define SERVO_MAX_US     2500u

#define RX_BUF_LEN       64u
static volatile uint8_t  rx_byte;
static char              rx_line[RX_BUF_LEN];
static volatile uint16_t rx_idx = 0;
static volatile uint8_t  line_ready = 0;

/* ---------------- yardımcılar ---------------- */
static uint16_t angle_to_pulse(float angle)
{
    if (angle < 0.0f)   angle = 0.0f;
    if (angle > 180.0f) angle = 180.0f;
    return (uint16_t)(SERVO_MIN_US +
           (angle / 180.0f) * (float)(SERVO_MAX_US - SERVO_MIN_US));
}

static void set_servos(float pan, float tilt)
{
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, angle_to_pulse(pan));
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, angle_to_pulse(tilt));
}

static void send_status(const char *code)
{
    char msg[32];
    int n = snprintf(msg, sizeof(msg), "DURUM,%s\n", code);
    HAL_UART_Transmit(&huart1, (uint8_t *)msg, (uint16_t)n, 100);
}

static uint8_t checksum_ok(char *line)
{
    /* Format: PAYLOAD*CS  (CS = payload baytlarinin XOR'u, 2 hane hex) */
    char *star = strchr(line, '*');
    if (star == NULL) return 0;
    *star = '\0';
    uint8_t cs = 0;
    for (char *p = line; *p; ++p) cs ^= (uint8_t)(*p);
    return (uint8_t)strtoul(star + 1, NULL, 16) == cs;
}

/* ---------------- UART kesmesi ---------------- */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart != &huart1) return;
    if (rx_byte == '\n') {
        rx_line[rx_idx] = '\0';
        rx_idx = 0;
        line_ready = 1;
    } else if (rx_idx < RX_BUF_LEN - 1) {
        rx_line[rx_idx++] = (char)rx_byte;
    } else {
        rx_idx = 0;               /* tasma: satiri at */
    }
    HAL_UART_Receive_IT(&huart1, (uint8_t *)&rx_byte, 1);
}

/* ---------------- komut işleme ---------------- */
static void fire(void)
{
    /* MOSFET surucuyu tetikle -> ateşleme mekanizması aktif */
    HAL_GPIO_WritePin(FIRE_GPIO_PORT, FIRE_GPIO_PIN, GPIO_PIN_SET);
    HAL_Delay(FIRE_PULSE_MS);
    HAL_GPIO_WritePin(FIRE_GPIO_PORT, FIRE_GPIO_PIN, GPIO_PIN_RESET);
    send_status("ATES_OK");       /* RPi'ye geri bildirim (kapali cevrim) */
}

static void process_line(char *line)
{
    if (!checksum_ok(line)) {
        send_status("CS_HATA");
        return;
    }
    if (strncmp(line, "ANG,", 4) == 0) {
        float pan  = strtof(line + 4, NULL);
        char *c = strchr(line + 4, ',');
        if (c == NULL) { send_status("FMT_HATA"); return; }
        float tilt = strtof(c + 1, NULL);
        set_servos(pan, tilt);
    } else if (strcmp(line, "ATES_ET") == 0) {
        fire();
    } else {
        send_status("KOMUT_HATA");
    }
}

/* ---------------- ana program ---------------- */
int main(void)
{
    HAL_Init();
    SystemClock_Config();          /* CubeMX uretimi */
    MX_GPIO_Init();
    MX_TIM2_Init();                /* PSC/ARR: 1 us tik, 20 ms periyot */
    MX_USART1_UART_Init();

    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);
    set_servos(90.0f, 90.0f);      /* orta konum */

    HAL_UART_Receive_IT(&huart1, (uint8_t *)&rx_byte, 1);
    send_status("HAZIR");

    for (;;) {
        if (line_ready) {
            line_ready = 0;
            char local[RX_BUF_LEN];
            strncpy(local, rx_line, RX_BUF_LEN);
            process_line(local);
        }
        /* Servo PWM donanimsal calisir; dongu yalnizca komut isler. */
    }
}
