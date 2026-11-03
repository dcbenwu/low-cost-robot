#pragma once
typedef int TickType_t;
#define pdMS_TO_TICKS(ms) (ms)
void vTaskDelay(TickType_t ticks);