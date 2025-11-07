#ifndef TIMER_H
#define TIMER_H

#include <time.h>

typedef struct {
    clock_t start_time;
    clock_t end_time;
} Timer;

void start_timer(Timer *timer);
void stop_timer(Timer *timer);
double get_elapsed_time(Timer *timer);

#endif // TIMER_H