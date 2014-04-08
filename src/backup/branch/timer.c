#include "timer.h"
#include <time.h>
#include <sched.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

static double get_cpu_freq()
{
	const double DEFAULT_CPU_FREQ = 3.33e9;
	static double freq = DBL_MAX;
	if (DBL_MAX == freq) {
		volatile double a = rand()%1024, b = rand()%1024;
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
		unsigned long long t1 = __rdtsc();
		int i = 0;
		for (i = 0; i < 1024*1024; i++) {
			a += a*b + b/a;
		}
		unsigned long long dt = __rdtsc() - t1;
		gettimeofday(&tv2, NULL);
		freq = dt/((tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec)/1.e6);
	}

	return freq;
}



void timer_init(timer * timer_)
{
				int i;
				for(i = 0; i < 500; i++)
				{
								timer_->sum[i] = 0;
								timer_->sumT[i] = 0.0;
								timer_->current[i] = -1;
				}
				
				struct timeval tv;
				gettimeofday(&tv, NULL);
				timer_->from[499] = (double)tv.tv_sec + (double)tv.tv_usec / 1.e6;
}

void timer_start(timer * t, work w)
{
				assert(t->current[w] == -1 && "This timer is currently running!");

				t->current[w] = __rdtsc();
				struct timeval tv;
				gettimeofday(&tv, NULL);
				t->curT[w] = (double)tv.tv_sec + (double)tv.tv_usec / 1.e6;
				t->from[w] = t->curT[w] - t->from[499];
}

void timer_stop(timer * t, work w)
{
				assert(t->current[w] != -1 && "This timer is not running!");

				t->sum[w] += (__rdtsc() - t->current[w]);
				struct timeval tv;
				gettimeofday(&tv, NULL);
				t->sumT[w] +=  (double)tv.tv_sec + (double)tv.tv_usec / 1.e6 - t->curT[w];
				t->to[w] = (double)tv.tv_sec + (double)tv.tv_usec / 1.e6 - t->from[499];
				t->current[w] = -1;
}

long long int timer_get_cycle(timer * t, work w)
{
				return t->sum[w];
}
double timer_get_time(timer * t, work w)
{
				return t->sumT[w];
}

