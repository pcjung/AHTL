/* enum for timer structure
 * add some more indices if you want
 */
#ifndef _TIMER_H
#define _TIMER_H
enum timer_code{
				ALL, 
				THREAD_HANDLING, 
				HISTOGRAM_CORE, 
				REDUCTION 
}; 

/* timer structure
 * Each histogram function must fill in this structure
 */
typedef struct
{
				unsigned long long int sum[500];
				unsigned long long int current[500];
				double sumT[500];
				double curT[500];
				double from[500];
				double to[500];
				//double freq;
} timer;

typedef enum timer_code work;

/* timer functions */
void timer_init(timer * timer_);
void timer_start(timer * timer_, work w);
void timer_stop(timer * timer_, work w);
long long int timer_get_cycle(timer * timer_, work w);
double timer_get_time(timer * timer_, work w);
#endif
