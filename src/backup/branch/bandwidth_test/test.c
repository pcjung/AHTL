#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define CHUNK 134217728
int main(void)
{
  omp_set_num_threads(2);

  unsigned int * buffer;
  unsigned int t, i;
#pragma omp parallel
  {
    if(omp_get_thread_num() == 0)
    {
      printf("read first\n");
      buffer = malloc(CHUNK * sizeof(unsigned int));
      for(i = 0; i < CHUNK; i++)
        t += buffer[i];
    }
#pragma omp barrier
    if(omp_get_thread_num() == 1)
    {
      printf("read second\n");
      for(i = 0; i < CHUNK; i++)
        t += buffer[i];
    }
  }
  printf("%d\n", t);
}
