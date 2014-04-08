#include"fixed.hpp"
#include<iostream>
#include<cstdlib>
#include<sys/time.h>

#define N 100000000
#define M 100000000

int main(void)
{
  float * data = new float[N];
  int i;
  int count = N;
  int * result = new int[M];
  struct timeval tv;
  double tmp1, tmp2;
  srand(time(NULL));

  gettimeofday(&tv, NULL);
  tmp1 = (double)tv.tv_sec + (double)tv.tv_usec / 1.e6;

  for(i = 0; i < count; i++)
    data[i] = ((float)rand() / (float)RAND_MAX) * M;

  gettimeofday(&tv, NULL);
  tmp2 = ((double)tv.tv_sec + (double)tv.tv_usec / 1.e6);
  std::cout << "Data init : " << tmp2 - tmp1 << " sec" << std::endl;
  tmp1 = tmp2;

  AHTL::FixedHistogram<float> x = AHTL::FixedHistogram<float>(M, 0, 1);
  

  gettimeofday(&tv, NULL);
  tmp2 = ((double)tv.tv_sec + (double)tv.tv_usec / 1.e6);
  std::cout << "Instance : " << tmp2 - tmp1 << " sec" << std::endl;
  tmp1 = tmp2;

  x.SetData(data, N);

  gettimeofday(&tv, NULL);
  tmp2 = ((double)tv.tv_sec + (double)tv.tv_usec / 1.e6);
  std::cout << "Set Data : " << tmp2 - tmp1 << " sec" << std::endl;
  tmp1 = tmp2;

  x.BuildHistogramPrivate();
  
  gettimeofday(&tv, NULL);
  tmp2 = ((double)tv.tv_sec + (double)tv.tv_usec / 1.e6);
  std::cout << "Build Histogram : " << tmp2 - tmp1 << " sec" << std::endl;
  tmp1 = tmp2;
  
  x.ExportResult(result);

  gettimeofday(&tv, NULL);
  tmp2 = ((double)tv.tv_sec + (double)tv.tv_usec / 1.e6);
  std::cout << "Export : " << tmp2 - tmp1 << " sec" << std::endl;
  tmp1 = tmp2;
  
  for(i = 0; i < 10; i++) std::cout<<result[i]<<std::endl;
  delete data;
}
