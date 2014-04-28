#include"ahtl.h"
#include<iostream>
#include<cstdlib>
#include<sys/time.h>
#include<omp.h>

#define N 10000000
#define M 256
#define EPS 0.01

#define TIMER_INIT() double __tmp1, __tmp2; struct timeval __tv;
#define TIMER_VARS __tmp1, __tmp2, __tv
#define TIMER_START() gettimeofday(&__tv, NULL); __tmp1 = (double)__tv.tv_sec + (double)__tv.tv_usec / 1.e6;
#define TIMER_CHECK(x) \
  gettimeofday(&__tv, NULL); __tmp2 = ((double)__tv.tv_sec + (double)__tv.tv_usec / 1.e6); \
  std::cout << x << " : " << __tmp2 - __tmp1 << " sec" << std::endl; \
  __tmp1 = __tmp2; \
   
#define RESULT_CHECK(result, result_compare)   \
  if (memcmp(result, result_compare, sizeof(int) * M) == 0) \
    std::cout << "--correct" << std::endl; \
  else \
  { \
    std::cout << "--incorrect" << std::endl; \
    for(int i = 0; i < M; i++) \
      if(result[i] != result_compare[i]) \
        std::cout << "different at " << i << " :" << result[i] << " " <<result_compare[i] << std::endl; \
  }

int main(void)
{
  float * data = new float[N];
  int i;
  int count = N;
  int * result = new int[M];
  int * result_compare = new int[M];
  srand(time(NULL));
  TIMER_INIT() 
  TIMER_START()
  for(i = 0; i < count; i++)
    data[i] = ((float)rand() / (float)RAND_MAX) * ((float)M - EPS);
  for(i = 0; i < M; i++)
    result_compare[i] = 0;
  for(i = 0; i < count; i++)
    result_compare[(size_t)(data[i])]++;
  TIMER_CHECK("Data & result init")
  AHTL::FixedHistogram<float> x = AHTL::FixedHistogram<float>(M, 0, 1);
  AHTL::VariableHistogram<float> y = AHTL::VariableHistogram<float>(M);
  TIMER_CHECK("Instance init")
  x.SetData(data, N);
  y.SetData(data, N);
  y.InitFixedWidthBoundaries(0,1);
  TIMER_CHECK("Set data")
  #pragma omp parallel
  {
    #pragma omp critical
    {
      std::cout << "OpenMP thread" << omp_get_thread_num() << "ready" << std::endl;
    }
  }
  TIMER_CHECK("OpenMP initialization")

  std::cout << "------ MULTI THREAD EXECUTION ------" << std::endl;
  std::cout << "--- PRIVATE HISTOGRAM ---" << std::endl;
  x.BuildHistogramPrivate();
  TIMER_CHECK("Histogram")
  x.ExportResult(result);
  TIMER_CHECK("Export result")
  RESULT_CHECK(result, result_compare) 
  x.CleanResult();
  TIMER_CHECK("Comparison and clean")


  std::cout << "--- SHARED HISTOGRAM ---" << std::endl;
  x.BuildHistogramShared();
  TIMER_CHECK("Histogram")
  x.ExportResult(result);
  TIMER_CHECK("Export Result")
  RESULT_CHECK(result, result_compare) 
  x.CleanResult();
  TIMER_CHECK("Comparison")


  std::cout << "--- LINEAR HISTOGRAM ---" << std::endl;
  y.BuildHistogramLinearSearch();
  TIMER_CHECK("Histogram")
  y.ExportResult(result);
  TIMER_CHECK("Export Result")
  RESULT_CHECK(result, result_compare) 
  y.CleanResult();
  TIMER_CHECK("Comparison and clean")


  std::cout << "--- BINARY HISTOGRAM ---" << std::endl;
  y.BuildHistogramBinarySearch();
  TIMER_CHECK("Histogram")
  y.ExportResult(result);
  TIMER_CHECK("Export Result")
  RESULT_CHECK(result, result_compare) 
  y.CleanResult();
  TIMER_CHECK("Comparison and clean")
  
  std::cout << "--- SORTING HISTOGRAM ---" << std::endl;
  y.BuildHistogramSortingSearch();
  TIMER_CHECK("Histogram")
  y.ExportResult(result);
  TIMER_CHECK("Export Result")
  RESULT_CHECK(result, result_compare) 
  y.CleanResult();
  TIMER_CHECK("Comparison and clean")

/*
  std::cout << "--- PARTITION HISTOGRAM ---" << std::endl;
  y.BuildHistogramPartitionSearch();
  TIMER_CHECK("Histogram")
  y.ExportResult(result);
  TIMER_CHECK("Export Result")
  RESULT_CHECK(result, result_compare) 
  y.CleanResult();
  TIMER_CHECK("Comparison and clean")
  */


  std::cout << "------ SINGLE THREAD EXECUTION ------" << std::endl;
#pragma omp parallel num_threads(2) shared(x,y,data,result,result_compare,TIMER_VARS) // The library do sequential histogram in omp_parallel region
  {
    while(1){
      if(omp_get_thread_num() != 0)
        break;
      std::cout << "--- PRIVATE HISTOGRAM ---" << std::endl;
      x.BuildHistogramPrivate();
      TIMER_CHECK("Histogram")
      x.ExportResult(result);
      TIMER_CHECK("Export result")
      RESULT_CHECK(result, result_compare) 
      x.CleanResult();
      TIMER_CHECK("Comparison and clean")


      std::cout << "--- SHARED HISTOGRAM ---" << std::endl;
      x.BuildHistogramShared();
      TIMER_CHECK("Histogram")
      x.ExportResult(result);
      TIMER_CHECK("Export Result")
      RESULT_CHECK(result, result_compare) 
      x.CleanResult();
      TIMER_CHECK("Comparison")


      std::cout << "--- LINEAR HISTOGRAM ---" << std::endl;
      y.BuildHistogramLinearSearch();
      TIMER_CHECK("Histogram")
      y.ExportResult(result);
      TIMER_CHECK("Export Result")
      RESULT_CHECK(result, result_compare) 
      y.CleanResult();
      TIMER_CHECK("Comparison and clean")


      std::cout << "--- BINARY HISTOGRAM ---" << std::endl;
      y.BuildHistogramBinarySearch();
      TIMER_CHECK("Histogram")
      y.ExportResult(result);
      TIMER_CHECK("Export Result")
      RESULT_CHECK(result, result_compare) 
      y.CleanResult();
      TIMER_CHECK("Comparison and clean")

      std::cout << "--- SORTING HISTOGRAM ---" << std::endl;
      y.BuildHistogramSortingSearch();
      TIMER_CHECK("Histogram")
      y.ExportResult(result);
      TIMER_CHECK("Export Result")
      RESULT_CHECK(result, result_compare) 
      y.CleanResult();
      TIMER_CHECK("Comparison and clean")

/*
      std::cout << "--- PARTITION HISTOGRAM ---" << std::endl;
      y.BuildHistogramPartitionSearch();
      TIMER_CHECK("Histogram")
      y.ExportResult(result);
      TIMER_CHECK("Export Result")
      RESULT_CHECK(result, result_compare) 
      y.CleanResult();
      TIMER_CHECK("Comparison and clean")
      */
      break;
    }
  }



    delete data;
  }
