#include "histogram.h"
#include "fixed.h"
#include <omp.h>

#include<iostream>

#define MY_START(size, num_thread, tid) (size * (tid / num_thread)
#define MY_SIZE

namespace AHTL{
  template <typename T>
    void FixedHistogram<T>::DoSample()
    {
    }

  template <typename T>
    void FixedHistogram<T>::BuildHistogram()
    {

    }

  template <typename T>
    void FixedHistogram<T>::BuildHistogramBoundaryCheck()
    {

    }

  template <typename T>
    void FixedHistogram<T>::BuildHistogramPrivate()
    {
      if(omp_in_parallel()) // Do not parallelize
      {
        std::cout<<num_thread<<std::endl;
        for(int i = 0; i < Histogram<T>::data_size_; i++)
          Histogram<T>::bin_[(unsigned int)(((Histogram<T>::data_[i]) - bin_base_) / bin_width_)]++;
      }
      else
      {
        int num_thread = omp_get_num_threads();
        std::cout<<num_thread<<std::endl;
        for(int i = 0; i < Histogram<T>::data_size_; i++)
          Histogram<T>::bin_[(unsigned int)(((Histogram<T>::data_[i]) - bin_base_) / bin_width_)]++;
      }
    }

#ifdef __MIC__
  void FixedHistogram<float>::BuildHistogramPrivate()
  {
    hist_uniform_float_simd_unrolled_4(data_, bin_base_, bin_width_, data_size_, bin_, num_bins_);  
  }
#else
  void FixedHistogram<float>::BuildHistogramPrivate()
  {
      if(omp_in_parallel()) // Do not parallelize
      {
        int num_thread = omp_get_num_threads();
        std::cout<<num_thread<<std::endl;
    hist_uniform_float(data_, bin_base_, bin_width_, data_size_, bin_, num_bins_);  
      }
      else
      {
        int ** private_bins;
#pragma omp parallel shared(private_bins)
        {
          int num_threads;
          int tid = omp_get_thread_num();

#pragma omp master
          {
            num_threads = omp_get_num_threads();
            private_bins = new int *[num_threads];
          }
#pragma omp barrier

          private_bins[tid] = new int[num_bins_];
          memset(private_bins[tid], 0, num_bins_);




        }
    hist_uniform_float(data_, bin_base_, bin_width_, data_size_, bin_, num_bins_);  
      }
  }
#endif

  template <typename T>
    void FixedHistogram<T>::BuildHistogramShared()
    {
      int i;
      for(i = 0; i < data_size_; i++)
        __sync_fetch_and_add(&bin_[(int)(((data_[i]) - bin_base_) / bin_width_)], 1);
    }

  void FixedHistogram<float>::BuildHistogramShared()
  {
    hist_uniform_float_atomic(data_, bin_base_, bin_width_, data_size_, bin_);  
  }
}

