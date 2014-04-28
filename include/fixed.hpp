#include "histogram.h"
#include "fixed.h"
#include <omp.h>

#include<iostream>

#define MY_START(tid, num_thread, size) (((tid) * size) / num_thread)
#define MY_END(tid, num_thread, size) (((tid + 1) * size) / num_thread)
#ifdef __MIC__
#define FIXED_WIDTH hist_uniform_float_simd_unrolled_4
#else
#define FIXED_WIDTH hist_uniform_float
#endif


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
      switch ( switch_value<T>::value )
      {
        case 1:
          FIXED_WIDTH(Histogram<T>::data_, bin_base_, bin_width_, Histogram<T>::data_size_, Histogram<T>::bin_, Histogram<T>::num_bins_);  
          break;

        default:
          for(int i = 0; i < Histogram<T>::data_size_; i++)
            Histogram<T>::bin_[(unsigned int)(((Histogram<T>::data_[i]) - bin_base_) / bin_width_)]++;
          break;
      }
    }
    else
    {
      int ** private_bins;
      int num_threads;
#pragma omp parallel shared(private_bins, num_threads)
      {
        int tid = omp_get_thread_num();

        size_t start, end;

#pragma omp master
        {
          num_threads = omp_get_num_threads();
          private_bins = new int *[num_threads];
        }

#pragma omp barrier
        // Private bin initilization & data partitioning
        private_bins[tid] = new int[Histogram<T>::num_bins_];
        memset(private_bins[tid], 0, sizeof(int) * Histogram<T>::num_bins_);
        start = MY_START(tid, num_threads, Histogram<T>::data_size_);
        end = MY_END(tid, num_threads, Histogram<T>::data_size_);

        // Perform histogram
        switch ( switch_value<T>::value )
        {
          case 1:
            FIXED_WIDTH(Histogram<T>::data_ + start, bin_base_, bin_width_, end - start, private_bins[tid], Histogram<T>::num_bins_);  
            break;

          default:
            for(int i = start; i < end; i++)
              Histogram<T>::bin_[(unsigned int)(((Histogram<T>::data_[i]) - bin_base_) / bin_width_)]++;
            break;
        }
#pragma omp barrier
        // Bin partitioning
        start = MY_START(tid, num_threads, Histogram<T>::num_bins_);
        end = MY_END(tid, num_threads, Histogram<T>::num_bins_);

        // Reduce the result
        for(int i = 0; i < num_threads; i++)
          for(int j = start; j < end; j++)
            Histogram<T>::bin_[j] += private_bins[i][j];

#pragma omp barrier
        delete private_bins[tid];

#pragma omp master
        {
          delete private_bins;
        }
      }
    }
  }


  template <typename T>
    void FixedHistogram<T>::BuildHistogramShared()
  {
    if(omp_in_parallel()) // Do not parallelize
    {
      switch ( switch_value<T>::value )
      {
        case 1:
          hist_uniform_float_atomic(Histogram<T>::data_, bin_base_, bin_width_, Histogram<T>::data_size_, Histogram<T>::bin_);  
          break;
        default:
          for(int i = 0; i < Histogram<T>::data_size_; i++)
            __sync_fetch_and_add(&Histogram<T>::bin_[(int)(((Histogram<T>::data_[i]) - bin_base_) / bin_width_)], 1);
          break;
      }
    }
    else
    {
      int num_threads;
#pragma omp parallel shared(num_threads)
      {
        int tid = omp_get_thread_num();

        size_t start, end;

#pragma omp master
        {
          num_threads = omp_get_num_threads();
        }

#pragma omp barrier
        // Private bin initilization & data partitioning
        start = MY_START(tid, num_threads, Histogram<T>::data_size_);
        end = MY_END(tid, num_threads, Histogram<T>::data_size_);

        // Perform histogram
        //
        switch ( switch_value<T>::value )
       {
        case 1:
          hist_uniform_float_atomic(Histogram<T>::data_ + start, bin_base_, bin_width_, end - start, Histogram<T>::bin_);  
          break;
        default:
          for(int i = start; i < end; i++)
            __sync_fetch_and_add(&Histogram<T>::bin_[(int)(((Histogram<T>::data_[i]) - bin_base_) / bin_width_)], 1);
          break;
        }
      }
    }
  }
}

