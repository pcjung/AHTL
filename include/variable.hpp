#include"histogram.h"
#include"linear.h"
#include"binary.h"
#include"sorting.h"
#include"partition.h"
#include <omp.h>

#include<iostream>

#define MY_START(tid, num_thread, size) (((tid) * size) / num_thread)
#define MY_END(tid, num_thread, size) (((tid + 1) * size) / num_thread)

#ifdef SIMD
#define LINEAR hist_linear_float_simd
#define BINARY hist_binary_float_simd
#define SORTING hist_sorting_float_simd
#ifdef __MIC__
#define PARTITION hist_partition_float_simd
#else
#define PARTITION hist_partition_float
#endif
#else
#define LINEAR hist_linear_float
#define BINARY hist_binary_float
#define SORTING hist_sorting_float_simd
#define PARTITION hist_partition_float
#endif

namespace AHTL{

  template <typename T>
    void VariableHistogram<T>::DoSample()
    {
      //Not yet implemented
    }

  template <typename T>
    void VariableHistogram<T>::BuildHistogram()
    {
      //Not yet implemented
    }

  template <typename T>
    void VariableHistogram<T>::BuildHistogramBoundaryCheck()
    {
      //Not yet implemented
    }

  template <typename T>
    void VariableHistogram<T>::BuildHistogramLinearSearch()
    {
      if(omp_in_parallel()) // Do not parallelize
      {
        switch ( switch_value<T>::value )
        {
          case 1:
            LINEAR(Histogram<T>::data_, boundaries_, Histogram<T>::data_size_, Histogram<T>::bin_, Histogram<T>::num_bins_);
            break;

          default:
            for(int i = 0; i < Histogram<T>::data_size_; i++)
            {
              int j;
              T d = Histogram<T>::data_[i];
              for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
              Histogram<T>::bin_[j]++;
            }
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
              LINEAR(Histogram<T>::data_ + start, boundaries_, end - start, private_bins[tid], Histogram<T>::num_bins_);
              break;

            default:
              for(int i = start; i < end; i++)
              {
                int j;
                T d = Histogram<T>::data_[i];
                for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
                private_bins[tid][j]++;
              } 
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
    void VariableHistogram<T>::BuildHistogramBinarySearch()
    {

      float * tree = hist_build_tree(boundaries_, Histogram<T>::num_bins_);
      if(omp_in_parallel()) // Do not parallelize
      {
        switch ( switch_value<T>::value )
        {
          case 1:
            BINARY(Histogram<T>::data_, boundaries_, Histogram<T>::data_size_, Histogram<T>::bin_, Histogram<T>::num_bins_, tree);
            break;

          default:
            for(int i = 0; i < Histogram<T>::data_size_; i++)
            {
              int j;
              T d = Histogram<T>::data_[i];
              for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
              Histogram<T>::bin_[j]++;
            }
            break;
        }
      }
      else
      {
        int ** private_bins;
        int num_threads;
#pragma omp parallel shared(private_bins, num_threads, tree)
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
              BINARY(Histogram<T>::data_ + start, boundaries_, end - start, private_bins[tid], Histogram<T>::num_bins_, tree);
              break;

            default:
              for(int i = start; i < end; i++)
              {
                int j;
                T d = Histogram<T>::data_[i];
                for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
                private_bins[tid][j]++;
              } 
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
      _mm_free(tree);
    }

  template <typename T>
    void VariableHistogram<T>::BuildHistogramSortingSearch()
    {
      if(omp_in_parallel()) // Do not parallelize
      {
        switch ( switch_value<T>::value )
        {
          case 1:
            SORTING(Histogram<T>::data_, boundaries_, Histogram<T>::data_size_, Histogram<T>::bin_, Histogram<T>::num_bins_);
            break;

          default:
            for(int i = 0; i < Histogram<T>::data_size_; i++)
            {
              int j;
              T d = Histogram<T>::data_[i];
              for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
              Histogram<T>::bin_[j]++;
            }
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
              SORTING(Histogram<T>::data_ + start, boundaries_, end - start, private_bins[tid], Histogram<T>::num_bins_);
              break;

            default:
              for(int i = start; i < end; i++)
              {
                int j;
                T d = Histogram<T>::data_[i];
                for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
                private_bins[tid][j]++;
              } 
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
    void VariableHistogram<T>::BuildHistogramPartitionSearch()
    {
      if(omp_in_parallel()) // Do not parallelize
      {
        switch ( switch_value<T>::value )
        {
          case 1:
            PARTITION(Histogram<T>::data_, boundaries_, Histogram<T>::data_size_, Histogram<T>::bin_, Histogram<T>::num_bins_);
            break;

          default:
            for(int i = 0; i < Histogram<T>::data_size_; i++)
            {
              int j;
              T d = Histogram<T>::data_[i];
              for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
              Histogram<T>::bin_[j]++;
            }
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
              PARTITION(Histogram<T>::data_ + start, boundaries_, end - start, private_bins[tid], Histogram<T>::num_bins_);
              break;

            default:
              for(int i = start; i < end; i++)
              {
                int j;
                T d = Histogram<T>::data_[i];
                for(j = Histogram<T>::num_bins_ - 1; d < boundaries_[j]; j--);
                private_bins[tid][j]++;
              } 
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
}

