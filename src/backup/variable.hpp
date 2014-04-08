#include"histogram.h"
#include<iostream>

float * hist_build_tree(float * _boundary, unsigned int _bin_count);
int hist_binary_float_simd 
(
 float * data, //data should be aligned
 float * _boundary,
 unsigned int count,
 unsigned int * bin,
 unsigned int _bin_count,
 float * simd_pack
 );

namespace AHTL{




#ifdef __MIC__
  template <typename T>
  void VariableHistogram<T>::BuildHistogramLinearSearch()
  {
     hist_linear_float_simd(data_, boundary_, count_, bin_, bin_count_);
  }
  
  template <typename T>
  void VariableHistogram<T>::BuildHistogramBinarySearch()
  {
    float * tree = hist_build_tree(boundary_, num_bins_);

    hist_binary_float_simd(data_, boundary_, data_size_, bin_, num_bins_, tree);

    _mm_free(tree);
  }

  template <typename T>
  void VariableHistogram<T>::BuildHistogramPartitionSearch()
  {

  }
  
  void VariableHistogram<float>::BuildHistogramLinearSearch()
  {
  }
  
  template <typename T>
  void VariableHistogram<float>::BuildHistogramBinarySearch()
  {
  }

  template <typename T>
  void VariableHistogram<float>::BuildHistogramPartitionSearch()
  {
  }
#else
  template <typename T>
  void VariableHistogram<T>::BuildHistogramLinearSearch()
  {
  }
  
  template <typename T>
  void VariableHistogram<T>::BuildHistogramBinarySearch()
  {
  }

  template <typename T>
  void VariableHistogram<T>::BuildHistogramPartitionSearch()
  {
  }
  
  void VariableHistogram<float>::BuildHistogramLinearSearch()
  {
  }
  
  template <typename T>
  void VariableHistogram<float>::BuildHistogramBinarySearch()
  {
  }

  template <typename T>
  void VariableHistogram<float>::BuildHistogramPartitionSearch()
  {
  }#endif

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

