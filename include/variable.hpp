#include"histogram.h"
#include"linear.h"
#include"binary.h"
#include"partition.h"

#include<iostream>

namespace AHTL{




#ifdef SIMD
  void VariableHistogram<float>::BuildHistogramLinearSearch()
  {
     hist_linear_float_simd(data_, boundaries_, data_size_, (unsigned int *)bin_, num_bins_);
  }
  
  void VariableHistogram<float>::BuildHistogramBinarySearch()
  {
    float * tree = hist_build_tree(boundaries_, num_bins_);
 
    hist_binary_float_simd(data_, boundaries_, data_size_, (unsigned int *)bin_, num_bins_, tree);

    _mm_free(tree);
  }

  void VariableHistogram<float>::BuildHistogramPartitionSearch()
  {

  }
  
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
#endif
}

