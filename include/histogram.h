#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H
#include<cstring>
//virtual class for variable histograms

namespace AHTL{
  template<typename InputType> class Histogram
  {
    public: 
      Histogram()
        : 
          initialized_(false), 
          data_available_(false)  
          //Note that number of thread is not detrmined explicitly.
          //To be compatible with OpenMP environment, 
          //we use OpenMP environment variables to specify the number of threads
    {}
      Histogram(int N)
        : 
          initialized_(true), 
          data_available_(false), 
          num_bins_(N) 
          //Note that number of thread is not detrmined explicitly.
          //To be compatible with OpenMP environment, 
          //we use OpenMP environment variables to specify the number of threads
    {
      bin_ = new int[N];
    }



      //Specify a data chunk to process.
      //We can process multiple Data chunk by calling SetData() and BuildHistogram() multipe times alternatively.
      void SetData(InputType * data, size_t size)
      {
        data_ = data;
        data_size_ = size;
        data_available_ = true;
      }

      //Do Sampling on the data chunk before building histogram
      //Adjust the method based on the sampling result
      //      virtual void DoSample() = 0;


      //Build Histogram using multiple threads, without boundary checks.
      //Segmentation faults may occur for OutofRange inputs(Not fit in bin ranges)
      virtual void BuildHistogram() = 0;

      //Bulid Histogram using multipe threads, with boudnary checks.
      //May throws exceptions or return ERRNO when OutOfRange input element is dected ..
      //Not decided yet.
      virtual void BuildHistogramBoundaryCheck() = 0;


      //Copy the histogram result to a buffer given as an argument (result)

      void ExportResult(int * result)
      {
        memcpy(result, bin_, sizeof(int) * num_bins_);
      }

      //Clean the bin
      void CleanResult()
      {
        memset(bin_, 0, sizeof(int) * num_bins_);
      }

      //some functions to access non-public fields
      inline size_t num_bins() const { return num_bins_; }
      inline int * bin() const { return bin_; }
      inline InputType * data() const { return data_; }
      inline size_t data_size() const { return data_size_; }
      inline int num_threads() const { return num_threads_; }
      inline int initialized() const { return initialized_ && data_available_;}

    protected:
      bool initialized_;   
      bool data_available_; 
      size_t num_bins_;
      int * bin_;
      InputType * data_;
      size_t data_size_;
      int num_threads_;
      float * tree_;
  };

  template<typename InputType> class FixedHistogram : public Histogram <InputType>
  {
    public:
      FixedHistogram(int N, InputType base, InputType width) 
        :
          Histogram<InputType>::Histogram(N)
    {
      bin_base_ = base;
      bin_width_ = width;
    }

      void DoSample();

      // General histogram funciton with adoption
      void BuildHistogram(); 
      void BuildHistogramBoundaryCheck();

      // Specific histogram functions
      void BuildHistogramPrivate();  
      void BuildHistogramShared();

      //some functions to access non-public fields
      inline InputType bin_base() const { return bin_base_; }
      inline InputType bin_width() const { return bin_width_; }

    protected:
      InputType bin_base_;
      InputType bin_width_;
      // BuildHistogramGatherScatter(int NumThread);
  };


  template<typename InputType> class VariableHistogram : public Histogram <InputType>
  {
    public:
      VariableHistogram(int N, InputType * boundary)
        : Histogram<InputType>::Histogram(N)  
      {
        boundaries_ = new InputType[N + 1];
        memcpy(boundaries_, boundary, sizeof(InputType) * (N + 1));
      }

      VariableHistogram(int N)
        : Histogram<InputType>::Histogram(N)  
      {
        boundaries_ = new InputType[N + 1];
      }
      void DoSample();

      // Fixed width boundary initialziation for convinience
      void InitFixedWidthBoundaries(InputType base, InputType width)
      {
        boundaries_[0] = base;
        for(int i = 1; i < Histogram<InputType>::num_bins_; i++)
        {
          boundaries_[i] = boundaries_[i-1] + width;
        }
      }

      // General histogram funciton with adoption
      void BuildHistogram(); 
      void BuildHistogramBoundaryCheck();

      // Specific histogram functions
      void BuildHistogramLinearSearch();
      void BuildHistogramBinarySearch();
      void BuildHistogramSortingSearch();
      void BuildHistogramPartitionSearch();

      //some functions to access non-public fields
      inline InputType * boundaries() const { return boundaries_; }
    protected:
      InputType * boundaries_;
  };

  class TextHistogram : public Histogram <char *>
  {
    public:
      TextHistogram(int N = 1000) // default N value
        //: 
        //initialized_(true),
        //num_bins_(N) 
        : Histogram<char *>::Histogram(N)
      {
      }
      void SetData(char ** data, size_t size);
      // Override SetData function, because text data needs some additional preprocessing operation
      void DoSample();
      // General histogram funciton with adoption
      void BuildHistogram(); 
      // Specific histogram functions
      void BuildHistogramPartition();
      void BuildHistogramPrivate();

      //some functions to access non-public fields
      inline char ** words() const { return words_; }
      inline char * base_addr() const { return base_addr_; }
    protected:
      char ** words_;
      char * base_addr_;

  };
  

  // Structure for type checking
  template <typename T>
  struct switch_value {};

  template <> struct switch_value<float>
  {
    enum { value = 1 };
  };
  template <> struct switch_value<int>
  {
    enum { value = 2 };
  };

}


#endif



