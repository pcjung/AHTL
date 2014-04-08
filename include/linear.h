#include <stdlib.h>
#include <string.h>
#ifdef __MIC__
#define UNROLL_N 2
#else
#define UNROLL_N 8
#endif

#ifdef SIMD
int hist_linear_float_simd
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_count
);
#endif


int hist_linear_float 
(
 float * data, 
 float * boundary,
 int count,
 int * bin,
 int bin_count
);
