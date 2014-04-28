#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef SIMD
#ifdef __MIC__
int hist_partition_float_simd
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 int * bin,
 unsigned int bin_count
 );
#endif
#endif

int hist_partition_float 
(
 float * data, 
 float * boundary,
 unsigned int count,
 int * bin,
 unsigned int bin_count
 );
