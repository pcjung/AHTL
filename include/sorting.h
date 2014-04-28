#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef SIMD
int hist_sorting_float_simd
(
 float * data, 
 float * boundary,
 unsigned int count,
 int * bin,
 unsigned int bin_count
 );
#endif
int hist_sorting_float 
(
 float * data, 
 float * boundary,
 unsigned int count,
 int * bin,
 unsigned int bin_count
 );
