#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef SIMD
#ifdef __MIC__
int hist_uniform_float_simd_unrolled_4 
(
 float * data, //data should be aligned
 float base, 
 float width,
 int count,
 int * bin,
 int bin_size 
 );
#endif
#endif

int hist_uniform_float 
(
 float * data,
 float base, 
 float width,
 int count,
 int * bin,
 int bin_size
 );

int hist_uniform_float_atomic 
(
 float * data,
 float base, 
 float width,
 int count,
 int * bin 
 );


