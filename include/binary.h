#include <stdlib.h>
#include <stdio.h>
#include <string.h>

float * hist_build_tree(float * _boundary, unsigned int _bin_count);

#ifdef SIMD
int hist_binary_float_simd 
(
 float * data, //data should be aligned
 float * _boundary,
 unsigned int count,
 int * bin,
 unsigned int _bin_count,
 float * simd_pack
 );
#ifdef __MIC__
int hist_binary_float_simd_sg 
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 int * bin,
 unsigned int bin_count,
 float * tree
 );
#endif
#endif

int hist_binary_float 
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 int * bin,
 unsigned int bin_count,
 float * tree
 );
