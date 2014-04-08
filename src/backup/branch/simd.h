
#ifndef _SIMD_H
#define _SIMD_H

#define BIN_ALIGN 1

#ifdef SIMD
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#ifdef SSE
#define VLEN 4
#define _VECTOR __m128
#define _MM_LOAD _mm_load_ps
#define _MM_STORE _mm_store_ps
#define _MM_ADD _mm_add_ps
#define _MM_SUB _mm_sub_ps
#define _MM_MUL _mm_mul_ps
#define _MM_MIN _mm_min_ps
#define _MM_MAX _mm_max_ps
#define _MM_FLOOR _mm_floor_ps
#define _MM_SET1 _mm_set1_ps
#elif AVX
#define VLEN 8
#define _VECTOR __m256
#define _MM_LOAD _mm256_load_ps
#define _MM_STORE _mm256_store_ps
#define _MM_ADD _mm256_add_ps
#define _MM_SUB _mm256_sub_ps
#define _MM_MUL _mm256_mul_ps
#define _MM_MIN _mm256_min_ps
#define _MM_MAX _mm256_max_ps
#define _MM_FLOOR _mm256_floor_ps
#define _MM_SET1 _mm256_set1_ps
#elif __MIC__
#define VLEN 16
#define _VECTOR __m512
#define _MM_LOAD _mm512_load_ps
#define _MM_STORE _mm512_store_ps
#define _MM_ADD _mm512_add_ps
#define _MM_SUB _mm512_sub_ps
#define _MM_MUL _mm512_mul_ps
#define _MM_MIN _mm512_min_ps
#define _MM_MAX _mm512_max_ps
#define _MM_FLOOR _mm512_floor_ps
#define _MM_SET1 _mm512_set1_ps
#endif


#endif
#endif
