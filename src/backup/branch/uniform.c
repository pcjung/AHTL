#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "/opt/intel/ipp/include/ipp.h"
#ifdef SIMD
#include "simd.h"
#endif

#define UPDATE_TYPE 0 // 0 : store&read, 1 : extract

//#define NO_UPDATE
//#define NO_CALC

#define ALPHA 1
//#define BIN_OPT 0
#ifdef __MIC__
#define UNROLL_N 4
#else
#define UNROLL_N 32
#endif
#define LOCK "lock ; "
//#define MANUAL_UNROLL
#ifdef SIMD
#ifdef __MIC__
static void printv_epi32(__m512i v, char *str)
{
  int i;
  __declspec(align(64)) int tmp[16];
  printf("%s:", str);
  _mm512_store_epi32(tmp, v);
  for(i=0; i < 16; i++)
  {
    tmp[0] = tmp[i];
    printf("[%d]=%d ", i, tmp[0]);
  }
  printf("\n");
}
#endif
#endif

#ifdef SIMD
#ifdef __MIC__
int hist_uniform_float_simd_unrolled_4 
(
 float * data, //data should be aligned
 float base, 
 float width,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_size 
 )
{
#ifdef USE_GATHER_SCATTER
		int *simd_bin = _mm_malloc((bin_size + 1)*sizeof(unsigned int)*VLEN, 64);
		memset(simd_bin, 0, (bin_size) *sizeof(unsigned int)*VLEN);
#endif
		_VECTOR v_base = _MM_SET1(base);
		width = 1 / width;
		_VECTOR v_width = _MM_SET1(width);
		size_t remainder;
		size_t start = 0;
		if(remainder = ((unsigned long long int)data) % (VLEN * sizeof(float)))
		{
				unsigned int i;
				start = (VLEN - (remainder / sizeof(float)));
				for(i = 0; i < start; i++)
						bin[(unsigned int)((data[i]-base)/width)]++;
		}
		unsigned int i,c;
#ifdef NO_UPDATE
		_VECTOR red = _MM_LOAD(data + start);
#endif

		__m512i offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
//#pragma noprefetch data
		for(i = start, c=0; i+VLEN*4 - 1 < count; i+=VLEN*4)
		{
		_VECTOR tmp =_MM_LOAD(data + i);
		_VECTOR tmp2 =_MM_LOAD(data + i + VLEN);
		_VECTOR tmp3 =_MM_LOAD(data + i + VLEN * 2);
		_VECTOR tmp4 =_MM_LOAD(data + i + VLEN * 3);


#ifndef NO_PREFETCH
		_mm_prefetch((char const *)(data + i + VLEN* 8), _MM_HINT_NT);
		_mm_prefetch((char const *)(data + i + VLEN* 9), _MM_HINT_NT);
		_mm_prefetch((char const *)(data + i + VLEN* 10), _MM_HINT_NT);
		_mm_prefetch((char const *)(data + i + VLEN* 11), _MM_HINT_NT);
#endif
#ifndef NO_CALC
		tmp = _MM_SUB(tmp, v_base);
		tmp2 = _MM_SUB(tmp2, v_base);
		tmp3 = _MM_SUB(tmp3, v_base);
		tmp4 = _MM_SUB(tmp4, v_base);
		tmp = _MM_MUL(tmp, v_width);
		tmp2 = _MM_MUL(tmp2, v_width);
		tmp3 = _MM_MUL(tmp3, v_width);
		tmp4 = _MM_MUL(tmp4, v_width);
		__m512i cvted = _mm512_cvtfxpnt_round_adjustps_epi32(
						tmp, (_MM_ROUND_MODE_ENUM)_MM_ROUND_MODE_DOWN, _MM_EXPADJ_NONE);
		__m512i cvted2 = _mm512_cvtfxpnt_round_adjustps_epi32(
						tmp2, (_MM_ROUND_MODE_ENUM)_MM_ROUND_MODE_DOWN, _MM_EXPADJ_NONE);
		__m512i cvted3 = _mm512_cvtfxpnt_round_adjustps_epi32(
						tmp3, (_MM_ROUND_MODE_ENUM)_MM_ROUND_MODE_DOWN, _MM_EXPADJ_NONE);
		__m512i cvted4 = _mm512_cvtfxpnt_round_adjustps_epi32(
						tmp4, (_MM_ROUND_MODE_ENUM)_MM_ROUND_MODE_DOWN, _MM_EXPADJ_NONE);

		__declspec(align(64)) int buf[16];
#endif
#ifndef NO_UPDATE
#ifdef USE_GATHER_SCATTER
		cvted = _mm512_fmadd_epi32(cvted, _mm512_set1_epi32(16), offsets);
		cvted2 = _mm512_fmadd_epi32(cvted2, _mm512_set1_epi32(16), offsets);
		cvted3 = _mm512_fmadd_epi32(cvted3, _mm512_set1_epi32(16), offsets);
		cvted4 = _mm512_fmadd_epi32(cvted4, _mm512_set1_epi32(16), offsets);
		__m512i cnts = _mm512_i32extgather_epi32(cvted,  simd_bin, _MM_UPCONV_EPI32_NONE,  _MM_SCALE_4, 0);
		cnts = _mm512_add_epi32(cnts, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(simd_bin, cvted, cnts, _MM_SCALE_4);

		__m512i cnts2 = _mm512_i32gather_epi32(cvted2, simd_bin, _MM_SCALE_4);
		cnts2 = _mm512_add_epi32(cnts2, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(simd_bin, cvted2, cnts2, _MM_SCALE_4);

		__m512i cnts3 = _mm512_i32gather_epi32(cvted3, simd_bin, _MM_SCALE_4);
		cnts3 = _mm512_add_epi32(cnts3, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(simd_bin, cvted3, cnts3, _MM_SCALE_4);

		__m512i cnts4 = _mm512_i32gather_epi32(cvted4, simd_bin, _MM_SCALE_4);
		cnts4 = _mm512_add_epi32(cnts4, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(simd_bin, cvted4, cnts4, _MM_SCALE_4);
#else
		_mm512_store_epi32(buf, cvted);
		bin[buf[0]]++;
		bin[buf[1]]++;
		bin[buf[2]]++;
		bin[buf[3]]++;
		bin[buf[4]]++;
		bin[buf[5]]++;
		bin[buf[6]]++;
		bin[buf[7]]++;
		bin[buf[8]]++;
		bin[buf[9]]++;
		bin[buf[10]]++;
		bin[buf[11]]++;
		bin[buf[12]]++;
		bin[buf[13]]++;
		bin[buf[14]]++;
		bin[buf[15]]++;
		_mm512_store_epi32(buf, cvted2);
		bin[buf[0]]++;
		bin[buf[1]]++;
		bin[buf[2]]++;
		bin[buf[3]]++;
		bin[buf[4]]++;
		bin[buf[5]]++;
		bin[buf[6]]++;
		bin[buf[7]]++;
		bin[buf[8]]++;
		bin[buf[9]]++;
		bin[buf[10]]++;
		bin[buf[11]]++;
		bin[buf[12]]++;
		bin[buf[13]]++;
		bin[buf[14]]++;
		bin[buf[15]]++;
		_mm512_store_epi32(buf, cvted3);
		bin[buf[0]]++;
		bin[buf[1]]++;
		bin[buf[2]]++;
		bin[buf[3]]++;
		bin[buf[4]]++;
		bin[buf[5]]++;
		bin[buf[6]]++;
		bin[buf[7]]++;
		bin[buf[8]]++;
		bin[buf[9]]++;
		bin[buf[10]]++;
		bin[buf[11]]++;
		bin[buf[12]]++;
		bin[buf[13]]++;
		bin[buf[14]]++;
		bin[buf[15]]++;
		_mm512_store_epi32(buf, cvted4);
		bin[buf[0]]++;
		bin[buf[1]]++;
		bin[buf[2]]++;
		bin[buf[3]]++;
		bin[buf[4]]++;
		bin[buf[5]]++;
		bin[buf[6]]++;
		bin[buf[7]]++;
		bin[buf[8]]++;
		bin[buf[9]]++;
		bin[buf[10]]++;
		bin[buf[11]]++;
		bin[buf[12]]++;
		bin[buf[13]]++;
		bin[buf[14]]++;
		bin[buf[15]]++;
#endif
#endif
	}
		__declspec(align(64)) int buf[16];
#ifdef NO_UPDATE
	_mm512_store_ps(buf, red);
#endif
	for(; i < count; i++)
			bin[(unsigned int)((data[i]-base)/width)]++;

	return 0;
}
#endif
#endif





#ifdef SIMD
#ifdef __MIC__
int hist_uniform_float_simd_unrolled 
(
 float * data, //data should be aligned
 float base, 
 float width,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_size 
 )
{
#ifdef USE_GATHER_SCATTER
#if BIN_OPT==0
		int *simd_bin = _mm_malloc((bin_size + 1)*sizeof(unsigned char)*VLEN, 64);
		memset(simd_bin, 0, (bin_size)*sizeof(unsigned char)*VLEN);
#elif BIN_OPT==1
		int *simd_bin = _mm_malloc((bin_size + 1)*sizeof(unsigned short)*VLEN, 64);
		memset(simd_bin, 0, (bin_size) *sizeof(unsigned short)*VLEN);
#else
		int *simd_bin = _mm_malloc((bin_size + 1)*sizeof(unsigned int)*VLEN, 64);
		memset(simd_bin, 0, (bin_size) *sizeof(unsigned int)*VLEN);
#endif
#endif
		_VECTOR v_base = _MM_SET1(base);
		width = 1 / width;
		_VECTOR v_width = _MM_SET1(width);
		size_t remainder;
		size_t start = 0;
		if(remainder = ((unsigned long long int)data) % (VLEN * sizeof(float)))
		{
				unsigned int i;
				start = (VLEN - (remainder / sizeof(float)));
				for(i = 0; i < start; i++)
						bin[(unsigned int)((data[i]-base)/width)]++;
		}
		unsigned int i,c;
		__m512i offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#pragma noprefetch data
		for(i = start, c=0; i+VLEN*2 < count; i+=VLEN*2,c+=2)
		{
		_VECTOR tmp =_MM_LOAD(data + i);
		_VECTOR tmp2 =_MM_LOAD(data + i + VLEN);
		_mm_prefetch((char const *)(data + i + VLEN*8), _MM_HINT_NT);
		_mm_prefetch((char const *)(data + i + VLEN*9), _MM_HINT_NT);
		tmp = _MM_SUB(tmp, v_base);
		tmp2 = _MM_SUB(tmp2, v_base);
		tmp = _MM_MUL(tmp, v_width);
		tmp2 = _MM_MUL(tmp2, v_width);
		//tmp = _MM_FLOOR(tmp);
		//tmp2 = _MM_FLOOR(tmp2);
		__m512i cvted = _mm512_cvtfxpnt_round_adjustps_epi32(
						tmp, (_MM_ROUND_MODE_ENUM)_MM_ROUND_MODE_DOWN, _MM_EXPADJ_NONE);
		__m512i cvted2 = _mm512_cvtfxpnt_round_adjustps_epi32(
						tmp2, (_MM_ROUND_MODE_ENUM)_MM_ROUND_MODE_DOWN, _MM_EXPADJ_NONE);
#ifdef USE_GATHER_SCATTER
		cvted = _mm512_fmadd_epi32(cvted, _mm512_set1_epi32(16), offsets);
		cvted2 = _mm512_fmadd_epi32(cvted2, _mm512_set1_epi32(16), offsets);
#if BIN_OPT==0
		__m512i cnts = _mm512_i32extgather_epi32(cvted, simd_bin,_MM_UPCONV_EPI32_UINT8, _MM_SCALE_1,_MM_HINT_NONE);
		cnts = _mm512_add_epi32(cnts, _mm512_set1_epi32(1));
		_mm512_i32extscatter_epi32(simd_bin, cvted, cnts, _MM_DOWNCONV_EPI32_UINT8, _MM_SCALE_1,_MM_HINT_NONE);
		__m512i cnts2 = _mm512_i32extgather_epi32(cvted2, simd_bin,_MM_UPCONV_EPI32_UINT8, _MM_SCALE_1,_MM_HINT_NONE);
		cnts2 = _mm512_add_epi32(cnts2, _mm512_set1_epi32(1));
		_mm512_i32extscatter_epi32(simd_bin, cvted2, cnts2, _MM_DOWNCONV_EPI32_UINT8, _MM_SCALE_1,_MM_HINT_NONE);
		if(c%(256)==0)
		{
				unsigned int j;
				for (j = 0; j < bin_size; j++)
				{
						bin[j]+=_mm512_reduce_add_epi32(_mm512_extload_epi32(((unsigned char *)simd_bin) + VLEN*j, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
				}
				memset(simd_bin, 0, bin_size*sizeof(unsigned char)*VLEN);
		}
#elif BIN_OPT==1
		__m512i cnts = _mm512_i32extgather_epi32(cvted, simd_bin,_MM_UPCONV_EPI32_UINT16, _MM_SCALE_2,_MM_HINT_NONE);
		cnts = _mm512_add_epi32(cnts, _mm512_set1_epi32(1));
		_mm512_i32extscatter_epi32(simd_bin, cvted, cnts, _MM_DOWNCONV_EPI32_UINT16, _MM_SCALE_2,_MM_HINT_NONE);
		__m512i cnts2 = _mm512_i32extgather_epi32(cvted2, simd_bin,_MM_UPCONV_EPI32_UINT16, _MM_SCALE_2,_MM_HINT_NONE);
		cnts2 = _mm512_add_epi32(cnts2, _mm512_set1_epi32(1));
		_mm512_i32extscatter_epi32(simd_bin, cvted2, cnts2, _MM_DOWNCONV_EPI32_UINT16, _MM_SCALE_2,_MM_HINT_NONE);
		if(c%(65536)==0)
		{
				unsigned int j;
				for (j = 0; j < bin_size; j++)
				{
						bin[j]+=_mm512_reduce_add_epi32(_mm512_extload_epi32(((unsigned short *)simd_bin) + VLEN*j, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
				}
				memset(simd_bin, 0, bin_size*sizeof(unsigned short)*VLEN);
		}
#else
		__m512i cnts = _mm512_i32gather_epi32(cvted, simd_bin, _MM_SCALE_4);
		cnts = _mm512_add_epi32(cnts, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(simd_bin, cvted, cnts, _MM_SCALE_4);
		__m512i cnts2 = _mm512_i32gather_epi32(cvted2, simd_bin, _MM_SCALE_4);
		cnts2 = _mm512_add_epi32(cnts2, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(simd_bin, cvted2, cnts2, _MM_SCALE_4);
#endif
#else
		__declspec(align(64)) int buf[16];
		__declspec(align(64)) int buf2[16];
		_mm512_store_epi32(buf, cvted);
		bin[buf[0]]++;
		bin[buf[1]]++;
		bin[buf[2]]++;
		bin[buf[3]]++;
		bin[buf[4]]++;
		bin[buf[5]]++;
		bin[buf[6]]++;
		bin[buf[7]]++;
		bin[buf[8]]++;
		bin[buf[9]]++;
		bin[buf[10]]++;
		bin[buf[11]]++;
		bin[buf[12]]++;
		bin[buf[13]]++;
		bin[buf[14]]++;
		bin[buf[15]]++;
		_mm512_store_epi32(buf2, cvted2);
		bin[buf2[0]]++;
		bin[buf2[1]]++;
		bin[buf2[2]]++;
		bin[buf2[3]]++;
		bin[buf2[4]]++;
		bin[buf2[5]]++;
		bin[buf2[6]]++;
		bin[buf2[7]]++;
		bin[buf2[8]]++;
		bin[buf2[9]]++;
		bin[buf2[10]]++;
		bin[buf2[11]]++;
		bin[buf2[12]]++;
		bin[buf2[13]]++;
		bin[buf2[14]]++;
		bin[buf2[15]]++;
#endif
	}
#ifdef USE_GATHER_SCATTER
		unsigned int j, k;
		for (j = 0; j < bin_size; j++)
		{
#if BIN_OPT==0
				bin[j]+=_mm512_reduce_add_epi32(_mm512_extload_epi32(((unsigned char *)simd_bin) + VLEN*j, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
#elif BIN_OPT==1
				bin[j]+=_mm512_reduce_add_epi32(_mm512_extload_epi32(((unsigned short *)simd_bin) + VLEN*j, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
#else
				bin[j] += _mm512_reduce_add_epi32(((__m512i *)simd_bin)[j]);
#endif
		}
		_mm_free(simd_bin);
#endif
	for(; i < count; i++)
			bin[(unsigned int)((data[i]-base)/width)]++;

	return 0;
}
#endif
#endif


#ifdef SIMD
int hist_uniform_float_simd 
(
 float * data, //data should be aligned
 float base, 
 float width,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_size 
 )
{
#ifdef __MIC__
#if UNROLL_N==2
		return hist_uniform_float_simd_unrolled(data, base, width, count, bin, bin_size);
#elif UNROLL_N==4
		return hist_uniform_float_simd_unrolled_4(data, base, width, count, bin, bin_size);
#endif
#endif
#ifdef USE_GATHER_SCATTER
#if BIN_OPT==0
		int *simd_bin = _mm_malloc((bin_size + 1)*sizeof(unsigned char)*VLEN, 64);
		memset(simd_bin, 0, (bin_size)*sizeof(unsigned char)*VLEN);
#elif BIN_OPT==1
		int *simd_bin = _mm_malloc((bin_size + 1)*sizeof(unsigned short)*VLEN, 64);
		memset(simd_bin, 0, (bin_size) *sizeof(unsigned short)*VLEN);
#else
		int *simd_bin = _mm_malloc((bin_size + 1)*sizeof(unsigned int)*VLEN, 64);
		memset(simd_bin, 0, (bin_size) *sizeof(unsigned int)*VLEN);
#endif
#endif
		_VECTOR v_base = _MM_SET1(base);
		width = 1 / width;
		_VECTOR v_width = _MM_SET1(width);
		size_t remainder;
		size_t start = 0;
		if(remainder = ((unsigned long long int)data) % (VLEN * sizeof(float)))
		{
				unsigned int i;
				start = (VLEN - (remainder / sizeof(float)));
				for(i = 0; i < start; i++)
						bin[(unsigned int)((data[i]-base)/width)]++;
		}
		unsigned int i,c;
#ifdef NO_UPDATE
		int tmpForReduction = 0;
#endif
#ifdef __MIC__
		__m512i offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#endif
//#pragma unroll(UNROLL_N)
		for(i = start, c=0; i+VLEN < count; i+=VLEN,c++)
		{
#ifdef SSE
				/*
		_VECTOR tmp =_MM_LOAD(data + i);
		_VECTOR tmp =_MM_LOAD(data + i);
		tmp = _MM_SUB(tmp, v_base);
		tmp = _MM_MUL(tmp, v_width);
		tmp = _MM_FLOOR(tmp);
		__m128i cvted = _mm_cvtps_epi32(tmp);
		//__declspec(align(64)) int buf[4];
		//_mm_store_ps((float *)buf, (__m128)cvted);
		bin[buf[0]]++;
		bin[buf[1]]++;
		bin[buf[2]]++;
		bin[buf[3]]++;
		bin[_mm_extract_epi32(cvted, 0)]++;
		bin[_mm_extract_epi32(cvted, 1)]++;
		bin[_mm_extract_epi32(cvted, 2)]++;
		bin[_mm_extract_epi32(cvted, 3)]++;
		*/
#elif AVX
#else  // __MIC__
		_VECTOR tmp =_MM_LOAD(data + i);
		tmp = _MM_SUB(tmp, v_base);
		tmp = _MM_MUL(tmp, v_width);
		tmp = _MM_FLOOR(tmp);

		__m512i cvted = _mm512_cvtfxpnt_round_adjustps_epi32(
						tmp, (_MM_ROUND_MODE_ENUM)_MM_ROUND_MODE_DOWN, _MM_EXPADJ_NONE);
#ifdef USE_GATHER_SCATTER
		cvted = _mm512_fmadd_epi32(cvted, _mm512_set1_epi32(16), offsets);
		__m512i cnts = _mm512_i32gather_epi32(cvted, simd_bin, _MM_SCALE_4);
		cnts = _mm512_add_epi32(cnts, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(simd_bin, cvted, cnts, _MM_SCALE_4);
#endif
		__declspec(align(64)) int buf[16];
		_mm512_store_epi32(buf, cvted);
		bin[buf[0]]++;
		bin[buf[1]]++;
		bin[buf[2]]++;
		bin[buf[3]]++;
		bin[buf[4]]++;
		bin[buf[5]]++;
		bin[buf[6]]++;
		bin[buf[7]]++;
		bin[buf[8]]++;
		bin[buf[9]]++;
		bin[buf[10]]++;
		bin[buf[11]]++;
		bin[buf[12]]++;
		bin[buf[13]]++;
		bin[buf[14]]++;
		bin[buf[15]]++;
#endif
	}
#ifdef NO_UPDATE
		bin[0] += tmpForReduction;
#endif
#ifdef USE_GATHER_SCATTER
		unsigned int j, k;
		for (j = 0; j < bin_size; j++)
		{
#if BIN_OPT==0
				bin[j]+=_mm512_reduce_add_epi32(_mm512_extload_epi32(((unsigned char *)simd_bin) + VLEN*j, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
#elif BIN_OPT==1
				bin[j]+=_mm512_reduce_add_epi32(_mm512_extload_epi32(((unsigned short *)simd_bin) + VLEN*j, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE));
#else
				bin[j] += _mm512_reduce_add_epi32(((__m512i *)simd_bin)[j]);
#endif
		}
		_mm_free(simd_bin);
#endif
	for(; i < count; i++)
			bin[(unsigned int)((data[i]-base)/width)]++;

	return 0;
}
#endif


int hist_uniform_float 
(
 float * data,
 float base, 
 float width,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_size
 )
{
				int i;
				float tmp=0;
#pragma unroll(UNROLL_N)
				for(i = 0; i < count; i++)
				{
#ifdef NO_UPDATE
						tmp += (data[i]);
						//tmp += (unsigned int)((data[i] - base) /width);
#else
								//bin[(unsigned int)(log((data[i] - base)/width + 1) + 1)]++;
								bin[(unsigned int)( (data[i] - base)/width) ]++;
#endif
								//unsigned int x = (unsigned int)((data[i] - base)/width);
				}
				
#ifdef NO_UPDATE
				bin[0]=(int)tmp;
#endif
				return 0;
}
int hist_uniform_short 
(
 unsigned short * data,
 unsigned short base, 
 unsigned short width,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_size
 )
{
				int i;
#pragma unroll(16)
				for(i = 0; i < count; i++)
				{
								bin[((data[i])-base)]++;
								//unsigned int x = (unsigned int)((data[i] - base)/width);
				}
				return 0;
}
int hist_uniform_short_ipp 
(
 unsigned short * data,
 unsigned short base, 
 unsigned short width,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_size
 )
{


#ifdef IPP
#ifndef __MIC__
#ifndef SIMD
				Ipp32s *levels = (Ipp32s *)ippMalloc(sizeof(int) * bin_size) ;
				int i;
		Ipp16u * img = (Ipp16u *)data;
		IppiSize imgSize = {count , 1};
		int t=0;

		IppStatus t2 = ippiHistogramEven_16u_C1R(img, count  * sizeof(Ipp16u), imgSize, (Ipp32s*)bin, (Ipp32s*)levels,bin_size, base, base + width * bin_size);
		
		
	//	printf("still alive!\n");
		ippFree(levels);

#endif
#endif
#endif
				return 0;
}

int hist_uniform_float_atomic 
(
 float * data,
 float base, 
 float width,
 unsigned int count,
 unsigned int * bin 
 )

{
				unsigned int i;
#ifndef MANUAL_UNROLL

#pragma unroll(UNROLL_N)
				for(i = 0; i < count; i++)
				{
								__sync_fetch_and_add(&bin[(unsigned int)((data[i] - base)/width)], 1);
				}
#else
				for(i = 0; i < count; i += 8)	//for the simplicty of implementation, set unroll=8
				{
								unsigned int i1 = (unsigned int)((data[i]-base)/width);
								unsigned int i2 = (unsigned int)((data[i+1]-base)/width);
								unsigned int i3 = (unsigned int)((data[i+2]-base)/width);
								unsigned int i4 = (unsigned int)((data[i+3]-base)/width);
								unsigned int i5 = (unsigned int)((data[i+4]-base)/width);
								unsigned int i6 = (unsigned int)((data[i+5]-base)/width);
								unsigned int i7 = (unsigned int)((data[i+6]-base)/width);
								unsigned int i8 = (unsigned int)((data[i+7]-base)/width);
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i1])
																:"m" (bin[i1]));	
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i2])
																:"m" (bin[i2]));	
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i3])
																:"m" (bin[i3]));	
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i4])
																:"m" (bin[i4]));	
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i5])
																:"m" (bin[i5]));	
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i6])
																:"m" (bin[i6]));	
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i7])
																:"m" (bin[i7]));	
								__asm__ __volatile__(
																LOCK "incl %0"
																:"=m" (bin[i8])
																:"m" (bin[i8]));	

				}
#endif
				return 0;
}

