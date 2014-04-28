#include "linear.h"
#include "simd.h"

#define UPDATE_TYPE 1 // 0 : store&read, 1 : extract
#define MANUAL_UNROLL
//#define BIN_SIMD

#ifdef NO_UPDATE
#ifdef __MIC__
#define UPDATE(bin , v, buf) \
			_mm512_store_epi32(buf, v);\
			tmpForReduction += buf[0];\
			tmpForReduction += buf[1];\
			tmpForReduction += buf[2];\
			tmpForReduction += buf[3];\
			tmpForReduction += buf[4];\
			tmpForReduction += buf[5];\
			tmpForReduction += buf[6];\
			tmpForReduction += buf[7];\
			tmpForReduction += buf[8];\
			tmpForReduction += buf[9];\
			tmpForReduction += buf[10];\
			tmpForReduction += buf[11];\
			tmpForReduction += buf[12];\
			tmpForReduction += buf[13];\
			tmpForReduction += buf[14];\
			tmpForReduction += buf[15];\
#elif AVX
#define UPDATE(bin , c1, c2) \
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 0)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 1)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 2)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 3)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c2, 0)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c2, 1)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c2, 2)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c2, 3)++;

#elif SSE
#define UPDATE(bin , c1) \
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 0)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 1)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 2)++;\
		tmpForReduction +=  (unsigned)-_mm_extract_epi32(c1, 3)++;
#endif

#else
#ifdef __MIC__
#define UPDATE(bin , v, buf) \
			_mm512_store_epi32(buf, v);\
			bin[buf[0]- 1]++;\
			bin[buf[1]- 1]++;\
			bin[buf[2]- 1]++;\
			bin[buf[3]- 1]++;\
			bin[buf[4]- 1]++;\
			bin[buf[5]- 1]++;\
			bin[buf[6]- 1]++;\
			bin[buf[7]- 1]++;\
			bin[buf[8]- 1]++;\
			bin[buf[9]- 1]++;\
			bin[buf[10]- 1]++;\
			bin[buf[11]- 1]++;\
			bin[buf[12]- 1]++;\
			bin[buf[13]- 1]++;\
			bin[buf[14]- 1]++;\
			bin[buf[15]- 1]++;
#elif AVX
#define UPDATE(bin , c1, c2) \
		bin[((unsigned)-_mm_extract_epi32(c1, 0)- 1)]++;\
		bin[((unsigned)-_mm_extract_epi32(c1, 1)- 1)]++;\
		bin[((unsigned)-_mm_extract_epi32(c1, 2)- 1)]++;\
		bin[((unsigned)-_mm_extract_epi32(c1, 3)- 1)]++;\
		bin[((unsigned)-_mm_extract_epi32(c2, 0)- 1)]++;\
		bin[((unsigned)-_mm_extract_epi32(c2, 1)- 1)]++;\
		bin[((unsigned)-_mm_extract_epi32(c2, 2)- 1)]++;\
		bin[((unsigned)-_mm_extract_epi32(c2, 3)- 1)]++;

#elif SSE
#define UPDATE(bin , c1) \
		bin[(unsigned)-_mm_extract_epi32(c1, 0)- 1]++;\
		bin[(unsigned)-_mm_extract_epi32(c1, 1)- 1]++;\
		bin[(unsigned)-_mm_extract_epi32(c1, 2)- 1]++;\
		bin[(unsigned)-_mm_extract_epi32(c1, 3)- 1]++;
#endif
#endif
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
int hist_linear_float_simd
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 int * bin,
 unsigned int bin_count
)
{
		size_t remainder;
		size_t start = 0;
		unsigned int i;
		int j;
		if(remainder = ((unsigned long long int)data) % (VLEN * sizeof(float)))
		{
				unsigned int i;
				start = (VLEN - (remainder / sizeof(float)));
				for(i = 0; i < start; i++)
				{
						float d = data[i];
						for(j = bin_count-1; d < boundary[j]; j--);
						bin[j]++;
				}
		}


		unsigned int n = bin_count;
#ifdef BIN_SIMD
		int VLENbin_count;
		for(VLENbin_count = 0; VLENbin_count < bin_count - 1; VLENbin_count+=VLEN);
		_VECTOR * v_boundary = (_VECTOR *)_mm_malloc(sizeof(_VECTOR) * VLENbin_count / VLEN , 64);
		float * tmp_boundary = (float *)_mm_malloc(sizeof(float) * VLENbin_count , 64);
		for(i = 1; i < bin_count+1; i++)tmp_boundary[i-1]=boundary[i];
		for( ; i<VLENbin_count + 1 ; i++) tmp_boundary[i-1] = boundary[bin_count];
		for(i = 0; i < VLENbin_count; i+=VLEN) v_boundary[i/VLEN] = _MM_LOAD((const float *)tmp_boundary + i);
		_mm_free(tmp_boundary);
	//	for(i = 0; i < VLENbin_count; i++) printf("%f\n", tmp_boundary[i]);
		int n2 = VLENbin_count / VLEN;


#else
		_VECTOR * v_boundary = (_VECTOR *)_mm_malloc(sizeof(_VECTOR) * (n + 1),64);

		for( i = 0; i < n+1; i++)
		{
			v_boundary[i] = _MM_SET1(boundary[i]);
		}

#endif
#ifdef NO_UPDATE
		int tmpForReduction = 0;
#endif


#ifdef __MIC__
		__m512i t = _mm512_set1_epi32(1);
#endif

#ifdef MANUAL_UNROLL // 4 way manually unrolled 
		for(i = start; i + VLEN*4 -1 < count; i+=VLEN * 4)
		{
			_VECTOR tmp1 = _MM_LOAD(data + i);
			_VECTOR tmp2 = _MM_LOAD(data + i + VLEN);
			_VECTOR tmp3 = _MM_LOAD(data + i + VLEN*2);
			_VECTOR tmp4 = _MM_LOAD(data + i + VLEN*3);
#ifdef SSE
			__m128i c1 = _mm_set1_epi32(0);
			__m128i c2 = _mm_set1_epi32(0);
			__m128i c3 = _mm_set1_epi32(0);
			__m128i c4 = _mm_set1_epi32(0);
#pragma unroll(UNROLL_N)
			for(j = 1; j < n+1; j++)
			{
					__m128i t1 = _mm_castps_si128( _mm_cmpge_ps(tmp1, v_boundary[j]));
					__m128i t2 = _mm_castps_si128( _mm_cmpge_ps(tmp2, v_boundary[j]));
					__m128i t3 = _mm_castps_si128( _mm_cmpge_ps(tmp3, v_boundary[j]));
					__m128i t4 = _mm_castps_si128( _mm_cmpge_ps(tmp4, v_boundary[j]));
					c1 = _mm_add_epi32(c1, t1);
					c2 = _mm_add_epi32(c2, t2);
					c3 = _mm_add_epi32(c3, t3);
					c4 = _mm_add_epi32(c4, t4);
			}
			UPDATE(bin, c1);
			UPDATE(bin, c2);
			UPDATE(bin, c3);
			UPDATE(bin, c4);

#elif AVX
			__m128i c11 = _mm_set1_epi32(0);
			__m128i c12 = _mm_set1_epi32(0);
			__m128i c21 = _mm_set1_epi32(0);
			__m128i c22 = _mm_set1_epi32(0);
			__m128i c31 = _mm_set1_epi32(0);
			__m128i c32 = _mm_set1_epi32(0);
			__m128i c41 = _mm_set1_epi32(0);
			__m128i c42 = _mm_set1_epi32(0);
#pragma unroll(UNROLL_N)
			for(j = 1; j < n+1; j++)
			{
				__m256i t1 = _mm256_castps_si256( _mm256_cmp_ps(tmp1, v_boundary[j], _CMP_GE_OS));
				__m256i t2 = _mm256_castps_si256( _mm256_cmp_ps(tmp2, v_boundary[j], _CMP_GE_OS));
				c11 = _mm_add_epi32(c11, _mm256_extractf128_si256(t1, 0));
				c12 = _mm_add_epi32(c12, _mm256_extractf128_si256(t1, 1));
				__m256i t3 = _mm256_castps_si256( _mm256_cmp_ps(tmp3, v_boundary[j], _CMP_GE_OS));
				c21 = _mm_add_epi32(c21, _mm256_extractf128_si256(t2, 0));
				c22 = _mm_add_epi32(c22, _mm256_extractf128_si256(t2, 1));
				__m256i t4 = _mm256_castps_si256( _mm256_cmp_ps(tmp4, v_boundary[j], _CMP_GE_OS));
				c31 = _mm_add_epi32(c31, _mm256_extractf128_si256(t3, 0));
				c32 = _mm_add_epi32(c32, _mm256_extractf128_si256(t3, 1));
				c41 = _mm_add_epi32(c41, _mm256_extractf128_si256(t4, 0));
				c42 = _mm_add_epi32(c42, _mm256_extractf128_si256(t4, 1));
			}
		  UPDATE(bin, c11, c12);
		  UPDATE(bin, c21, c22);
		  UPDATE(bin, c31, c32);
		  UPDATE(bin, c41, c42);

#elif __MIC__
			__m512i c1 = _mm512_set1_epi32(0);
			__m512i c2 = _mm512_set1_epi32(0);
			__m512i c3 = _mm512_set1_epi32(0);
			__m512i c4 = _mm512_set1_epi32(0);
#pragma unroll(UNROLL_N)
			for(j = 1; j < n+1; j++)
			{
				c1 = _mm512_mask_add_epi32(c1, _mm512_cmp_ps_mask(tmp1, v_boundary[j], _CMP_GE_OS), c1, t);
				c2 = _mm512_mask_add_epi32(c2, _mm512_cmp_ps_mask(tmp2, v_boundary[j], _CMP_GE_OS), c2, t);
				c3 = _mm512_mask_add_epi32(c3, _mm512_cmp_ps_mask(tmp3, v_boundary[j], _CMP_GE_OS), c3, t);
				c4 = _mm512_mask_add_epi32(c4, _mm512_cmp_ps_mask(tmp4, v_boundary[j], _CMP_GE_OS), c4, t);
			}
			__declspec(align(64)) int buf[16];
			UPDATE(bin, c1, buf);
			UPDATE(bin, c2, buf);
			UPDATE(bin, c3, buf);
			UPDATE(bin, c4, buf);
#endif

		}
//#endif
#else

		for(i = start; i+VLEN < count; i+=VLEN)
		{
			_VECTOR tmp = _MM_LOAD(data + i);
#ifdef SSE
			__m128i c1 = _mm_set1_epi32(0);
#pragma unroll(UNROLL_N)
			for(j = 1; j < n+1; j++)
			{
					__m128i t = _mm_castps_si128( _mm_cmpge_ps(tmp, v_boundary[j]));
					c1 = _mm_add_epi32(c1, t);
			}
			UPDATE(bin, c1);

#elif AVX
			__m128i c1 = _mm_set1_epi32(0);
			__m128i c2 = _mm_set1_epi32(0);
#pragma unroll(UNROLL_N)
			for(j = 1; j < n+1; j++)
			{
				__m256i t = _mm256_castps_si256( _mm256_cmp_ps(tmp, v_boundary[j], _CMP_GE_OS));
				c1 = _mm_add_epi32(c1, _mm256_extractf128_si256(t, 0));
				c2 = _mm_add_epi32(c2, _mm256_extractf128_si256(t, 1));
			}
		  UPDATE(bin, c1, c2);

#elif __MIC__
			__m512i c = _mm512_set1_epi32(0);
#pragma unroll(UNROLL_N)
			for(j = 1; j < n+1; j++)
			{
				c = _mm512_mask_add_epi32(c, _mm512_cmp_ps_mask(tmp, v_boundary[j], _CMP_GE_OS), c, t);
			}
			__declspec(align(64)) int buf[16];
			UPDATE(bin, c, buf);
#endif

		}
#endif

#ifdef NO_UPDATE
		bin[0] += tmpForReduction;
#endif

		for(; i < count; i++)
		{
			float d = data[i];
			for(j = bin_count-1; d < boundary[j]; j--);
			bin[j]++;
		}
		_mm_free(v_boundary);
		return 0;
}
#endif

int hist_linear_float 
(
 float * data, 
 float * boundary,
 int count,
 int * bin,
 int bin_count
)
{
		int i, j;
		for(i = 0; i < count; i++){
			float d = data[i];
//#pragma unroll(UNROLL_N)
			for(j = bin_count-1; d < boundary[j]; j--);
			bin[j]++;
		}
		return 0;
}
