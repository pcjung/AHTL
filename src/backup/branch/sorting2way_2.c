#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef SIMD
#include "simd.h"
#endif

#ifdef SIMD

#ifdef AVX
#define BMCOEFF 16
#elif SSE
#define BMCOEFF 16
#elif __MIC__
#define BMCOEFF 4
#endif
#ifdef __MIC__

#define TRANSPOSE_LOAD(A, B, C, D, X, src) \
		A = _mm512_mask_loadunpacklo_ps(A, 0x1111, (float *)src+(X+0)*16+ 0); \
B = _mm512_mask_loadunpacklo_ps(B, 0x1111, (float *)src+(X+0)*16+ 4); \
C = _mm512_mask_loadunpacklo_ps(C, 0x1111, (float *)src+(X+0)*16+ 8); \
D = _mm512_mask_loadunpacklo_ps(D, 0x1111, (float *)src+(X+0)*16+12); \
A = _mm512_mask_loadunpacklo_ps(A, 0x2222, (float *)src+(X+1)*16+ 0); \
B = _mm512_mask_loadunpacklo_ps(B, 0x2222, (float *)src+(X+1)*16+ 4); \
C = _mm512_mask_loadunpacklo_ps(C, 0x2222, (float *)src+(X+1)*16+ 8); \
D = _mm512_mask_loadunpacklo_ps(D, 0x2222, (float *)src+(X+1)*16+12); \
A = _mm512_mask_loadunpacklo_ps(A, 0x4444, (float *)src+(X+2)*16+ 0); \
B = _mm512_mask_loadunpacklo_ps(B, 0x4444, (float *)src+(X+2)*16+ 4); \
C = _mm512_mask_loadunpacklo_ps(C, 0x4444, (float *)src+(X+2)*16+ 8); \
D = _mm512_mask_loadunpacklo_ps(D, 0x4444, (float *)src+(X+2)*16+12); \
A = _mm512_mask_loadunpacklo_ps(A, 0x8888, (float *)src+(X+3)*16+ 0); \
B = _mm512_mask_loadunpacklo_ps(B, 0x8888, (float *)src+(X+3)*16+ 4); \
C = _mm512_mask_loadunpacklo_ps(C, 0x8888, (float *)src+(X+3)*16+ 8); \
D = _mm512_mask_loadunpacklo_ps(D, 0x8888, (float *)src+(X+3)*16+12); \


#define TRANSPOSE_STORE(A, B, C, D, X, Dst) \
		_mm512_mask_packstorelo_ps((float *)Dst+(X+0)*16+ 0, 0x000f, A); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+0)*16+ 4, 0x000f, B); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+0)*16+ 8, 0x000f, C); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+0)*16+12, 0x000f, D); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+1)*16+ 0, 0x00f0, A); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+1)*16+ 4, 0x00f0, B); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+1)*16+ 8, 0x00f0, C); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+1)*16+12, 0x00f0, D); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+2)*16+ 0, 0x0f00, A); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+2)*16+ 4, 0x0f00, B); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+2)*16+ 8, 0x0f00, C); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+2)*16+12, 0x0f00, D); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+3)*16+ 0, 0xf000, A); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+3)*16+ 4, 0xf000, B); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+3)*16+ 8, 0xf000, C); \
_mm512_mask_packstorelo_ps((float *)Dst+(X+3)*16+12, 0xf000, D); \

#define TRANSPOSE_ARRAY(a) { \
		__m512 V0 = _mm512_undefined_ps(); \
		__m512 V1 = _mm512_undefined_ps(); \
		__m512 V2 = _mm512_undefined_ps(); \
		__m512 V3 = _mm512_undefined_ps(); \
		__m512 V4 = _mm512_undefined_ps(); \
		__m512 V5 = _mm512_undefined_ps(); \
		__m512 V6 = _mm512_undefined_ps(); \
		__m512 V7 = _mm512_undefined_ps(); \
		__m512 V8 = _mm512_undefined_ps(); \
		__m512 V9 = _mm512_undefined_ps(); \
		__m512 V10 = _mm512_undefined_ps(); \
		__m512 V11 = _mm512_undefined_ps(); \
		__m512 V12 = _mm512_undefined_ps(); \
		__m512 V13 = _mm512_undefined_ps(); \
		__m512 V14 = _mm512_undefined_ps(); \
		__m512 V15 = _mm512_undefined_ps(); \
		TRANSPOSE_LOAD( V0, V1, V2, V3, 0, a); \
		TRANSPOSE_LOAD( V4, V5, V6, V7, 4, a); \
		TRANSPOSE_LOAD( V8, V9, V10, V11, 8, a); \
		TRANSPOSE_LOAD( V12, V13, V14, V15, 12, a); \
		TRANSPOSE_STORE( V0, V4, V8, V12, 0, a); \
		TRANSPOSE_STORE( V1, V5, V9, V13, 4, a); \
		TRANSPOSE_STORE( V2, V6, V10, V14, 8, a); \
		TRANSPOSE_STORE( V3, V7, V11, V15, 12, a); \
}

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
static void printv(__m512 v, char * str)
{
		int i;
		__declspec(align(64)) float tmp[16];
		printf("%s:", str);
		_mm512_store_ps(tmp, v);
		for(i=0; i < 16; i++)
		{
				tmp[0] = tmp[i];
				printf("[%d]=%f ", i, tmp[0]);
		}
		printf("\n");
}
#elif AVX
static void printv(__m256 v, char * str)
{
		int i;
		__declspec(align(64)) float tmp[8];
		printf("%s:", str);
		_mm256_store_ps(tmp, v);
		for(i=0; i < 8; i++)
		{
				tmp[0] = tmp[i];
				printf("[%d]=%f ", i, tmp[0]);
		}
		printf("\n");
}
#else //SSE
static void printv(__m128 v, char * str)
{
		int i;
		__declspec(align(64)) float tmp[4];
		printf("%s:", str);
		_mm_store_ps(tmp, v);
		for(i=0; i < 4; i++)
		{
				tmp[0] = tmp[i];
				printf("[%d]=%f ", i, tmp[0]);
		}
		printf("\n");
}
#endif


static void printArray(float * a, int count)
{
		int i;
		for(i=0;i<count;i++)printf("%d %f\n", i, a[i]);
}


#define CMOVZ(t, old, new) \
__asm__ ("testl %1, %1\n\t" \ 
"cmovz %2, %0\n\t" \
:"+r"(old)\ 
:"r"(t), "r"(new) \
); 


#ifdef SSE
static int merge_4way(float ** bufs, int BLOCK_SIZE)
{
				float BIG = (float) BLOCK_SIZE;
				int step, chunk_size;
				for(step = 0, chunk_size=VLEN; chunk_size * 4 <= BLOCK_SIZE; step++, chunk_size *= 2)
				{
								//printf("%d\n", step);

								float * from, *to;
								from = bufs[step % 2];
								to = bufs[(step + 1) % 2];
								int chunk1, chunk2;
								int chunk3, chunk4;
								int chunk5, chunk6;
								int chunk7, chunk8;
								for(
																chunk1 = 0, chunk2 = chunk1 + chunk_size, 
																chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
																chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
																chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size;
																chunk8 < BLOCK_SIZE; 
																chunk1 += chunk_size * 8, chunk2 = chunk1 + chunk_size,
																chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
																chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
																chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size
									 )
								{
#ifdef DEBUG_SORT
												printf("merge chunk : %d %d %d %d %d %d %d %d\n", chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, chunk7, chunk8);
#endif
												int idx1= chunk1, idx2, out1;
												int idx3= chunk3, idx4, out3;
												int idx5= chunk5, idx6, out5;
												int idx7= chunk7, idx8, out7;
												_VECTOR tmp1, tmp2 = _MM_LOAD(from + idx1); 
												_VECTOR tmp3, tmp4 = _MM_LOAD(from + idx3); 
												_VECTOR tmp5, tmp6 = _MM_LOAD(from + idx5); 
												_VECTOR tmp7, tmp8 = _MM_LOAD(from + idx7); 
												float last1 = *(from + idx1 + VLEN), last2;
												float last3 = *(from + idx3 + VLEN), last4;
												float last5 = *(from + idx5 + VLEN), last6;
												float last7 = *(from + idx7 + VLEN), last8;
												idx1 += VLEN;
												idx3 += VLEN;
												idx5 += VLEN;
												idx7 += VLEN;
												int get_from_front1 = 0;
												int get_from_front3 = 0;
												int get_from_front5 = 0;
												int get_from_front7 = 0;
												int chunk1end = chunk1 + chunk_size;
												int chunk2end = chunk2 + chunk_size;
												int chunk3end = chunk3 + chunk_size;
												int chunk4end = chunk4 + chunk_size;
												int chunk5end = chunk5 + chunk_size;
												int chunk6end = chunk6 + chunk_size;
												int chunk7end = chunk7 + chunk_size;
												int chunk8end = chunk8 + chunk_size;

												float chunk1tmp = from[chunk1end];
												float chunk2tmp = from[chunk2end];
												float chunk3tmp = from[chunk3end];
												float chunk4tmp = from[chunk4end];
												float chunk5tmp = from[chunk5end];
												float chunk6tmp = from[chunk6end];
												float chunk7tmp = from[chunk7end];
												float chunk8tmp = from[chunk8end];

												idx2 = chunk2; out1 = chunk1;
												idx4 = chunk4; out3 = chunk3;
												idx6 = chunk6; out5 = chunk5;
												idx8 = chunk8; out7 = chunk7;

												tmp1 = _MM_LOAD(from + idx2);
												tmp3 = _MM_LOAD(from + idx4);
												tmp5 = _MM_LOAD(from + idx6);
												tmp7 = _MM_LOAD(from + idx8);

												//merge tmp1, tmp2 here
												//reverse t2
#ifdef DEBUG_SORT
												printf("merge %d tmp1\n", idx1);
												printf("merge %d tmp2\n", idx2);
												printf("merge %d tmp3\n", idx3);
												printf("merge %d tmp4\n", idx4);
												printf("merge %d tmp5\n", idx5);
												printf("merge %d tmp6\n", idx6);
												printf("merge %d tmp7\n", idx7);
												printf("merge %d tmp8\n", idx8);

												printv(tmp1, "tmp1");
												printv(tmp2, "tmp2");
												printv(tmp3, "tmp3");
												printv(tmp4, "tmp4");
												printv(tmp5, "tmp5");
												printv(tmp6, "tmp6");
												printv(tmp7, "tmp7");
												printv(tmp8, "tmp8");
#endif
												tmp2 = _mm_shuffle_ps(tmp2, tmp2, 0x1b);
												tmp4 = _mm_shuffle_ps(tmp4, tmp4, 0x1b);
												tmp6 = _mm_shuffle_ps(tmp6, tmp6, 0x1b);
												tmp8 = _mm_shuffle_ps(tmp8, tmp8, 0x1b);

#ifdef DEBUG_SORT
												printv(tmp1, "tmp1");
												printv(tmp2, "tmp2");
#endif
												__m128 t1, t2;
												__m128 t3, t4;
												__m128 t5, t6;
												__m128 t7, t8;

												t1 = _MM_MIN(tmp1, tmp2); 
												t2 = _MM_MAX(tmp1, tmp2); 
												t3 = _MM_MIN(tmp3, tmp4); 
												t4 = _MM_MAX(tmp3, tmp4); 
												t5 = _MM_MIN(tmp5, tmp6); 
												t6 = _MM_MAX(tmp5, tmp6); 
												t7 = _MM_MIN(tmp7, tmp8); 
												t8 = _MM_MAX(tmp7, tmp8); 

												tmp1 = _mm_shuffle_ps(t1, t2, 0x1b);
												tmp2 = _mm_shuffle_ps(t1, t2, 0xb1);
												tmp3 = _mm_shuffle_ps(t3, t4, 0x1b);
												tmp4 = _mm_shuffle_ps(t3, t4, 0xb1);
												tmp5 = _mm_shuffle_ps(t5, t6, 0x1b);
												tmp6 = _mm_shuffle_ps(t5, t6, 0xb1);
												tmp7 = _mm_shuffle_ps(t7, t8, 0x1b);
												tmp8 = _mm_shuffle_ps(t7, t8, 0xb1);

#ifdef DEBUG_SORT
												printv(tmp1, "tmp1");
												printv(tmp2, "tmp2");
#endif
												t1 = _MM_MIN(tmp1, tmp2); 
												t2 = _MM_MAX(tmp1, tmp2); 
												t3 = _MM_MIN(tmp3, tmp4); 
												t4 = _MM_MAX(tmp3, tmp4); 
												t5 = _MM_MIN(tmp5, tmp6); 
												t6 = _MM_MAX(tmp5, tmp6); 
												t7 = _MM_MIN(tmp7, tmp8); 
												t8 = _MM_MAX(tmp7, tmp8); 

												tmp1 = _mm_blend_ps(t1, t2, 0xa);
												tmp2 = _mm_blend_ps(t2, t1, 0xa);
												tmp2 = _mm_shuffle_ps(tmp2, tmp2, 0xb1);
												tmp3 = _mm_blend_ps(t3, t4, 0xa);
												tmp4 = _mm_blend_ps(t4, t3, 0xa);
												tmp4 = _mm_shuffle_ps(tmp4, tmp4, 0xb1);
												tmp5 = _mm_blend_ps(t5, t6, 0xa);
												tmp6 = _mm_blend_ps(t6, t5, 0xa);
												tmp6 = _mm_shuffle_ps(tmp6, tmp6, 0xb1);
												tmp7 = _mm_blend_ps(t7, t8, 0xa);
												tmp8 = _mm_blend_ps(t8, t7, 0xa);
												tmp8 = _mm_shuffle_ps(tmp8, tmp8, 0xb1);

#ifdef DEBUG_SORT
												printv(tmp1, "tmp1");
												printv(tmp2, "tmp2");
#endif
												t1 = _MM_MIN(tmp1, tmp2); 
												t2 = _MM_MAX(tmp1, tmp2); 
												t3 = _MM_MIN(tmp3, tmp4); 
												t4 = _MM_MAX(tmp3, tmp4); 
												t5 = _MM_MIN(tmp5, tmp6); 
												t6 = _MM_MAX(tmp5, tmp6); 
												t7 = _MM_MIN(tmp7, tmp8); 
												t8 = _MM_MAX(tmp7, tmp8); 

												tmp1 = _mm_unpacklo_ps(t1, t2);
												tmp2 = _mm_unpackhi_ps(t1, t2);
												tmp3 = _mm_unpacklo_ps(t3, t4);
												tmp4 = _mm_unpackhi_ps(t3, t4);
												tmp5 = _mm_unpacklo_ps(t5, t6);
												tmp6 = _mm_unpackhi_ps(t5, t6);
												tmp7 = _mm_unpacklo_ps(t7, t8);
												tmp8 = _mm_unpackhi_ps(t7, t8);
#ifdef DEBUG_SORT

												printv(tmp1, "tmp1");
												printv(tmp2, "tmp2");
												printv(tmp3, "tmp3");
												printv(tmp4, "tmp4");
												printv(tmp5, "tmp5");
												printv(tmp6, "tmp6");
												printv(tmp7, "tmp7");
												printv(tmp8, "tmp8");

												printf("store at %d\n", out1);
												printf("store at %d\n", out3);
												printf("store at %d\n", out5);
												printf("store at %d\n\n", out7);
												printf("\n\n");
#endif
												_MM_STORE(to + out1, tmp1);
												_MM_STORE(to + out3, tmp3);
												_MM_STORE(to + out5, tmp5);
												_MM_STORE(to + out7, tmp7);

												idx1 += (get_from_front1)*VLEN;
												idx2 += (!get_from_front1)*VLEN;
												idx3 += (get_from_front3)*VLEN;
												idx4 += (!get_from_front3)*VLEN;
												idx5 += (get_from_front5)*VLEN;
												idx6 += (!get_from_front5)*VLEN;
												idx7 += (get_from_front7)*VLEN;
												idx8 += (!get_from_front7)*VLEN;
												last1 = from[idx1];
												last2 = from[idx2];
												last3 = from[idx3];
												last4 = from[idx4];
												last5 = from[idx5];
												last6 = from[idx6];
												last7 = from[idx7];
												last8 = from[idx8];
												get_from_front1 = (last1 < last2);
												get_from_front3 = (last3 < last4);
												get_from_front5 = (last5 < last6);
												get_from_front7 = (last7 < last8);

												from[chunk1end] = BIG; 
												from[chunk2end] = BIG; 
												from[chunk3end] = BIG; 
												from[chunk4end] = BIG; 
												from[chunk5end] = BIG; 
												from[chunk6end] = BIG; 
												from[chunk7end] = BIG; 
												from[chunk8end] = BIG; 

												out1 += VLEN;
												out3 += VLEN;
												out5 += VLEN;
												out7 += VLEN;

												int x1, x2, x3, x4;
												for( 
																				;
																				out1 < chunk1 + chunk_size * 2 - VLEN;
																				out1 += VLEN,
																				out3 += VLEN,
																				out5 += VLEN,
																				out7 += VLEN 
													 )
												{
																x1 = idx1;
																CMOVZ(get_from_front1, x1, idx2);
																x2 = idx3;
																CMOVZ(get_from_front3, x2, idx4);
																x3 = idx5;
																CMOVZ(get_from_front5, x3, idx6);
																x4 = idx7;
																CMOVZ(get_from_front7, x4, idx8);
																tmp1 = _MM_LOAD(from + x1);
																tmp3 = _MM_LOAD(from + x2);
																tmp5 = _MM_LOAD(from + x3);
																tmp7 = _MM_LOAD(from + x4);
																//printf("%d %d %d %d\n", get_from_front1, get_from_front3, get_from_front5, get_from_front7);
																/*
																tmp1 = _MM_LOAD(from + (get_from_front1) * idx1 + (!get_from_front1) * idx2);
																tmp3 = _MM_LOAD(from + (get_from_front3) * idx3 + (!get_from_front3) * idx4);
																tmp5 = _MM_LOAD(from + (get_from_front5) * idx5 + (!get_from_front5) * idx6);
																tmp7 = _MM_LOAD(from + (get_from_front7) * idx7 + (!get_from_front7) * idx8);
																*/

																/*
																tmp1 = _MM_LOAD(from + ((get_from_front1 ==1)? idx1 : idx2));
																tmp3 = _MM_LOAD(from + ((get_from_front3 ==1)? idx3 : idx4));
																tmp5 = _MM_LOAD(from + ((get_from_front5 ==1)? idx5 : idx6));
																tmp7 = _MM_LOAD(from + ((get_from_front7 ==1)? idx7 : idx8));
																*/

																																//merge tmp1, tmp2 here
																//reverse t2
#ifdef DEBUG_SORT
																printf("merge %d tmp1\n", idx1);
																printf("merge %d tmp2\n", idx2);
																printf("merge %d tmp3\n", idx3);
																printf("merge %d tmp4\n", idx4);
																printf("merge %d tmp5\n", idx5);
																printf("merge %d tmp6\n", idx6);
																printf("merge %d tmp7\n", idx7);
																printf("merge %d tmp8\n", idx8);

																printv(tmp1, "tmp1");
																printv(tmp2, "tmp2");
																printv(tmp3, "tmp3");
																printv(tmp4, "tmp4");
																printv(tmp5, "tmp5");
																printv(tmp6, "tmp6");
																printv(tmp7, "tmp7");
																printv(tmp8, "tmp8");
#endif
																tmp2 = _mm_shuffle_ps(tmp2, tmp2, 0x1b);
																tmp4 = _mm_shuffle_ps(tmp4, tmp4, 0x1b);
																tmp6 = _mm_shuffle_ps(tmp6, tmp6, 0x1b);
																tmp8 = _mm_shuffle_ps(tmp8, tmp8, 0x1b);

#ifdef DEBUG_SORT
																printv(tmp1, "tmp1");
																printv(tmp2, "tmp2");
#endif
																__m128 t1, t2;
																__m128 t3, t4;
																__m128 t5, t6;
																__m128 t7, t8;

																t1 = _MM_MIN(tmp1, tmp2); 
																t2 = _MM_MAX(tmp1, tmp2); 
																t3 = _MM_MIN(tmp3, tmp4); 
																t4 = _MM_MAX(tmp3, tmp4); 
																t5 = _MM_MIN(tmp5, tmp6); 
																t6 = _MM_MAX(tmp5, tmp6); 
																t7 = _MM_MIN(tmp7, tmp8); 
																t8 = _MM_MAX(tmp7, tmp8); 

																tmp1 = _mm_shuffle_ps(t1, t2, 0x1b);
																tmp2 = _mm_shuffle_ps(t1, t2, 0xb1);
																tmp3 = _mm_shuffle_ps(t3, t4, 0x1b);
																tmp4 = _mm_shuffle_ps(t3, t4, 0xb1);
																tmp5 = _mm_shuffle_ps(t5, t6, 0x1b);
																tmp6 = _mm_shuffle_ps(t5, t6, 0xb1);
																tmp7 = _mm_shuffle_ps(t7, t8, 0x1b);
																tmp8 = _mm_shuffle_ps(t7, t8, 0xb1);

#ifdef DEBUG_SORT
																printv(tmp1, "tmp1");
																printv(tmp2, "tmp2");
#endif
																t1 = _MM_MIN(tmp1, tmp2); 
																t2 = _MM_MAX(tmp1, tmp2); 
																t3 = _MM_MIN(tmp3, tmp4); 
																t4 = _MM_MAX(tmp3, tmp4); 
																t5 = _MM_MIN(tmp5, tmp6); 
																t6 = _MM_MAX(tmp5, tmp6); 
																t7 = _MM_MIN(tmp7, tmp8); 
																t8 = _MM_MAX(tmp7, tmp8); 

																tmp1 = _mm_blend_ps(t1, t2, 0xa);
																tmp2 = _mm_blend_ps(t2, t1, 0xa);
																tmp2 = _mm_shuffle_ps(tmp2, tmp2, 0xb1);
																tmp3 = _mm_blend_ps(t3, t4, 0xa);
																tmp4 = _mm_blend_ps(t4, t3, 0xa);
																tmp4 = _mm_shuffle_ps(tmp4, tmp4, 0xb1);
																tmp5 = _mm_blend_ps(t5, t6, 0xa);
																tmp6 = _mm_blend_ps(t6, t5, 0xa);
																tmp6 = _mm_shuffle_ps(tmp6, tmp6, 0xb1);
																tmp7 = _mm_blend_ps(t7, t8, 0xa);
																tmp8 = _mm_blend_ps(t8, t7, 0xa);
																tmp8 = _mm_shuffle_ps(tmp8, tmp8, 0xb1);

#ifdef DEBUG_SORT
																printv(tmp1, "tmp1");
																printv(tmp2, "tmp2");
#endif
																t1 = _MM_MIN(tmp1, tmp2); 
																t2 = _MM_MAX(tmp1, tmp2); 
																t3 = _MM_MIN(tmp3, tmp4); 
																t4 = _MM_MAX(tmp3, tmp4); 
																t5 = _MM_MIN(tmp5, tmp6); 
																t6 = _MM_MAX(tmp5, tmp6); 
																t7 = _MM_MIN(tmp7, tmp8); 
																t8 = _MM_MAX(tmp7, tmp8); 

																tmp1 = _mm_unpacklo_ps(t1, t2);
																tmp2 = _mm_unpackhi_ps(t1, t2);
																tmp3 = _mm_unpacklo_ps(t3, t4);
																tmp4 = _mm_unpackhi_ps(t3, t4);
																tmp5 = _mm_unpacklo_ps(t5, t6);
																tmp6 = _mm_unpackhi_ps(t5, t6);
																tmp7 = _mm_unpacklo_ps(t7, t8);
																tmp8 = _mm_unpackhi_ps(t7, t8);
#ifdef DEBUG_SORT

																printv(tmp1, "tmp1");
																printv(tmp2, "tmp2");
																printv(tmp3, "tmp3");
																printv(tmp4, "tmp4");
																printv(tmp5, "tmp5");
																printv(tmp6, "tmp6");
																printv(tmp7, "tmp7");
																printv(tmp8, "tmp8");

																printf("store at %d\n", out1);
																printf("store at %d\n", out3);
																printf("store at %d\n", out5);
																printf("store at %d\n\n", out7);
																printf("\n\n");
#endif
																_MM_STORE(to + out1, tmp1);
																_MM_STORE(to + out3, tmp3);
																_MM_STORE(to + out5, tmp5);
																_MM_STORE(to + out7, tmp7);

																/*
																CMOVZ(!get_from_front1, idx1, idx1 + VLEN);
																CMOVZ(get_from_front1, idx2, idx2 + VLEN);
																CMOVZ(!get_from_front3, idx3, idx3 + VLEN);
																CMOVZ(get_from_front3, idx4, idx4 + VLEN);
																CMOVZ(!get_from_front5, idx5, idx5 + VLEN);
																CMOVZ(get_from_front5, idx6, idx6 + VLEN);
																CMOVZ(!get_from_front7, idx7, idx7 + VLEN);
																CMOVZ(get_from_front7, idx8, idx8 + VLEN);
																*/
																idx1 += (get_from_front1)*VLEN;
																idx2 += (!get_from_front1)*VLEN;
																idx3 += (get_from_front3)*VLEN;
																idx4 += (!get_from_front3)*VLEN;
																idx5 += (get_from_front5)*VLEN;
																idx6 += (!get_from_front5)*VLEN;
																idx7 += (get_from_front7)*VLEN;
																idx8 += (!get_from_front7)*VLEN;
																last1 = from[idx1];
																last2 = from[idx2];
																last3 = from[idx3];
																last4 = from[idx4];
																last5 = from[idx5];
																last6 = from[idx6];
																last7 = from[idx7];
																last8 = from[idx8];
																get_from_front1 = (last1 < last2);
																get_from_front3 = (last3 < last4);
																get_from_front5 = (last5 < last6);
																get_from_front7 = (last7 < last8);
												}
#ifdef DEBUG_SORT
												printf("store at %d\n", out1);
												printf("store at %d\n", out3);
												printf("store at %d\n", out5);
												printf("store at %d\n\n", out7);
#endif
												_mm_store_ps(to + out1, tmp2);
												_mm_store_ps(to + out3, tmp4);
												_mm_store_ps(to + out5, tmp6);
												_mm_store_ps(to + out7, tmp8);

												from[chunk1end] = chunk1tmp; 
												from[chunk2end] = chunk2tmp; 
												from[chunk3end] = chunk3tmp; 
												from[chunk4end] = chunk4tmp; 
												from[chunk5end] = chunk5tmp; 
												from[chunk6end] = chunk6tmp; 
												from[chunk7end] = chunk7tmp; 
												from[chunk8end] = chunk8tmp; 
								}
				}
				//exit(-1);
				return step;
}
#elif AVX
				#define imm1 0x1b // imm for reverse operation
				#define imm2 0x01
static int merge_4way(float ** bufs, int BLOCK_SIZE)
{
				float BIG = (float) BLOCK_SIZE;
				int step, chunk_size;
				for(step = 0, chunk_size=VLEN; chunk_size * 4 <= BLOCK_SIZE; step++, chunk_size *= 2)
				{

						float * from, *to;
						from = bufs[step % 2];
						to = bufs[(step + 1) % 2];
						int chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, chunk7, chunk8;
						for(
										chunk1 = 0, chunk2 = chunk1 + chunk_size, 
										chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
										chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
										chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size;
										chunk8 < BLOCK_SIZE; 
										chunk1 += chunk_size * 8, chunk2 = chunk1 + chunk_size,
										chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
										chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
										chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size

						   )
						{
								//		printf("merge %d & %d....\n", chunk1, chunk2);
								int idx1 = chunk1, idx2, out1;
								int idx3 = chunk3, idx4, out2;
								int idx5 = chunk5, idx6, out3;
								int idx7 = chunk7, idx8, out4;
								float last1, last2;
								float last3, last4;
								float last5, last6;
								float last7, last8;
								__m256 tmp1, tmp2;
								__m256 tmp3, tmp4;
								__m256 tmp5, tmp6;
								__m256 tmp7, tmp8;
								tmp2 = _mm256_load_ps(from + idx1);
								tmp4 = _mm256_load_ps(from + idx3);
								tmp6 = _mm256_load_ps(from + idx5);
								tmp8 = _mm256_load_ps(from + idx7);
								int get_from_front1 = 0;
								int get_from_front2 = 0;
								int get_from_front3 = 0;
								int get_from_front4 = 0;
								last1 = *(from + idx1 + VLEN);
								last3 = *(from + idx3 + VLEN);
								last5 = *(from + idx5 + VLEN);
								last7 = *(from + idx7 + VLEN);
								idx1 += VLEN;
								idx3 += VLEN;
								idx5 += VLEN;
								idx7 += VLEN;
								idx2 = chunk2; out1 = chunk1;
								idx4 = chunk4; out2 = chunk3;
								idx6 = chunk6; out3 = chunk5;
								idx8 = chunk8; out4 = chunk7;
				int chunk1end = chunk1 + chunk_size;
												int chunk2end = chunk2 + chunk_size;
												int chunk3end = chunk3 + chunk_size;
												int chunk4end = chunk4 + chunk_size;
												int chunk5end = chunk5 + chunk_size;
												int chunk6end = chunk6 + chunk_size;
												int chunk7end = chunk7 + chunk_size;
												int chunk8end = chunk8 + chunk_size;

												float chunk1tmp = from[chunk1end];
												float chunk2tmp = from[chunk2end];
												float chunk3tmp = from[chunk3end];
												float chunk4tmp = from[chunk4end];
												float chunk5tmp = from[chunk5end];
												float chunk6tmp = from[chunk6end];
												float chunk7tmp = from[chunk7end];
												float chunk8tmp = from[chunk8end];

				
								int x1, x2, x3, x4;
								x1 = idx2;
								x2 = idx4;
								x3 = idx6;
								x4 = idx8;
								tmp1 = _MM_LOAD(from + x1);
								tmp3 = _MM_LOAD(from + x2);
								tmp5 = _MM_LOAD(from + x3);
								tmp7 = _MM_LOAD(from + x4);

										__m256 t1, t2;
										__m256 t3, t4;
										__m256 t5, t6;
										__m256 t7, t8;

										//	printf("%d %d %d \n", __LINE__, idx1, idx2);
										tmp2 = _mm256_permute_ps(tmp2, imm1); //reverse tmp2
										tmp2 = _mm256_permute2f128_ps(tmp2, tmp2, imm2); 
										tmp4 = _mm256_permute_ps(tmp4, imm1); //reverse tmp2
										tmp4 = _mm256_permute2f128_ps(tmp4, tmp4, imm2); 
										tmp6 = _mm256_permute_ps(tmp6, imm1); //reverse tmp2
										tmp6 = _mm256_permute2f128_ps(tmp6, tmp6, imm2); 
										tmp8 = _mm256_permute_ps(tmp8, imm1); //reverse tmp2
										tmp8 = _mm256_permute2f128_ps(tmp8, tmp8, imm2); 

										__m256 high1 = _mm256_max_ps(tmp1, tmp2);
										__m256 low1 = _mm256_min_ps(tmp1, tmp2);
										__m256 high2 = _mm256_max_ps(tmp3, tmp4);
										__m256 low2 = _mm256_min_ps(tmp3, tmp4);
										__m256 high3 = _mm256_max_ps(tmp5, tmp6);
										__m256 low3 = _mm256_min_ps(tmp5, tmp6);
										__m256 high4 = _mm256_max_ps(tmp7, tmp8);
										__m256 low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_permute2f128_ps(low1, high1, 0x20);
										tmp2 = _mm256_permute2f128_ps(low1, high1, 0x31);
										tmp3 = _mm256_permute2f128_ps(low2, high2, 0x20);
										tmp4 = _mm256_permute2f128_ps(low2, high2, 0x31);
										tmp5 = _mm256_permute2f128_ps(low3, high3, 0x20);
										tmp6 = _mm256_permute2f128_ps(low3, high3, 0x31);
										tmp7 = _mm256_permute2f128_ps(low4, high4, 0x20);
										tmp8 = _mm256_permute2f128_ps(low4, high4, 0x31);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_shuffle_ps(low1, high1, 0x44);
										tmp2 = _mm256_shuffle_ps(low1, high1, 0xEE);
										tmp3 = _mm256_shuffle_ps(low2, high2, 0x44);
										tmp4 = _mm256_shuffle_ps(low2, high2, 0xEE);
										tmp5 = _mm256_shuffle_ps(low3, high3, 0x44);
										tmp6 = _mm256_shuffle_ps(low3, high3, 0xEE);
										tmp7 = _mm256_shuffle_ps(low4, high4, 0x44);
										tmp8 = _mm256_shuffle_ps(low4, high4, 0xEE);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_blend_ps(low1, high1, 0xAA);
										t1 = _mm256_blend_ps(high1, low1, 0xAA);
										tmp2 = _mm256_shuffle_ps(t1, t1, 0XB1);
										tmp3 = _mm256_blend_ps(low2, high2, 0xAA);
										t3 = _mm256_blend_ps(high2, low2, 0xAA);
										tmp4 = _mm256_shuffle_ps(t3, t3, 0XB1);
										tmp5 = _mm256_blend_ps(low3, high3, 0xAA);
										t5 = _mm256_blend_ps(high3, low3, 0xAA);
										tmp6 = _mm256_shuffle_ps(t5, t5, 0XB1);
										tmp7 = _mm256_blend_ps(low4, high4, 0xAA);
										t7 = _mm256_blend_ps(high4, low4, 0xAA);
										tmp8 = _mm256_shuffle_ps(t7, t7, 0XB1);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										t1 = _mm256_unpacklo_ps(low1, high1);
										t2 = _mm256_unpackhi_ps(low1, high1);
										t3 = _mm256_unpacklo_ps(low2, high2);
										t4 = _mm256_unpackhi_ps(low2, high2);
										t5 = _mm256_unpacklo_ps(low3, high3);
										t6 = _mm256_unpackhi_ps(low3, high3);
										t7 = _mm256_unpacklo_ps(low4, high4);
										t8 = _mm256_unpackhi_ps(low4, high4);

										tmp1 = _mm256_permute2f128_ps(t1, t2, 0x20);
										tmp2 = _mm256_permute2f128_ps(t1, t2, 0x31);
										tmp3 = _mm256_permute2f128_ps(t3, t4, 0x20);
										tmp4 = _mm256_permute2f128_ps(t3, t4, 0x31);
										tmp5 = _mm256_permute2f128_ps(t5, t6, 0x20);
										tmp6 = _mm256_permute2f128_ps(t5, t6, 0x31);
										tmp7 = _mm256_permute2f128_ps(t7, t8, 0x20);
										tmp8 = _mm256_permute2f128_ps(t7, t8, 0x31);

										_mm256_store_ps(to + out1, tmp1);
										_mm256_store_ps(to + out2, tmp3);
										_mm256_store_ps(to + out3, tmp5);
										_mm256_store_ps(to + out4, tmp7);

										idx1 += (get_from_front1)*VLEN;
										idx2 += (!get_from_front1)*VLEN;
										idx3 += (get_from_front2)*VLEN;
										idx4 += (!get_from_front2)*VLEN;
										idx5 += (get_from_front3)*VLEN;
										idx6 += (!get_from_front3)*VLEN;
										idx7 += (get_from_front4)*VLEN;
										idx8 += (!get_from_front4)*VLEN;
										last1 = from[idx1];
										last2 = from[idx2];
										last3 = from[idx3];
										last4 = from[idx4];
										last5 = from[idx5];
										last6 = from[idx6];
										last7 = from[idx7];
										last8 = from[idx8];
										get_from_front1 = (last1 < last2);
										get_from_front2 = (last3 < last4);
										get_from_front3 = (last5 < last6);
										get_from_front4 = (last7 < last8);
							from[chunk1end] = BIG; 
												from[chunk2end] = BIG; 
												from[chunk3end] = BIG; 
												from[chunk4end] = BIG; 
												from[chunk5end] = BIG; 
												from[chunk6end] = BIG; 
												from[chunk7end] = BIG; 
												from[chunk8end] = BIG; 
			
												out1 += VLEN;
												out2 += VLEN;
												out3 += VLEN;
												out4 += VLEN;

								for(
																;
												out1 < chunk1 + chunk_size * 2 - VLEN;
												out1 += VLEN,
												out2 += VLEN,
												out3 += VLEN,
												out4 += VLEN
								   )
								{
										
																x1 = idx1;
																CMOVZ(get_from_front1, x1, idx2);
																x2 = idx3;
																CMOVZ(get_from_front2, x2, idx4);
																x3 = idx5;
																CMOVZ(get_from_front3, x3, idx6);
																x4 = idx7;
																CMOVZ(get_from_front4, x4, idx8);
																tmp1 = _MM_LOAD(from + x1);
																tmp3 = _MM_LOAD(from + x2);
																tmp5 = _MM_LOAD(from + x3);
																tmp7 = _MM_LOAD(from + x4);

										__m256 t1, t2;
										__m256 t3, t4;
										__m256 t5, t6;
										__m256 t7, t8;
										//	printf("%d %d %d \n", __LINE__, idx1, idx2);
										tmp2 = _mm256_permute_ps(tmp2, imm1); //reverse tmp2
										tmp2 = _mm256_permute2f128_ps(tmp2, tmp2, imm2); 
										tmp4 = _mm256_permute_ps(tmp4, imm1); //reverse tmp2
										tmp4 = _mm256_permute2f128_ps(tmp4, tmp4, imm2); 
										tmp6 = _mm256_permute_ps(tmp6, imm1); //reverse tmp2
										tmp6 = _mm256_permute2f128_ps(tmp6, tmp6, imm2); 
										tmp8 = _mm256_permute_ps(tmp8, imm1); //reverse tmp2
										tmp8 = _mm256_permute2f128_ps(tmp8, tmp8, imm2); 

										__m256 high1 = _mm256_max_ps(tmp1, tmp2);
										__m256 low1 = _mm256_min_ps(tmp1, tmp2);
										__m256 high2 = _mm256_max_ps(tmp3, tmp4);
										__m256 low2 = _mm256_min_ps(tmp3, tmp4);
										__m256 high3 = _mm256_max_ps(tmp5, tmp6);
										__m256 low3 = _mm256_min_ps(tmp5, tmp6);
										__m256 high4 = _mm256_max_ps(tmp7, tmp8);
										__m256 low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_permute2f128_ps(low1, high1, 0x20);
										tmp2 = _mm256_permute2f128_ps(low1, high1, 0x31);
										tmp3 = _mm256_permute2f128_ps(low2, high2, 0x20);
										tmp4 = _mm256_permute2f128_ps(low2, high2, 0x31);
										tmp5 = _mm256_permute2f128_ps(low3, high3, 0x20);
										tmp6 = _mm256_permute2f128_ps(low3, high3, 0x31);
										tmp7 = _mm256_permute2f128_ps(low4, high4, 0x20);
										tmp8 = _mm256_permute2f128_ps(low4, high4, 0x31);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_shuffle_ps(low1, high1, 0x44);
										tmp2 = _mm256_shuffle_ps(low1, high1, 0xEE);
										tmp3 = _mm256_shuffle_ps(low2, high2, 0x44);
										tmp4 = _mm256_shuffle_ps(low2, high2, 0xEE);
										tmp5 = _mm256_shuffle_ps(low3, high3, 0x44);
										tmp6 = _mm256_shuffle_ps(low3, high3, 0xEE);
										tmp7 = _mm256_shuffle_ps(low4, high4, 0x44);
										tmp8 = _mm256_shuffle_ps(low4, high4, 0xEE);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_blend_ps(low1, high1, 0xAA);
										t1 = _mm256_blend_ps(high1, low1, 0xAA);
										tmp2 = _mm256_shuffle_ps(t1, t1, 0XB1);
										tmp3 = _mm256_blend_ps(low2, high2, 0xAA);
										t3 = _mm256_blend_ps(high2, low2, 0xAA);
										tmp4 = _mm256_shuffle_ps(t3, t3, 0XB1);
										tmp5 = _mm256_blend_ps(low3, high3, 0xAA);
										t5 = _mm256_blend_ps(high3, low3, 0xAA);
										tmp6 = _mm256_shuffle_ps(t5, t5, 0XB1);
										tmp7 = _mm256_blend_ps(low4, high4, 0xAA);
										t7 = _mm256_blend_ps(high4, low4, 0xAA);
										tmp8 = _mm256_shuffle_ps(t7, t7, 0XB1);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										t1 = _mm256_unpacklo_ps(low1, high1);
										t2 = _mm256_unpackhi_ps(low1, high1);
										t3 = _mm256_unpacklo_ps(low2, high2);
										t4 = _mm256_unpackhi_ps(low2, high2);
										t5 = _mm256_unpacklo_ps(low3, high3);
										t6 = _mm256_unpackhi_ps(low3, high3);
										t7 = _mm256_unpacklo_ps(low4, high4);
										t8 = _mm256_unpackhi_ps(low4, high4);

										tmp1 = _mm256_permute2f128_ps(t1, t2, 0x20);
										tmp2 = _mm256_permute2f128_ps(t1, t2, 0x31);
										tmp3 = _mm256_permute2f128_ps(t3, t4, 0x20);
										tmp4 = _mm256_permute2f128_ps(t3, t4, 0x31);
										tmp5 = _mm256_permute2f128_ps(t5, t6, 0x20);
										tmp6 = _mm256_permute2f128_ps(t5, t6, 0x31);
										tmp7 = _mm256_permute2f128_ps(t7, t8, 0x20);
										tmp8 = _mm256_permute2f128_ps(t7, t8, 0x31);

										_mm256_store_ps(to + out1, tmp1);
										_mm256_store_ps(to + out2, tmp3);
										_mm256_store_ps(to + out3, tmp5);
										_mm256_store_ps(to + out4, tmp7);

										idx1 += (get_from_front1)*VLEN;
										idx2 += (!get_from_front1)*VLEN;
										idx3 += (get_from_front2)*VLEN;
										idx4 += (!get_from_front2)*VLEN;
										idx5 += (get_from_front3)*VLEN;
										idx6 += (!get_from_front3)*VLEN;
										idx7 += (get_from_front4)*VLEN;
										idx8 += (!get_from_front4)*VLEN;
										last1 = from[idx1];
										last2 = from[idx2];
										last3 = from[idx3];
										last4 = from[idx4];
										last5 = from[idx5];
										last6 = from[idx6];
										last7 = from[idx7];
										last8 = from[idx8];
										get_from_front1 = (last1 < last2);
										get_from_front2 = (last3 < last4);
										get_from_front3 = (last5 < last6);
										get_from_front4 = (last7 < last8);

								}
								_mm256_store_ps(to + out1, tmp2);
								_mm256_store_ps(to + out2, tmp4);
								_mm256_store_ps(to + out3, tmp6);
								_mm256_store_ps(to + out4, tmp8);
from[chunk1end] = chunk1tmp; 
												from[chunk2end] = chunk2tmp; 
												from[chunk3end] = chunk3tmp; 
												from[chunk4end] = chunk4tmp; 
												from[chunk5end] = chunk5tmp; 
												from[chunk6end] = chunk6tmp; 
												from[chunk7end] = chunk7tmp; 
												from[chunk8end] = chunk8tmp; 

						}
				}
				return step;
}
#elif __MIC__
static int merge_4way(float ** bufs, int BLOCK_SIZE)
{
				float BIG = (float) BLOCK_SIZE;
				int step, chunk_size;
const __m512i offset1 = _mm512_set_epi32(8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0);
		const __m512i offset2 = _mm512_set_epi32(7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15);
		const __m512i offset3 = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

for(step = 0, chunk_size=VLEN; chunk_size * 2 <= BLOCK_SIZE; step++, chunk_size *= 2)
				{
						float * from, *to;
						from = bufs[step % 2];
						to = bufs[(step + 1) % 2];
						int chunk1, chunk2;
						int chunk3, chunk4;
						int get_from_front1 = 0;
						int get_from_front2 = 0;
						int get_from_front3 = 0;
						int get_from_front4 = 0;
						for(
										chunk1 = 0, chunk2 = chunk_size,
										chunk3 = chunk_size * 2, chunk4 = chunk_size * 3;
										chunk4 < BLOCK_SIZE;
										chunk1 += chunk_size * 2, chunk2 += chunk_size * 2,
										chunk3 += chunk_size * 2, chunk4 += chunk_size * 2
						   )
						{

								int idx1= chunk1, idx2, out1;
								int idx3= chunk3, idx4, out3;
								int chunk1end = chunk1 + chunk_size;
								int chunk2end = chunk2 + chunk_size;
								int chunk3end = chunk3 + chunk_size;
								int chunk4end = chunk4 + chunk_size;
								_VECTOR tmp1, tmp2 = _MM_LOAD(from + idx1); 
								_VECTOR tmp3, tmp4 = _MM_LOAD(from + idx3); 
								tmp2 = _mm512_permutevar_epi32(offset3, tmp2); // reverse
								tmp4 = _mm512_permutevar_epi32(offset3, tmp4); // reverse
								float last1 = *(from + idx1 + VLEN), last2;
								float last3 = *(from + idx3 + VLEN), last4;
								idx1 += VLEN;
								idx3 += VLEN;
							//	CMOVZ(idx1, idx2, out1);
								idx2 = chunk2; out1 = chunk1;
								idx4 = chunk4; out3 = chunk3;
								//printf("chunk1 %d chunk2 %d\n", chunk1, chunk2);
										tmp1 = _MM_LOAD(from + idx2);
										tmp3 = _MM_LOAD(from + idx4);
										last2 = from[idx2 + VLEN];
										last4 = from[idx4 + VLEN];
										idx2 += VLEN;
										idx4 += VLEN;

//#define DEBUG_SORT
										//merge tmp1, tmp2 here
										//reverse t2
#ifdef DEBUG_SORT
										printf("merge %d tmp1\n", idx1);
										printf("merge %d tmp2\n", idx2);
										printv(tmp1, "input1");
										printv(tmp2, "input2");

#endif
										_VECTOR t1 = _MM_MIN(tmp1, tmp2);
										_VECTOR t2 = _MM_MAX(tmp1, tmp2);
										_VECTOR t3 = _MM_MIN(tmp3, tmp4);
										_VECTOR t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_permute4f128_epi32(t1, 0x00ff, t2, _MM_PERM_BADC);
										tmp2 = _mm512_mask_permute4f128_epi32(t2, 0xff00, t1, _MM_PERM_BADC);
										tmp3 = _mm512_mask_permute4f128_epi32(t3, 0x00ff, t4, _MM_PERM_BADC);
										tmp4 = _mm512_mask_permute4f128_epi32(t4, 0xff00, t3, _MM_PERM_BADC);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_permute4f128_epi32(t1, 0x0f0f, t2, _MM_PERM_CDAB);
										tmp2 = _mm512_mask_permute4f128_epi32(t2, 0xf0f0, t1, _MM_PERM_CDAB);
										tmp3 = _mm512_mask_permute4f128_epi32(t3, 0x0f0f, t4, _MM_PERM_CDAB);
										tmp4 = _mm512_mask_permute4f128_epi32(t4, 0xf0f0, t3, _MM_PERM_CDAB);
#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_swizzle_ps(t1, 0x3333, t2, _MM_SWIZ_REG_BADC);
										tmp2 = _mm512_mask_swizzle_ps(t2, 0xcccc, t1, _MM_SWIZ_REG_BADC);
										tmp3 = _mm512_mask_swizzle_ps(t3, 0x3333, t4, _MM_SWIZ_REG_BADC);
										tmp4 = _mm512_mask_swizzle_ps(t4, 0xcccc, t3, _MM_SWIZ_REG_BADC);
										//tmp1 = _mm512_mask_swizzle_ps(t1, 0xffff, t2, _MM_PERM_BADC);
										//tmp2 = _mm512_mask_swizzle_ps(t2, 0xffff, t1, _MM_PERM_BADC);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_swizzle_ps(t1, 0x5555, t2, _MM_SWIZ_REG_CDAB);
										tmp2 = _mm512_mask_swizzle_ps(t2, 0xaaaa, t1, _MM_SWIZ_REG_CDAB);
										tmp3 = _mm512_mask_swizzle_ps(t3, 0x5555, t4, _MM_SWIZ_REG_CDAB);
										tmp4 = _mm512_mask_swizzle_ps(t4, 0xaaaa, t3, _MM_SWIZ_REG_CDAB);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										t1 = _mm512_permutevar_epi32(offset2, t1);
										t2 = _mm512_permutevar_epi32(offset1, t2);
										tmp2 = _mm512_mask_mov_ps(t1,0x5555,t2);
										tmp1 = _mm512_mask_mov_ps(t2,0x5555,t1);
										t3 = _mm512_permutevar_epi32(offset2, t3);
										t4 = _mm512_permutevar_epi32(offset1, t4);
										tmp4 = _mm512_mask_mov_ps(t3,0x5555,t4);
										tmp3 = _mm512_mask_mov_ps(t4,0x5555,t3);

#ifdef DEBUG_SORT
										printv(tmp1, "output1");
										printv(tmp2, "output2");
										printf("\n");
#endif

										
										_MM_STORE(to + out1, tmp1);
										_MM_STORE(to + out3, tmp3);

										get_from_front1 =  (last1 < last2 );
										get_from_front3 =  (last3 < last4 );

								out1 += VLEN;
								out3 += VLEN;
							float chunk1tmp = from[chunk1end]; 
							float chunk2tmp = from[chunk2end]; 
							float chunk3tmp = from[chunk3end]; 
							float chunk4tmp = from[chunk4end]; 
							from[chunk1end] = BIG; 
												from[chunk2end] = BIG; 
							from[chunk3end] = BIG; 
												from[chunk4end] = BIG; 

								for( 
										;
												out1 < chunk1 + chunk_size * 2 - VLEN,
												out3 < chunk3 + chunk_size * 2 - VLEN;
												out1 += VLEN,
												out3 += VLEN
								   )
								{
										//break;
										/*
										if(get_from_front1)
										{
												tmp1 = _MM_LOAD(from + idx1);
												last1 = from[idx1 + VLEN];
												idx1 += VLEN;
										}
										else
										{
												tmp1 = _MM_LOAD(from + idx2);
												last2 = from[idx2 + VLEN];
												idx2 += VLEN;
										}

										if(get_from_front3)
										{
												tmp3 = _MM_LOAD(from + idx3);
												last3 = from[idx3 + VLEN];
												idx3 += VLEN;
										}
										else
										{
												tmp3 = _MM_LOAD(from + idx4);
												last4 = from[idx4 + VLEN];
												idx4 += VLEN;
										}
										*/
										int x1, x2;
										x1 = get_from_front1 * idx1 + !get_from_front1 * idx2;
										x2 = get_from_front3 * idx3 + !get_from_front3 * idx4;

										tmp1 = _MM_LOAD(from + x1);
										tmp3 = _MM_LOAD(from + x2);

										
										idx1 += VLEN * get_from_front1;
										idx2 += VLEN * !get_from_front1;
										idx3 += VLEN * get_from_front2;
										idx4 += VLEN * !get_from_front2;

										last1 = from[idx1];
										last2 = from[idx2];
										last3 = from[idx3];
										last4 = from[idx4];
										
										_VECTOR t1 = _MM_MIN(tmp1, tmp2);
										_VECTOR t2 = _MM_MAX(tmp1, tmp2);
										_VECTOR t3 = _MM_MIN(tmp3, tmp4);
										_VECTOR t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_permute4f128_epi32(t1, 0x00ff, t2, _MM_PERM_BADC);
										tmp2 = _mm512_mask_permute4f128_epi32(t2, 0xff00, t1, _MM_PERM_BADC);
										tmp3 = _mm512_mask_permute4f128_epi32(t3, 0x00ff, t4, _MM_PERM_BADC);
										tmp4 = _mm512_mask_permute4f128_epi32(t4, 0xff00, t3, _MM_PERM_BADC);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_permute4f128_epi32(t1, 0x0f0f, t2, _MM_PERM_CDAB);
										tmp2 = _mm512_mask_permute4f128_epi32(t2, 0xf0f0, t1, _MM_PERM_CDAB);
										tmp3 = _mm512_mask_permute4f128_epi32(t3, 0x0f0f, t4, _MM_PERM_CDAB);
										tmp4 = _mm512_mask_permute4f128_epi32(t4, 0xf0f0, t3, _MM_PERM_CDAB);
#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_swizzle_ps(t1, 0x3333, t2, _MM_SWIZ_REG_BADC);
										tmp2 = _mm512_mask_swizzle_ps(t2, 0xcccc, t1, _MM_SWIZ_REG_BADC);
										tmp3 = _mm512_mask_swizzle_ps(t3, 0x3333, t4, _MM_SWIZ_REG_BADC);
										tmp4 = _mm512_mask_swizzle_ps(t4, 0xcccc, t3, _MM_SWIZ_REG_BADC);
										//tmp1 = _mm512_mask_swizzle_ps(t1, 0xffff, t2, _MM_PERM_BADC);
										//tmp2 = _mm512_mask_swizzle_ps(t2, 0xffff, t1, _MM_PERM_BADC);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										tmp1 = _mm512_mask_swizzle_ps(t1, 0x5555, t2, _MM_SWIZ_REG_CDAB);
										tmp2 = _mm512_mask_swizzle_ps(t2, 0xaaaa, t1, _MM_SWIZ_REG_CDAB);
										tmp3 = _mm512_mask_swizzle_ps(t3, 0x5555, t4, _MM_SWIZ_REG_CDAB);
										tmp4 = _mm512_mask_swizzle_ps(t4, 0xaaaa, t3, _MM_SWIZ_REG_CDAB);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);
										t3 = _MM_MIN(tmp3, tmp4);
										t4 = _MM_MAX(tmp3, tmp4);

										t1 = _mm512_permutevar_epi32(offset2, t1);
										t2 = _mm512_permutevar_epi32(offset1, t2);
										tmp2 = _mm512_mask_mov_ps(t1,0x5555,t2);
										tmp1 = _mm512_mask_mov_ps(t2,0x5555,t1);
										t3 = _mm512_permutevar_epi32(offset2, t3);
										t4 = _mm512_permutevar_epi32(offset1, t4);
										tmp4 = _mm512_mask_mov_ps(t3,0x5555,t4);
										tmp3 = _mm512_mask_mov_ps(t4,0x5555,t3);

#ifdef DEBUG_SORT
										printv(tmp1, "output1");
										printv(tmp2, "output2");
										printf("\n");
#endif

										
										_MM_STORE(to + out1, tmp1);
										_MM_STORE(to + out3, tmp3);

										get_from_front1 =  (last1 < last2 );
										get_from_front3 =  (last3 < last4 );
//#define DEBUG_SORT
										//merge tmp1, tmp2 here
										//reverse t2
										//
										//
								}
								tmp2 = _mm512_permutevar_epi32(offset3, tmp2); // reverse
								tmp4 = _mm512_permutevar_epi32(offset3, tmp4); // reverse
								_MM_STORE(to + out1, tmp2);
								_MM_STORE(to + out3, tmp4);
from[chunk1end] = chunk1tmp; 
												from[chunk2end] = chunk2tmp; 
from[chunk3end] = chunk3tmp; 
												from[chunk4end] = chunk4tmp; 
						}
				}
				return step;
}
#endif

int hist_sorting_float_simd 
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_count
 )
{

		//fprintf(stderr,"%d here\n", __LINE__);
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

		int N2bin_count;  
		for(N2bin_count=1; N2bin_count < bin_count; N2bin_count *= 2); 

		int BLOCK_SIZE = N2bin_count * BMCOEFF; // 4-way loop unrolled 

		int work_size = ((count - start) / BLOCK_SIZE ) * BLOCK_SIZE;


		for(i = start+work_size; i<count; i++){
				float d = data[i];
				for(j = bin_count-1; d < boundary[j]; j--);
				bin[j]++;
		}


		float * buffer1 = _mm_malloc(sizeof(float) * BLOCK_SIZE + VLEN, 64);
		float * buffer2 = _mm_malloc(sizeof(float) * BLOCK_SIZE + VLEN, 64);
		float * bufs[2];
		bufs[0] = buffer1;
		bufs[1] = buffer2;

		int step=0, chunk_size;

		int iter_base;
		for(iter_base = 0; iter_base < work_size; iter_base += BLOCK_SIZE){ // operate on 1 block
				memcpy(buffer1, data + start + iter_base, sizeof(float) * BLOCK_SIZE);

#ifdef SSE
				//in-register sort
				for(i = 0; i < BLOCK_SIZE; i+=16) // for 8 * 8 elements in block
				{
						//#define DEBUG_IN
						_VECTOR f1 = _MM_LOAD(buffer1 + i);
						_VECTOR f2 = _MM_LOAD(buffer1 + i + VLEN );
						_VECTOR f3 = _MM_LOAD(buffer1 + i + VLEN * 2 );
						_VECTOR f4 = _MM_LOAD(buffer1 + i + VLEN * 3 );
#ifdef DEBUG_IN
						printv(f1, "in1");
						printv(f2, "in2");
						printv(f3, "in3");
						printv(f4, "in4");
#endif
						_VECTOR b1 = _MM_MIN(f1, f2);
						_VECTOR b2 = _MM_MAX(f1, f2);
						_VECTOR b3 = _MM_MIN(f3, f4);
						_VECTOR b4 = _MM_MAX(f3, f4);

						f1 = b1;
						f2 = _MM_MIN(b2, b3);
						f3 = _MM_MAX(b2, b3);
						f4 = b4;

						b1 = _MM_MIN(f1, f2);
						b2 = _MM_MAX(f1, f2);
						b3 = _MM_MIN(f3, f4);
						b4 = _MM_MAX(f3, f4);

						f1 = b1;
						f2 = _MM_MIN(b2, b3);
						f3 = _MM_MAX(b2, b3);
						f4 = b4;

						b1 = _mm_shuffle_ps(f1, f2, _MM_SHUFFLE(1,0,1,0));
						b2 = _mm_shuffle_ps(f3, f4, _MM_SHUFFLE(1,0,1,0));

						_VECTOR t1 = _mm_shuffle_ps(b1, b2, _MM_SHUFFLE(2,0,2,0));
						_VECTOR t2 = _mm_shuffle_ps(b1, b2, _MM_SHUFFLE(3,1,3,1));

						b3 = _mm_shuffle_ps(f1, f2, _MM_SHUFFLE(3,2,3,2));
						f1 = t1;
						f2 = t2;
						b4 = _mm_shuffle_ps(f3, f4, _MM_SHUFFLE(3,2,3,2));

						f3 = _mm_shuffle_ps(b3, b4, _MM_SHUFFLE(2,0,2,0));
						f4 = _mm_shuffle_ps(b3, b4, _MM_SHUFFLE(3,1,3,1));

#ifdef DEBUG_IN
						printf("at %d", i);
						printv(f1, "out1");
						printf("at %d", i + VLEN);
						printv(f2, "out2");
						printf("at %d", i + VLEN*2);
						printv(f3, "out3");
						printf("at %d", i + VLEN*3);
						printv(f4, "out4");
#endif

						_MM_STORE(buffer1 + i, f1);
						_MM_STORE(buffer1 + i + VLEN, f2 );
						_MM_STORE(buffer1 + i + VLEN * 2, f3 );
						_MM_STORE(buffer1 + i + VLEN * 3, f4 );
				}
				step = merge_4way(bufs, BLOCK_SIZE);
				//#define DEBUG_SORT

				
				/*
				for(step = 0, chunk_size=VLEN; chunk_size * 4 <= BLOCK_SIZE; step++, chunk_size *= 2)
				{
								//printf("%d\n", step);

						float * from, *to;
						from = bufs[step % 2];
						to = bufs[(step + 1) % 2];
						int chunk1, chunk2;
						int chunk3, chunk4;
						int chunk5, chunk6;
						int chunk7, chunk8;
						for(
										chunk1 = 0, chunk2 = chunk1 + chunk_size, 
										chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
										chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
										chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size;
										chunk8 < BLOCK_SIZE; 
										chunk1 += chunk_size * 8, chunk2 = chunk1 + chunk_size,
										chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
										chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
										chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size
						   )
						{
#ifdef DEBUG_SORT
								printf("merge chunk : %d %d %d %d %d %d %d %d\n", chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, chunk7, chunk8);
#endif
								int idx1= chunk1, idx2, out1;
								int idx3= chunk3, idx4, out3;
								int idx5= chunk5, idx6, out5;
								int idx7= chunk7, idx8, out7;
								_VECTOR tmp1, tmp2 = _MM_LOAD(from + idx1); 
								_VECTOR tmp3, tmp4 = _MM_LOAD(from + idx3); 
								_VECTOR tmp5, tmp6 = _MM_LOAD(from + idx5); 
								_VECTOR tmp7, tmp8 = _MM_LOAD(from + idx7); 
								float last1 = *(from + idx1 + VLEN), last2;
								float last3 = *(from + idx3 + VLEN), last4;
								float last5 = *(from + idx5 + VLEN), last6;
								float last7 = *(from + idx7 + VLEN), last8;
								idx1 += VLEN;
								idx3 += VLEN;
								idx5 += VLEN;
								idx7 += VLEN;
								int get_from_front1 = 0;
								int get_from_front3 = 0;
								int get_from_front5 = 0;
								int get_from_front7 = 0;

								for( 
												idx2 = chunk2, out1 = chunk1,
												idx4 = chunk4, out3 = chunk3,
												idx6 = chunk6, out5 = chunk5,
												idx8 = chunk8, out7 = chunk7;
												out1 < chunk1 + chunk_size * 2 - VLEN;
												out1 += VLEN,
												out3 += VLEN,
												out5 += VLEN,
												out7 += VLEN 
								   )
								{
										//printf("%d %d %d %d %d %d %d %d\n", idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8);
										//printf("%d %d %d %d\n", get_from_front1, get_from_front3, get_from_front5, get_from_front7);
										//printf("%d %d %d %d\n", out1, out3, out5, out7);
										//break;
										if(get_from_front1)
										{
												tmp1 = _MM_LOAD(from + idx1);
												last1 = from[idx1 + VLEN];
												idx1 += VLEN;
												//_mm_prefetch(from + idx1 + VLEN*8, _MM_HINT_T0);
												
										}
										else
										{
												tmp1 = _MM_LOAD(from + idx2);
												last2 = from[idx2 + VLEN];
												idx2 += VLEN;
												//_mm_prefetch(from + idx2 + VLEN*8, _MM_HINT_T0);
										}

										if(get_from_front3)
										{
												tmp3 = _MM_LOAD(from + idx3);
												last3 = from[idx3 + VLEN];
												idx3 += VLEN;
												//_mm_prefetch(from + idx3 + VLEN*8, _MM_HINT_T0);
										}
										else
										{
												tmp3 = _MM_LOAD(from + idx4);
												last4 = from[idx4 + VLEN];
												idx4 += VLEN;
												//_mm_prefetch(from + idx4 + VLEN*8, _MM_HINT_T0);
										}
										if(get_from_front5)
										{
												tmp5 = _MM_LOAD(from + idx5);
												last5 = from[idx5 + VLEN];
												idx5 += VLEN;
												//_mm_prefetch(from + idx5 + VLEN*8, _MM_HINT_T0);
										}
										else
										{
												tmp5 = _MM_LOAD(from + idx6);
												last6 = from[idx6 + VLEN];
												idx6 += VLEN;
												//_mm_prefetch(from + idx6 + VLEN*8, _MM_HINT_T0);
										}
										if(get_from_front7)
										{
												tmp7 = _MM_LOAD(from + idx7);
												last7 = from[idx7 + VLEN];
												idx7 += VLEN;
												//_mm_prefetch(from + idx7 + VLEN*8, _MM_HINT_T0);
										}
										else
										{
												tmp7 = _MM_LOAD(from + idx8);
												last8 = from[idx8 + VLEN];
												idx8 += VLEN;
												//_mm_prefetch(from + idx8 + VLEN*8, _MM_HINT_T0);
										}
										//merge tmp1, tmp2 here
										//reverse t2
#ifdef DEBUG_SORT
										printf("merge %d tmp1\n", idx1);
										printf("merge %d tmp2\n", idx2);
										printf("merge %d tmp3\n", idx3);
										printf("merge %d tmp4\n", idx4);
										printf("merge %d tmp5\n", idx5);
										printf("merge %d tmp6\n", idx6);
										printf("merge %d tmp7\n", idx7);
										printf("merge %d tmp8\n", idx8);

										printv(tmp1, "tmp1");
										printv(tmp2, "tmp2");
										printv(tmp3, "tmp3");
										printv(tmp4, "tmp4");
										printv(tmp5, "tmp5");
										printv(tmp6, "tmp6");
										printv(tmp7, "tmp7");
										printv(tmp8, "tmp8");
#endif
										tmp2 = _mm_shuffle_ps(tmp2, tmp2, 0x1b);
										tmp4 = _mm_shuffle_ps(tmp4, tmp4, 0x1b);
										tmp6 = _mm_shuffle_ps(tmp6, tmp6, 0x1b);
										tmp8 = _mm_shuffle_ps(tmp8, tmp8, 0x1b);

#ifdef DEBUG_SORT
										printv(tmp1, "tmp1");
										printv(tmp2, "tmp2");
#endif
										__m128 t1, t2;
										__m128 t3, t4;
										__m128 t5, t6;
										__m128 t7, t8;

										t1 = _MM_MIN(tmp1, tmp2); 
										t2 = _MM_MAX(tmp1, tmp2); 
										t3 = _MM_MIN(tmp3, tmp4); 
										t4 = _MM_MAX(tmp3, tmp4); 
										t5 = _MM_MIN(tmp5, tmp6); 
										t6 = _MM_MAX(tmp5, tmp6); 
										t7 = _MM_MIN(tmp7, tmp8); 
										t8 = _MM_MAX(tmp7, tmp8); 

										tmp1 = _mm_shuffle_ps(t1, t2, 0x1b);
										tmp2 = _mm_shuffle_ps(t1, t2, 0xb1);
										tmp3 = _mm_shuffle_ps(t3, t4, 0x1b);
										tmp4 = _mm_shuffle_ps(t3, t4, 0xb1);
										tmp5 = _mm_shuffle_ps(t5, t6, 0x1b);
										tmp6 = _mm_shuffle_ps(t5, t6, 0xb1);
										tmp7 = _mm_shuffle_ps(t7, t8, 0x1b);
										tmp8 = _mm_shuffle_ps(t7, t8, 0xb1);

#ifdef DEBUG_SORT
										printv(tmp1, "tmp1");
										printv(tmp2, "tmp2");
#endif
										t1 = _MM_MIN(tmp1, tmp2); 
										t2 = _MM_MAX(tmp1, tmp2); 
										t3 = _MM_MIN(tmp3, tmp4); 
										t4 = _MM_MAX(tmp3, tmp4); 
										t5 = _MM_MIN(tmp5, tmp6); 
										t6 = _MM_MAX(tmp5, tmp6); 
										t7 = _MM_MIN(tmp7, tmp8); 
										t8 = _MM_MAX(tmp7, tmp8); 

										tmp1 = _mm_blend_ps(t1, t2, 0xa);
										tmp2 = _mm_blend_ps(t2, t1, 0xa);
										tmp2 = _mm_shuffle_ps(tmp2, tmp2, 0xb1);
										tmp3 = _mm_blend_ps(t3, t4, 0xa);
										tmp4 = _mm_blend_ps(t4, t3, 0xa);
										tmp4 = _mm_shuffle_ps(tmp4, tmp4, 0xb1);
										tmp5 = _mm_blend_ps(t5, t6, 0xa);
										tmp6 = _mm_blend_ps(t6, t5, 0xa);
										tmp6 = _mm_shuffle_ps(tmp6, tmp6, 0xb1);
										tmp7 = _mm_blend_ps(t7, t8, 0xa);
										tmp8 = _mm_blend_ps(t8, t7, 0xa);
										tmp8 = _mm_shuffle_ps(tmp8, tmp8, 0xb1);

#ifdef DEBUG_SORT
										printv(tmp1, "tmp1");
										printv(tmp2, "tmp2");
#endif
										t1 = _MM_MIN(tmp1, tmp2); 
										t2 = _MM_MAX(tmp1, tmp2); 
										t3 = _MM_MIN(tmp3, tmp4); 
										t4 = _MM_MAX(tmp3, tmp4); 
										t5 = _MM_MIN(tmp5, tmp6); 
										t6 = _MM_MAX(tmp5, tmp6); 
										t7 = _MM_MIN(tmp7, tmp8); 
										t8 = _MM_MAX(tmp7, tmp8); 

										tmp1 = _mm_unpacklo_ps(t1, t2);
										tmp2 = _mm_unpackhi_ps(t1, t2);
										tmp3 = _mm_unpacklo_ps(t3, t4);
										tmp4 = _mm_unpackhi_ps(t3, t4);
										tmp5 = _mm_unpacklo_ps(t5, t6);
										tmp6 = _mm_unpackhi_ps(t5, t6);
										tmp7 = _mm_unpacklo_ps(t7, t8);
										tmp8 = _mm_unpackhi_ps(t7, t8);
#ifdef DEBUG_SORT

										printv(tmp1, "tmp1");
										printv(tmp2, "tmp2");
										printv(tmp3, "tmp3");
										printv(tmp4, "tmp4");
										printv(tmp5, "tmp5");
										printv(tmp6, "tmp6");
										printv(tmp7, "tmp7");
										printv(tmp8, "tmp8");

										printf("store at %d\n", out1);
										printf("store at %d\n", out3);
										printf("store at %d\n", out5);
										printf("store at %d\n\n", out7);
										printf("\n\n");
#endif
										_MM_STORE(to + out1, tmp1);
										_MM_STORE(to + out3, tmp3);
										_MM_STORE(to + out5, tmp5);
										_MM_STORE(to + out7, tmp7);

										get_from_front1 = ((last1 < last2 ) || !(idx2 < chunk2 + chunk_size)) && (idx1 < chunk1 + chunk_size);
										get_from_front3 = ((last3 < last4 ) || !(idx4 < chunk4 + chunk_size)) && (idx3 < chunk3 + chunk_size);
										get_from_front5 = ((last5 < last6 ) || !(idx6 < chunk6 + chunk_size)) && (idx5 < chunk5 + chunk_size);
										get_from_front7 = ((last7 < last8 ) || !(idx8 < chunk8 + chunk_size)) && (idx7 < chunk7 + chunk_size);
								}
#ifdef DEBUG_SORT
								printf("store at %d\n", out1);
								printf("store at %d\n", out3);
								printf("store at %d\n", out5);
								printf("store at %d\n\n", out7);
#endif
								_mm_store_ps(to + out1, tmp2);
								_mm_store_ps(to + out3, tmp4);
								_mm_store_ps(to + out5, tmp6);
								_mm_store_ps(to + out7, tmp8);
						}
				}
				*/
#ifdef DEBUG_SORT
				printArray(bufs[1], BLOCK_SIZE);
				exit(-1);
#endif
#elif AVX
				#define imm1 0x1b // imm for reverse operation
				#define imm2 0x01
				for(i = 0; i < BLOCK_SIZE; i+=64) // for 8 * 8 elements in block
				{
						__m256 f1 = _mm256_load_ps(buffer1 + i);
						__m256 f2 = _mm256_load_ps(buffer1 + i + 8);
						__m256 f3 = _mm256_load_ps(buffer1 + i + 16);
						__m256 f4 = _mm256_load_ps(buffer1 + i + 24);
						__m256 f5 = _mm256_load_ps(buffer1 + i + 32);
						__m256 f6 = _mm256_load_ps(buffer1 + i + 40);
						__m256 f7 = _mm256_load_ps(buffer1 + i + 48);
						__m256 f8 = _mm256_load_ps(buffer1 + i + 56);

						__m256 b1, b2, b3, b4, b5, b6, b7, b8;

						//odd-even merge sort
						b1 = _mm256_min_ps(f1, f2);
						b2 = _mm256_max_ps(f1, f2);
						b3 = _mm256_min_ps(f3, f4);
						b4 = _mm256_max_ps(f3, f4);
						b5 = _mm256_min_ps(f5, f6);
						b6 = _mm256_max_ps(f5, f6);
						b7 = _mm256_min_ps(f7, f8);
						b8 = _mm256_max_ps(f7, f8);

						f1 = _mm256_min_ps(b1, b3);
						f3 = _mm256_max_ps(b1, b3);
						f2 = _mm256_min_ps(b2, b4);
						f4 = _mm256_max_ps(b2, b4);
						f5 = _mm256_min_ps(b5, b7);
						f7 = _mm256_max_ps(b5, b7);
						f6 = _mm256_min_ps(b6, b8);
						f8 = _mm256_max_ps(b6, b8);

						b1 = f1;
						b2 = _mm256_min_ps(f2, f3);
						b3 = _mm256_max_ps(f2, f3);
						b4 = f4;
						b5 = f5;
						b6 = _mm256_min_ps(f6, f7);
						b7 = _mm256_max_ps(f6, f7);
						b8 = f8;

						f1 = _mm256_min_ps(b1, b5); // finish
						f5 = _mm256_max_ps(b1, b5);
						f2 = _mm256_min_ps(b2, b6); 
						f6 = _mm256_max_ps(b2, b6);
						f3 = _mm256_min_ps(b3, b7);
						f7 = _mm256_max_ps(b3, b7);
						f4 = _mm256_min_ps(b4, b8);
						f8 = _mm256_max_ps(b4, b8); // finish

						b2 = f2;
						b3 = _mm256_min_ps(f3, f5);
						b5 = _mm256_max_ps(f3, f5);
						b4 = _mm256_min_ps(f4, f6);
						b6 = _mm256_max_ps(f4, f6);
						b7 = f7;

						f2 = _mm256_min_ps(b2, b3); // finish
						f3 = _mm256_max_ps(b2, b3); // finish
						f4 = _mm256_min_ps(b4, b5); // finish
						f5 = _mm256_max_ps(b4, b5); // finish
						f6 = _mm256_min_ps(b6, b7); // finish
						f7 = _mm256_max_ps(b6, b7); // finish

						//transpose 8 * 8-way SIMD registers
						b1 = _mm256_unpacklo_ps(f1, f2);
						b2 = _mm256_unpackhi_ps(f1, f2);
						b3 = _mm256_unpacklo_ps(f3, f4);
						b4 = _mm256_unpackhi_ps(f3, f4);
						b5 = _mm256_unpacklo_ps(f5, f6);
						b6 = _mm256_unpackhi_ps(f5, f6);
						b7 = _mm256_unpacklo_ps(f7, f8);
						b8 = _mm256_unpackhi_ps(f7, f8);
						f1 = _mm256_shuffle_ps(b1,b3,_MM_SHUFFLE(1,0,1,0));
						f2 = _mm256_shuffle_ps(b1,b3,_MM_SHUFFLE(3,2,3,2));
						f3 = _mm256_shuffle_ps(b2,b4,_MM_SHUFFLE(1,0,1,0));
						f4 = _mm256_shuffle_ps(b2,b4,_MM_SHUFFLE(3,2,3,2));
						f5 = _mm256_shuffle_ps(b5,b7,_MM_SHUFFLE(1,0,1,0));
						f6 = _mm256_shuffle_ps(b5,b7,_MM_SHUFFLE(3,2,3,2));
						f7 = _mm256_shuffle_ps(b6,b8,_MM_SHUFFLE(1,0,1,0));
						f8 = _mm256_shuffle_ps(b6,b8,_MM_SHUFFLE(3,2,3,2));
						b1 = _mm256_permute2f128_ps(f1, f5, 0x20);
						b2 = _mm256_permute2f128_ps(f2, f6, 0x20);
						b3 = _mm256_permute2f128_ps(f3, f7, 0x20);
						b4 = _mm256_permute2f128_ps(f4, f8, 0x20);
						b5 = _mm256_permute2f128_ps(f1, f5, 0x31);
						b6 = _mm256_permute2f128_ps(f2, f6, 0x31);
						b7 = _mm256_permute2f128_ps(f3, f7, 0x31);
						b8 = _mm256_permute2f128_ps(f4, f8, 0x31);


						_mm256_store_ps(buffer1 + i , b1);
						_mm256_store_ps(buffer1 + i + 8, b2);
						_mm256_store_ps(buffer1 + i + 16, b3);
						_mm256_store_ps(buffer1 + i + 24, b4);
						_mm256_store_ps(buffer1 + i + 32, b5);
						_mm256_store_ps(buffer1 + i + 40, b6);
						_mm256_store_ps(buffer1 + i + 48, b7);
						_mm256_store_ps(buffer1 + i + 56, b8);
				}

				step = merge_4way(bufs, BLOCK_SIZE);

				/*
				for(step = 0, chunk_size=VLEN; chunk_size * 4 <= BLOCK_SIZE; step++, chunk_size *= 2)
				{
						//printf("step %d\n", step);

						float * from, *to;
						from = bufs[step % 2];
						to = bufs[(step + 1) % 2];
						int chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, chunk7, chunk8;
						for(
										chunk1 = 0, chunk2 = chunk1 + chunk_size, 
										chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
										chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
										chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size;
										chunk8 < BLOCK_SIZE; 
										chunk1 += chunk_size * 8, chunk2 = chunk1 + chunk_size,
										chunk3 = chunk2 + chunk_size, chunk4 = chunk3 + chunk_size, 
										chunk5 = chunk4 + chunk_size, chunk6 = chunk5 + chunk_size,
										chunk7 = chunk6 + chunk_size, chunk8 = chunk7 + chunk_size

						   )
						{
								//		printf("merge %d & %d....\n", chunk1, chunk2);
								int idx1 = chunk1, idx2, out1;
								int idx3 = chunk3, idx4, out2;
								int idx5 = chunk5, idx6, out3;
								int idx7 = chunk7, idx8, out4;
								float last1, last2;
								float last3, last4;
								float last5, last6;
								float last7, last8;
								__m256 tmp1, tmp2;
								__m256 tmp3, tmp4;
								__m256 tmp5, tmp6;
								__m256 tmp7, tmp8;
								tmp2 = _mm256_load_ps(from + idx1);
								tmp4 = _mm256_load_ps(from + idx3);
								tmp6 = _mm256_load_ps(from + idx5);
								tmp8 = _mm256_load_ps(from + idx7);
								int get_from_front1 = 0;
								int get_from_front2 = 0;
								int get_from_front3 = 0;
								int get_from_front4 = 0;
								last1 = *(from + idx1 + VLEN);
								last3 = *(from + idx3 + VLEN);
								last5 = *(from + idx5 + VLEN);
								last7 = *(from + idx7 + VLEN);
								idx1 += 8;
								idx3 += 8;
								idx5 += 8;
								idx7 += 8;

								for(
												idx2 = chunk2, out1 = chunk1,
												idx4 = chunk4, out2 = chunk3,
												idx6 = chunk6, out3 = chunk5,
												idx8 = chunk8, out4 = chunk7;
												out1 < chunk1 + chunk_size * 2 - 8;
												out1 += VLEN,
												out2 += VLEN,
												out3 += VLEN,
												out4 += VLEN
								   )
								{
										if	(get_from_front1)
										{
												tmp1 = _mm256_load_ps(from + idx1);
												last1 = from[idx1 + VLEN];
												idx1 += VLEN;
										}
										else
										{
												tmp1 = _mm256_load_ps(from + idx2) ;
												last2 = from[idx2 + VLEN];
												idx2 += VLEN;
										}
										if(get_from_front2)
										{
												tmp3 = _mm256_load_ps(from + idx3);
												last3 = from[idx3 + VLEN];
												idx3 += VLEN;
										}
										else
										{
												tmp3 = _mm256_load_ps(from + idx4) ;
												last4 = from[idx4 + VLEN];
												idx4 += VLEN;
										}
										if(get_from_front3)
										{
												tmp5 = _mm256_load_ps(from + idx5);
												last5 = from[idx5 + VLEN];
												idx5 += VLEN;
										}
										else
										{
												tmp5 = _mm256_load_ps(from + idx6) ;
												last6 = from[idx6 + VLEN];
												idx6 += VLEN;
										}
										if(get_from_front4)
										{
												tmp7 = _mm256_load_ps(from + idx7);
												last7 = from[idx7 + VLEN];
												idx7 += VLEN;
										}
										else
										{
												tmp7 = _mm256_load_ps(from + idx8) ;
												last8 = from[idx8 + VLEN];
												idx8 += VLEN;
										}

										__m256 t1, t2;
										__m256 t3, t4;
										__m256 t5, t6;
										__m256 t7, t8;

										//	printf("%d %d %d \n", __LINE__, idx1, idx2);
										tmp2 = _mm256_permute_ps(tmp2, imm1); //reverse tmp2
										tmp2 = _mm256_permute2f128_ps(tmp2, tmp2, imm2); 
										tmp4 = _mm256_permute_ps(tmp4, imm1); //reverse tmp2
										tmp4 = _mm256_permute2f128_ps(tmp4, tmp4, imm2); 
										tmp6 = _mm256_permute_ps(tmp6, imm1); //reverse tmp2
										tmp6 = _mm256_permute2f128_ps(tmp6, tmp6, imm2); 
										tmp8 = _mm256_permute_ps(tmp8, imm1); //reverse tmp2
										tmp8 = _mm256_permute2f128_ps(tmp8, tmp8, imm2); 

										__m256 high1 = _mm256_max_ps(tmp1, tmp2);
										__m256 low1 = _mm256_min_ps(tmp1, tmp2);
										__m256 high2 = _mm256_max_ps(tmp3, tmp4);
										__m256 low2 = _mm256_min_ps(tmp3, tmp4);
										__m256 high3 = _mm256_max_ps(tmp5, tmp6);
										__m256 low3 = _mm256_min_ps(tmp5, tmp6);
										__m256 high4 = _mm256_max_ps(tmp7, tmp8);
										__m256 low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_permute2f128_ps(low1, high1, 0x20);
										tmp2 = _mm256_permute2f128_ps(low1, high1, 0x31);
										tmp3 = _mm256_permute2f128_ps(low2, high2, 0x20);
										tmp4 = _mm256_permute2f128_ps(low2, high2, 0x31);
										tmp5 = _mm256_permute2f128_ps(low3, high3, 0x20);
										tmp6 = _mm256_permute2f128_ps(low3, high3, 0x31);
										tmp7 = _mm256_permute2f128_ps(low4, high4, 0x20);
										tmp8 = _mm256_permute2f128_ps(low4, high4, 0x31);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_shuffle_ps(low1, high1, 0x44);
										tmp2 = _mm256_shuffle_ps(low1, high1, 0xEE);
										tmp3 = _mm256_shuffle_ps(low2, high2, 0x44);
										tmp4 = _mm256_shuffle_ps(low2, high2, 0xEE);
										tmp5 = _mm256_shuffle_ps(low3, high3, 0x44);
										tmp6 = _mm256_shuffle_ps(low3, high3, 0xEE);
										tmp7 = _mm256_shuffle_ps(low4, high4, 0x44);
										tmp8 = _mm256_shuffle_ps(low4, high4, 0xEE);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										tmp1 = _mm256_blend_ps(low1, high1, 0xAA);
										t1 = _mm256_blend_ps(high1, low1, 0xAA);
										tmp2 = _mm256_shuffle_ps(t1, t1, 0XB1);
										tmp3 = _mm256_blend_ps(low2, high2, 0xAA);
										t3 = _mm256_blend_ps(high2, low2, 0xAA);
										tmp4 = _mm256_shuffle_ps(t3, t3, 0XB1);
										tmp5 = _mm256_blend_ps(low3, high3, 0xAA);
										t5 = _mm256_blend_ps(high3, low3, 0xAA);
										tmp6 = _mm256_shuffle_ps(t5, t5, 0XB1);
										tmp7 = _mm256_blend_ps(low4, high4, 0xAA);
										t7 = _mm256_blend_ps(high4, low4, 0xAA);
										tmp8 = _mm256_shuffle_ps(t7, t7, 0XB1);

										high1 = _mm256_max_ps(tmp1, tmp2);
										low1 = _mm256_min_ps(tmp1, tmp2);
										high2 = _mm256_max_ps(tmp3, tmp4);
										low2 = _mm256_min_ps(tmp3, tmp4);
										high3 = _mm256_max_ps(tmp5, tmp6);
										low3 = _mm256_min_ps(tmp5, tmp6);
										high4 = _mm256_max_ps(tmp7, tmp8);
										low4 = _mm256_min_ps(tmp7, tmp8);

										t1 = _mm256_unpacklo_ps(low1, high1);
										t2 = _mm256_unpackhi_ps(low1, high1);
										t3 = _mm256_unpacklo_ps(low2, high2);
										t4 = _mm256_unpackhi_ps(low2, high2);
										t5 = _mm256_unpacklo_ps(low3, high3);
										t6 = _mm256_unpackhi_ps(low3, high3);
										t7 = _mm256_unpacklo_ps(low4, high4);
										t8 = _mm256_unpackhi_ps(low4, high4);

										tmp1 = _mm256_permute2f128_ps(t1, t2, 0x20);
										tmp2 = _mm256_permute2f128_ps(t1, t2, 0x31);
										tmp3 = _mm256_permute2f128_ps(t3, t4, 0x20);
										tmp4 = _mm256_permute2f128_ps(t3, t4, 0x31);
										tmp5 = _mm256_permute2f128_ps(t5, t6, 0x20);
										tmp6 = _mm256_permute2f128_ps(t5, t6, 0x31);
										tmp7 = _mm256_permute2f128_ps(t7, t8, 0x20);
										tmp8 = _mm256_permute2f128_ps(t7, t8, 0x31);

										_mm256_store_ps(to + out1, tmp1);
										_mm256_store_ps(to + out2, tmp3);
										_mm256_store_ps(to + out3, tmp5);
										_mm256_store_ps(to + out4, tmp7);

										get_from_front1 = ((last1 < last2 ) || !(idx2 < chunk2 + chunk_size)) && (idx1 < chunk1 + chunk_size);
										get_from_front2 = ((last3 < last4 ) || !(idx4 < chunk4 + chunk_size)) && (idx3 < chunk3 + chunk_size);
										get_from_front3 = ((last5 < last6 ) || !(idx6 < chunk6 + chunk_size)) && (idx5 < chunk5 + chunk_size);
										get_from_front4 = ((last7 < last8 ) || !(idx8 < chunk8 + chunk_size)) && (idx7 < chunk7 + chunk_size);

								}
								_mm256_store_ps(to + out1, tmp2);
								_mm256_store_ps(to + out2, tmp4);
								_mm256_store_ps(to + out3, tmp6);
								_mm256_store_ps(to + out4, tmp8);
						}
				}
		*/
				//exit(-1);
#elif __MIC__

		const __m512i offset1 = _mm512_set_epi32(8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0);
		const __m512i offset2 = _mm512_set_epi32(7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15);
		const __m512i offset3 = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
				//in-register sort
				for(i = 0; i + 255 <  BLOCK_SIZE; i+=256) // for 16 * 16 elements in block
				{
						_VECTOR f1 = _MM_LOAD(buffer1 + i );
						_VECTOR f2 = _MM_LOAD(buffer1 + i + 1 * VLEN);
						_VECTOR f3 = _MM_LOAD(buffer1 + i + 2 * VLEN);
						_VECTOR f4 = _MM_LOAD(buffer1 + i + 3 * VLEN);
						_VECTOR f5 = _MM_LOAD(buffer1 + i + 4 * VLEN);
						_VECTOR f6 = _MM_LOAD(buffer1 + i + 5 * VLEN);
						_VECTOR f7 = _MM_LOAD(buffer1 + i + 6 * VLEN);
						_VECTOR f8 = _MM_LOAD(buffer1 + i + 7 * VLEN);
						_VECTOR f9 = _MM_LOAD(buffer1 + i + 8 * VLEN);
						_VECTOR f10 = _MM_LOAD(buffer1 + i + 9 * VLEN);
						_VECTOR f11 = _MM_LOAD(buffer1 + i + 10 * VLEN);
						_VECTOR f12 = _MM_LOAD(buffer1 + i + 11 * VLEN);
						_VECTOR f13 = _MM_LOAD(buffer1 + i + 12 * VLEN);
						_VECTOR f14 = _MM_LOAD(buffer1 + i + 13 * VLEN);
						_VECTOR f15 = _MM_LOAD(buffer1 + i + 14 * VLEN);
						_VECTOR f16 = _MM_LOAD(buffer1 + i + 15 * VLEN);

						//pairwise cmparison with 16 input

						_VECTOR b1 = _MM_MIN(f1, f2);
						_VECTOR b2 = _MM_MAX(f1, f2);
						_VECTOR b3 = _MM_MIN(f3, f4);
						_VECTOR b4 = _MM_MAX(f3, f4);
						_VECTOR b5 = _MM_MIN(f5, f6);
						_VECTOR b6 = _MM_MAX(f5, f6);
						_VECTOR b7 = _MM_MIN(f7, f8);
						_VECTOR b8 = _MM_MAX(f7, f8);
						_VECTOR b9 = _MM_MIN(f9, f10);
						_VECTOR b10= _MM_MAX(f9, f10);
						_VECTOR b11= _MM_MIN(f11, f12);
						_VECTOR b12= _MM_MAX(f11, f12);
						_VECTOR b13= _MM_MIN(f13, f14);
						_VECTOR b14= _MM_MAX(f13, f14);
						_VECTOR b15= _MM_MIN(f15, f16);
						_VECTOR b16= _MM_MAX(f15, f16);

						f1 = _MM_MIN(b1, b3);
						f3 = _MM_MAX(b1, b3);
						f2 = _MM_MIN(b2, b4);
						f4 = _MM_MAX(b2, b4);
						f5 = _MM_MIN(b5, b7);
						f7 = _MM_MAX(b5, b7);
						f6 = _MM_MIN(b6, b8);
						f8 = _MM_MAX(b6, b8);
						f9 = _MM_MIN(b9, b11);
						f11 = _MM_MAX(b9, b11);
						f10 = _MM_MIN(b10, b12);
						f12 = _MM_MAX(b10, b12);
						f13 = _MM_MIN(b13, b15);
						f15 = _MM_MAX(b13, b15);
						f14 = _MM_MIN(b14, b16);
						f16 = _MM_MAX(b14, b16);

						b1 = _MM_MIN(f1, f5);
						b5 = _MM_MAX(f1, f5);
						b2 = _MM_MIN(f2, f6);
						b6 = _MM_MAX(f2, f6);
						b3 = _MM_MIN(f3, f7);
						b7 = _MM_MAX(f3, f7);
						b4 = _MM_MIN(f4, f8);
						b8 = _MM_MAX(f4, f8);
						b9 = _MM_MIN(f9, f13);
						b13 = _MM_MAX(f9, f13);
						b10 = _MM_MIN(f10, f14);
						b14 = _MM_MAX(f10, f14);
						b11 = _MM_MIN(f11, f15);
						b15 = _MM_MAX(f11, f15);
						b12 = _MM_MIN(f12, f16);
						b16 = _MM_MAX(f12, f16);

						f1 = _MM_MIN(b1, b9);
						f9 = _MM_MAX(b1, b9);
						f2 = _MM_MIN(b2, b10);
						f10 = _MM_MAX(b2, b10);
						f3 = _MM_MIN(b3, b11);
						f11 = _MM_MAX(b3, b11);
						f4 = _MM_MIN(b4, b12);
						f12 = _MM_MAX(b4, b12);
						f5 = _MM_MIN(b5, b13);
						f13 = _MM_MAX(b5, b13);
						f6 = _MM_MIN(b6, b14);
						f14 = _MM_MAX(b6, b14);
						f7 = _MM_MIN(b7, b15);
						f15 = _MM_MAX(b7, b15);
						f8 = _MM_MIN(b8, b16);
						f16 = _MM_MAX(b8, b16);

						//f1, f16 finished

						b2 = f2;
						b3 = f3;
						b4 = f4;
						b5 = _MM_MIN(f5, f9);
						b9 = _MM_MAX(f5, f9);
						b6 = _MM_MIN(f6, f10);
						b10 = _MM_MAX(f6, f10);
						b7 = _MM_MIN(f7, f11);
						b11 = _MM_MAX(f7, f11);
						b8 = _MM_MIN(f8, f12);
						b12 = _MM_MAX(f8, f12);
						b13 = f13;
						b14 = f14;
						b15 = f15;

						f3 = _MM_MIN(b3, b9);
						f9 = _MM_MAX(b3, b9);
						f4 = _MM_MIN(b4, b10);
						f10 = _MM_MAX(b4, b10);
						f5 = b5;
						f6 = b6;
						f7 = _MM_MIN(b7, b13);
						f13 = _MM_MAX(b7, b13);
						f8 = _MM_MIN(b8, b14);
						f14 = _MM_MAX(b8, b14);
						f11 = b11;
						f12 = b12;

						b3 = _MM_MIN(f3, f5);
						b5 = _MM_MAX(f3, f5);
						b4 = _MM_MIN(f4, f6);
						b6 = _MM_MAX(f4, f6);
						b7 = _MM_MIN(f7, f9);
						b9 = _MM_MAX(f7, f9);
						b8 = _MM_MIN(f8, f10);
						b10 = _MM_MAX(f8, f10);
						b11 = _MM_MIN(f11, f13);
						b13 = _MM_MAX(f11, f13);
						b12 = _MM_MIN(f12, f14);
						b14 = _MM_MAX(f12, f14);

						f2 = _MM_MIN(b2, b9);
						f9 = _MM_MAX(b2, b9);
						f4 = _MM_MIN(b4, b11);
						f11 = _MM_MAX(b4, b11);
						f6 = _MM_MIN(b6, b13);
						f13 = _MM_MAX(b6, b13);
						f8 = _MM_MIN(b8, b15);
						f15 = _MM_MAX(b8, b15);
						f3 = b3;
						f5 = b5;
						f7 = b7;
						f10 = b10;
						f12 = b12;
						f14 = b14;

						b2 = _MM_MIN(f2, f5);
						b5 = _MM_MAX(f2, f5);
						b3 = f3;
						b6 = _MM_MIN(f6, f9);
						b9 = _MM_MAX(f6, f9);
						b10 = _MM_MIN(f10, f13);
						b13 = _MM_MAX(f10, f13);
						b4 = _MM_MIN(f4, f7);
						b7 = _MM_MAX(f4, f7);
						b8 = _MM_MIN(f8, f11);
						b11 = _MM_MAX(f8, f11);
						b14 = f14;
						b12 = _MM_MIN(f12, f15);
						b15 = _MM_MAX(f12, f15);

						f2 = _MM_MIN(b2, b3);
						f3 = _MM_MAX(b2, b3);
						f4 = _MM_MIN(b4, b5);
						f5 = _MM_MAX(b4, b5);
						f6 = _MM_MIN(b6, b7);
						f7 = _MM_MAX(b6, b7);
						f8 = _MM_MIN(b8, b9);
						f9 = _MM_MAX(b8, b9);
						f10 = _MM_MIN(b10, b11);
						f11 = _MM_MAX(b10, b11);
						f12 = _MM_MIN(b12, b13);
						f13 = _MM_MAX(b12, b13);
						f14 = _MM_MIN(b14, b15);
						f15 = _MM_MAX(b14, b15);

						_MM_STORE(buffer1 + i, f1 );
						_MM_STORE(buffer1 + i + 1 * VLEN, f2 );
						_MM_STORE(buffer1 + i + 2 * VLEN, f3 );
						_MM_STORE(buffer1 + i + 3 * VLEN, f4 );
						_MM_STORE(buffer1 + i + 4 * VLEN, f5 );
						_MM_STORE(buffer1 + i + 5 * VLEN, f6 );
						_MM_STORE(buffer1 + i + 6 * VLEN, f7 );
						_MM_STORE(buffer1 + i + 7 * VLEN, f8 );
						_MM_STORE(buffer1 + i + 8 * VLEN, f9 );
						_MM_STORE(buffer1 + i + 9 * VLEN, f10);
						_MM_STORE(buffer1 + i + 10 * VLEN, f11);
						_MM_STORE(buffer1 + i + 11 * VLEN, f12);
						_MM_STORE(buffer1 + i + 12 * VLEN, f13);
						_MM_STORE(buffer1 + i + 13 * VLEN, f14);
						_MM_STORE(buffer1 + i + 14 * VLEN, f15);
						_MM_STORE(buffer1 + i + 15 * VLEN, f16);

						TRANSPOSE_ARRAY(buffer1 + i)


				}

				step = merge_4way(bufs, BLOCK_SIZE);
/*
				for(step = 0, chunk_size=VLEN; chunk_size <= BLOCK_SIZE; step++, chunk_size *= 2)
				{
						float * from, *to;
						from = bufs[step % 2];
						to = bufs[(step + 1) % 2];
						int chunk1, chunk2;
						int get_from_front1 = 0;
						for(
										chunk1 = 0, chunk2 = chunk_size;
										chunk2 < BLOCK_SIZE;
										chunk1 += chunk_size * 2, chunk2 += chunk_size * 2
						   )
						{
								int idx1= chunk1, idx2, out1;
								_VECTOR tmp1, tmp2 = _MM_LOAD(from + idx1); 
								tmp2 = _mm512_permutevar_epi32(offset3, tmp2); // reverse
								float last1 = *(from + idx1 + VLEN), last2;
								idx1 += VLEN;
								//printf("chunk1 %d chunk2 %d\n", chunk1, chunk2);

								for( 
												idx2 = chunk2, out1 = chunk1;
												out1 < chunk1 + chunk_size * 2 - VLEN;
												out1 += VLEN
								   )
								{
										//break;
										if(get_from_front1)
										{
												tmp1 = _MM_LOAD(from + idx1);
												last1 = from[idx1 + VLEN];
												idx1 += VLEN;
										}
										else
										{
												tmp1 = _MM_LOAD(from + idx2);
												last2 = from[idx2 + VLEN];
												idx2 += VLEN;
										}

//#define DEBUG_SORT
										//merge tmp1, tmp2 here
										//reverse t2
#ifdef DEBUG_SORT
										printf("merge %d tmp1\n", idx1);
										printf("merge %d tmp2\n", idx2);
										printv(tmp1, "input1");
										printv(tmp2, "input2");

#endif
										_VECTOR t1 = _MM_MIN(tmp1, tmp2);
										_VECTOR t2 = _MM_MAX(tmp1, tmp2);

										tmp1 = _mm512_mask_permute4f128_epi32(t1, 0x00ff, t2, _MM_PERM_BADC);
										tmp2 = _mm512_mask_permute4f128_epi32(t2, 0xff00, t1, _MM_PERM_BADC);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);

										tmp1 = _mm512_mask_permute4f128_epi32(t1, 0x0f0f, t2, _MM_PERM_CDAB);
										tmp2 = _mm512_mask_permute4f128_epi32(t2, 0xf0f0, t1, _MM_PERM_CDAB);
#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);

										tmp1 = _mm512_mask_swizzle_ps(t1, 0x3333, t2, _MM_SWIZ_REG_BADC);
										tmp2 = _mm512_mask_swizzle_ps(t2, 0xcccc, t1, _MM_SWIZ_REG_BADC);
										//tmp1 = _mm512_mask_swizzle_ps(t1, 0xffff, t2, _MM_PERM_BADC);
										//tmp2 = _mm512_mask_swizzle_ps(t2, 0xffff, t1, _MM_PERM_BADC);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);

										tmp1 = _mm512_mask_swizzle_ps(t1, 0x5555, t2, _MM_SWIZ_REG_CDAB);
										tmp2 = _mm512_mask_swizzle_ps(t2, 0xaaaa, t1, _MM_SWIZ_REG_CDAB);

#ifdef DEBUG_SORT
										printv(tmp1, "input1");
										printv(tmp2, "input2");
										printf("\n");
#endif
										t1 = _MM_MIN(tmp1, tmp2);
										t2 = _MM_MAX(tmp1, tmp2);

										t1 = _mm512_permutevar_epi32(offset2, t1);
										t2 = _mm512_permutevar_epi32(offset1, t2);
										tmp2 = _mm512_mask_mov_ps(t1,0x5555,t2);
										tmp1 = _mm512_mask_mov_ps(t2,0x5555,t1);

#ifdef DEBUG_SORT
										printv(tmp1, "output1");
										printv(tmp2, "output2");
										printf("\n");
#endif

										
										_MM_STORE(to + out1, tmp1);

										get_from_front1 = ((last1 < last2 ) || !(idx2 < chunk2 + chunk_size)) && (idx1 < chunk1 + chunk_size);
								}
								tmp2 = _mm512_permutevar_epi32(offset3, tmp2); // reverse
								_MM_STORE(to + out1, tmp2);
						}
				}
				*/
#endif
continue;
				float * sorted_data = bufs[((step + 1) % 2)]; 
				int rangecnt, bincnt;
				float cmp, tmp;
				int o;
				//printf("hello %d\n", BLOCK_SIZE);

#ifdef __MIC__
for(o = 0; o < BLOCK_SIZE; o+=BLOCK_SIZE/2)
				{
				for(i = o, bincnt = 0, rangecnt = 1, cmp = boundary[rangecnt]; i < o + BLOCK_SIZE/2 ; i++, bincnt++)
				{
								//printf("%f\n", sorted_data[i]);
								if(((tmp = sorted_data[i]) >= cmp))
								{
												bin[rangecnt-1] += bincnt;
												bincnt = 0;
												for(;tmp >= (cmp = boundary[++rangecnt]) ;);
								}
				}
				bin[rangecnt-1] += bincnt;
				}

/*
				for(i = 0, bincnt = 0, rangecnt = 1, cmp = boundary[rangecnt]; i < BLOCK_SIZE ; i++, bincnt++)
				{
				//				printf("%f\n", sorted_data[i]);
								if(((tmp = sorted_data[i]) >= cmp))
								{
												bin[rangecnt-1] += bincnt;
												bincnt = 0;
												for(;tmp >= (cmp = boundary[++rangecnt]) ;);
								}
				}
				bin[rangecnt-1] += bincnt;

*/
#else
				for(o = 0; o < BLOCK_SIZE; o+=BLOCK_SIZE/4)
				{
				for(i = o, bincnt = 0, rangecnt = 1, cmp = boundary[rangecnt]; i < o + BLOCK_SIZE/4 ; i++, bincnt++)
				{
								//printf("%f\n", sorted_data[i]);
								if(((tmp = sorted_data[i]) >= cmp))
								{
												bin[rangecnt-1] += bincnt;
												bincnt = 0;
												for(;tmp >= (cmp = boundary[++rangecnt]) ;);
								}
				}
				bin[rangecnt-1] += bincnt;
				}
#endif
		}

		_mm_free(bufs[0]);
		_mm_free(bufs[1]);

		return 0;
}

#endif
