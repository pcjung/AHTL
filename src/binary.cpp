#include "binary.h"
#include "simd.h"
#include "float.h"

#define EPS 1e-3
#ifdef SIMD
#ifdef __MIC__


//#define DEBUG_TREE_BUILD
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

static void printv_ps(__m512 v, char *str)
{
  int i;
  __declspec(align(64)) float tmp[16];
  printf("%s:", str);
  _mm512_store_ps(tmp, v);
  for(i=0; i < 16; i++)
  {
    tmp[0] = tmp[i];
    printf("[%d]=%.3f ", i, tmp[0]);
  }
  printf("\n");
}

#endif
#endif

float * hist_build_tree(float * _boundary, unsigned int _bin_count)
{
#ifdef USE_GATHER_SCATTER
  float * __boundary = _mm_malloc(sizeof(float) *(_bin_count + 1), 64); 
  memcpy(__boundary, _boundary, sizeof(float) * (_bin_count + 1));
  return __boundary;
#endif
#ifdef SIMD
  int i, j, simd=0;


  int N2bin_count; // complete binary tree size

  for(N2bin_count=1; N2bin_count < _bin_count; N2bin_count *= VLEN); 

  float * boundary = (float *)_mm_malloc(sizeof(float) * N2bin_count, 64); 
  memcpy(boundary, _boundary, sizeof(float) * _bin_count);

  //float largest = _boundary[_bin_count] + EPS;
  float largest = FLT_MAX;
  for(i = _bin_count; i < N2bin_count; i++) //init the empty tree element with the largest bin boundary
    boundary[i] = largest;	

  float * simd_pack = (float *)_mm_malloc(sizeof(float) * N2bin_count * 2, 4096); //simd packed tree
  memset(simd_pack, 0, sizeof(float) * N2bin_count * 2); //size of simd packed tree is larger than bin.

  int H_OFFSET; //height of small tree with VLEN - 1 element
  for(H_OFFSET = 0, i=1; i != VLEN; i *= 2, H_OFFSET++); 

  int middle = N2bin_count / 2; // root idx

  //first, make heap-like BST


  int HEIGHT; // height of BST
  for(HEIGHT = 0, i=N2bin_count; i > 0; i /= 2, HEIGHT++); 


  unsigned int sortidx[VLEN];
#ifdef SSE
  sortidx[0] = 1;
  sortidx[1] = 0;
  sortidx[2] = 2;
#elif AVX
  sortidx[0] = 3;
  sortidx[1] = 1;
  sortidx[2] = 5;
  sortidx[3] = 0;
  sortidx[4] = 2;
  sortidx[5] = 4;
  sortidx[6] = 6;
#else // MIC
  sortidx[0] = 7;
  sortidx[1] = 3;
  sortidx[2] = 11;
  sortidx[3] = 1;
  sortidx[4] = 5;
  sortidx[5] = 9;
  sortidx[6] = 13;
  sortidx[7] = 0;
  sortidx[8] = 2;
  sortidx[9] = 4;
  sortidx[10] = 6;
  sortidx[11] = 8;
  sortidx[12] = 10;
  sortidx[13] = 12;
  sortidx[14] = 14;
#endif


//#define DEBUG_TREE_BUILD
  int global_height, global_offset, global_from, global_count, global_by;
  int local_height, local_offset, local_from, local_to, local_by, local_count, li;

  for(global_height = 0, global_offset = middle, global_from = middle, global_count = 1, global_by = global_offset * 2;
      global_height < HEIGHT - 1;
      global_height += H_OFFSET, global_by /= VLEN, global_count *= VLEN, global_offset /= VLEN
     )//set vars for pointing subtrees
  {
    for(i = global_from; i < global_from + global_by * global_count; i += global_by) // point 1 subtree 
    {
      li = 0;

      for(local_height = 0, local_from = i, local_by = global_by, local_count = 1;
          local_height < H_OFFSET;
          local_height++, local_by /= 2, local_from -= local_by / 2, local_count *= 2) // add one row per loop
      {
        for(j = local_from; j < local_from + local_count * local_by; j += local_by)
        {
          simd_pack[simd + sortidx[li++]] = boundary[j];
#ifdef DEBUG_TREE_BUILD
          printf("li=%d : from %d to %d %f\n",li,j,simd+sortidx[li-1], boundary[j]);
          fflush(stdout);
#endif
          boundary[j] = 0;
        }
      }

      simd_pack[simd + li++] = largest; // for aligning
#ifdef DEBUG_TREE_BUILD
      printf("!!\n");
#endif
      simd+=VLEN;
    }

    for(i = VLEN; i > 1; i /=2)
      global_from -= global_offset / i;
  }


#ifdef DEBUG_TREE_BUILD
  for(i=0;i<N2bin_count;i++)
  {
    printf("%d %f \n", i, simd_pack[i]);
  }
#endif

  _mm_free (boundary);

  return simd_pack;
#else
  float * boundary = (float *)_mm_malloc(sizeof(float) * _bin_count, 64); 
  memcpy(boundary, _boundary, sizeof(float) * _bin_count);
  return boundary;
#endif
}
#ifdef SIMD
int hist_binary_float_simd 
(
 float * data, //data should be aligned
 float * _boundary,
 unsigned int count,
 unsigned int * bin,
 unsigned int _bin_count,
 float * simd_pack
 )
{
  // First, make binary tree structure


  //unsigned int bin[1024]={0};
  //printf("%llx\n", bin);

  int i, j, simd=0;

  int N2bin_count; // complete binary tree size

  for(N2bin_count=1; N2bin_count < _bin_count; N2bin_count *= VLEN); 

  //		printf("N2bin_count = %d, count=%d\n", N2bin_count, count);


  float * boundary = (float *)_mm_malloc(sizeof(float) * N2bin_count, 64); 
  memcpy(boundary, _boundary, sizeof(float) * _bin_count);

  float largest = _boundary[_bin_count] + EPS;
  for(i = _bin_count; i < N2bin_count; i++) //init the empty tree element with the largest bin boundary
    boundary[i] = largest;	


  int H_OFFSET; //height of small tree with VLEN - 1 element
  for(H_OFFSET = 0, i=1; i != VLEN; i *= 2, H_OFFSET++); 

  int HEIGHT; // height of BST
  for(HEIGHT = 0, i=N2bin_count; i > 0; i /= 2, HEIGHT++); 



#ifdef SSE
  unsigned int table[8]={0};
  table[0] = 0;
  table[1] = 1;
  table[3] = 2;
  table[7] = 3;
#elif AVX
  unsigned int table[128];
  table[0] = 0;
  table[1] = 1;
  table[3] = 2;
  table[7] = 3;
  table[15] = 4;
  table[31] = 5;
  table[63] = 6;
  table[127] = 7;
  /*
     table[0] = 0;
     table[8] = 1;
     table[10] = 2;
     table[26] = 3;
     table[27] = 4;
     table[59] = 5;
     table[63] = 6;
     table[127] = 7;
     */
#else // MIC
  /*
     unsigned int table[256];
     table[0] = 0;
     table[1] = 1;
     table[3] = 2;
     table[7] = 3;
     table[15] = 4;
     table[31] = 5;
     table[63] = 6;
     table[127] = 7;
     table[255] = 8;
     unsigned int table[32768];
     table[0] = 0;
     table[1] = 1;
     table[3] = 2;
     table[7] = 3;
     table[15] = 4;
     table[31] = 5;
     table[63] = 6;
     table[127] = 7;
     table[255] = 8;
     table[511] = 9;
     table[1023] = 10;
     table[2047] = 11;
     table[4095] = 12;
     table[8191] = 13;
     table[16383] = 14;
     table[32767] = 15;
     */
#endif

  //#define DEBUG_BIN_SEARCH
  unsigned int idx, tst, curidx, tmpForReduction = 0 ;
  //#define PRECALCULATED_IDX
#ifdef PRECALCULATED_IDX
  unsigned int tsts[16];
  for(i = 0, tst=0 ; i < HEIGHT / H_OFFSET; i++)
  {
    tst = VLEN * tst + VLEN;
    tsts[i] = tst;
  }
#endif
#ifdef DEBUG_BIN_SEARCH
  printf("%d iteration per element\n", HEIGHT / H_OFFSET);
#endif
  float x;
  //#pragma noprefetch data
  //#pragma unroll(2)
#define MANUAL_UNROLL
#ifdef MANUAL_UNROLL
  float x1, x2, x3, x4;
  unsigned int idx1, idx2, idx3, idx4;
  unsigned int curidx1, curidx2, curidx3, curidx4;
  _VECTOR xmm_range = _MM_LOAD(simd_pack);
  for(i = 0; i + 3 < count; i+=4)
  {
    x1 = data[i];
    x2 = data[i+1];
    x3 = data[i+2];
    x4 = data[i+3];
    _VECTOR item1 = _MM_SET1(x1);
    _VECTOR item2 = _MM_SET1(x2);
    _VECTOR item3 = _MM_SET1(x3);
    _VECTOR item4 = _MM_SET1(x4);
    tst = VLEN;
#ifdef __MIC__
    curidx1 = _mm512_cmp_ps_mask(item1, xmm_range , _CMP_GE_OS);
    idx1 = curidx1 = _mm_countbits_32(curidx1);
    curidx2 = _mm512_cmp_ps_mask(item2, xmm_range , _CMP_GE_OS);
    idx2 = curidx2 = _mm_countbits_32(curidx2);
    curidx3 = _mm512_cmp_ps_mask(item3, xmm_range , _CMP_GE_OS);
    idx3 = curidx3 = _mm_countbits_32(curidx3);
    curidx4 = _mm512_cmp_ps_mask(item4, xmm_range , _CMP_GE_OS);
    idx4 = curidx4 = _mm_countbits_32(curidx4);

#elif SSE
    curidx1 = _mm_movemask_ps( _mm_cmpge_ps(item1, xmm_range ));
    curidx2 = _mm_movemask_ps( _mm_cmpge_ps(item2, xmm_range ));
    curidx3 = _mm_movemask_ps( _mm_cmpge_ps(item3, xmm_range ));
    curidx4 = _mm_movemask_ps( _mm_cmpge_ps(item4, xmm_range ));
    idx1 = curidx1 = table[curidx1];
    idx2 = curidx2 = table[curidx2];
    idx3 = curidx3 = table[curidx3];
    idx4 = curidx4 = table[curidx4];
#elif AVX
    curidx1 = _mm256_movemask_ps( _mm256_cmp_ps(item1, xmm_range , _CMP_GE_OS));
    curidx2 = _mm256_movemask_ps( _mm256_cmp_ps(item2, xmm_range , _CMP_GE_OS));
    curidx3 = _mm256_movemask_ps( _mm256_cmp_ps(item3, xmm_range , _CMP_GE_OS));
    curidx4 = _mm256_movemask_ps( _mm256_cmp_ps(item4, xmm_range , _CMP_GE_OS));

    idx1 = curidx1 = table[curidx1];
    idx2 = curidx2 = table[curidx2];
    idx3 = curidx3 = table[curidx3];
    idx4 = curidx4 = table[curidx4];
#endif
    for(j = 1; j < HEIGHT / H_OFFSET; j++)
    {
#ifdef __MIC__
      _VECTOR xmm_range1 = _MM_LOAD(simd_pack + tst + idx1 * VLEN);
      _VECTOR xmm_range2 = _MM_LOAD(simd_pack + tst + idx2 * VLEN);
      _VECTOR xmm_range3 = _MM_LOAD(simd_pack + tst + idx3 * VLEN);
      _VECTOR xmm_range4 = _MM_LOAD(simd_pack + tst + idx4 * VLEN);
      tst = VLEN * tst + VLEN;
      curidx1 = _mm512_cmp_ps_mask(item1, xmm_range1, _CMP_GE_OS);
      curidx1 = _mm_countbits_32(curidx1);
      curidx2 = _mm512_cmp_ps_mask(item2, xmm_range2, _CMP_GE_OS);
      idx1 = VLEN * idx1 + curidx1;
      curidx2 = _mm_countbits_32(curidx2);
      curidx3 = _mm512_cmp_ps_mask(item3, xmm_range3, _CMP_GE_OS);
      idx2 = VLEN * idx2 + curidx2;
      curidx3 = _mm_countbits_32(curidx3);
      curidx4 = _mm512_cmp_ps_mask(item4, xmm_range4, _CMP_GE_OS);
      idx3 = VLEN * idx3 + curidx3;
      curidx4 = _mm_countbits_32(curidx4);
      idx4 = VLEN * idx4 + curidx4;

#elif SSE
      _VECTOR xmm_range1 = _MM_LOAD(simd_pack + tst + idx1 * VLEN);
      _VECTOR xmm_range2 = _MM_LOAD(simd_pack + tst + idx2 * VLEN);
      _VECTOR xmm_range3 = _MM_LOAD(simd_pack + tst + idx3 * VLEN);
      _VECTOR xmm_range4 = _MM_LOAD(simd_pack + tst + idx4 * VLEN);

      curidx1 = _mm_movemask_ps( _mm_cmpge_ps(item1, xmm_range1));
      curidx2 = _mm_movemask_ps( _mm_cmpge_ps(item2, xmm_range2));
      curidx3 = _mm_movemask_ps( _mm_cmpge_ps(item3, xmm_range3));
      curidx4 = _mm_movemask_ps( _mm_cmpge_ps(item4, xmm_range4));

      curidx1 = table[curidx1];
      curidx2 = table[curidx2];
      curidx3 = table[curidx3];
      curidx4 = table[curidx4];

      tst = VLEN * tst + VLEN;

      idx1 = VLEN * idx1 + curidx1;
      idx2 = VLEN * idx2 + curidx2;
      idx3 = VLEN * idx3 + curidx3;
      idx4 = VLEN * idx4 + curidx4;
#elif AVX
      _VECTOR xmm_range1 = _MM_LOAD(simd_pack + tst + idx1 * VLEN);
      _VECTOR xmm_range2 = _MM_LOAD(simd_pack + tst + idx2 * VLEN);
      _VECTOR xmm_range3 = _MM_LOAD(simd_pack + tst + idx3 * VLEN);
      _VECTOR xmm_range4 = _MM_LOAD(simd_pack + tst + idx4 * VLEN);

      curidx1 = _mm256_movemask_ps( _mm256_cmp_ps(item1, xmm_range1, _CMP_GE_OS));
      curidx2 = _mm256_movemask_ps( _mm256_cmp_ps(item2, xmm_range2, _CMP_GE_OS));
      curidx3 = _mm256_movemask_ps( _mm256_cmp_ps(item3, xmm_range3, _CMP_GE_OS));
      curidx4 = _mm256_movemask_ps( _mm256_cmp_ps(item4, xmm_range4, _CMP_GE_OS));

      curidx1 = table[curidx1];
      curidx2 = table[curidx2];
      curidx3 = table[curidx3];
      curidx4 = table[curidx4];

      tst = VLEN * tst + VLEN;

      idx1 = VLEN * idx1 + curidx1;
      idx2 = VLEN * idx2 + curidx2;
      idx3 = VLEN * idx3 + curidx3;
      idx4 = VLEN * idx4 + curidx4;
#endif

    }
    bin[idx1]++;
    bin[idx2]++;
    bin[idx3]++;
    bin[idx4]++;
  }
#else
  for(i = 0; i < count; i++)
  {
    x = data[i];
    //if(!(i & 0xf))
    //	_mm_prefetch((char const *)(data + i + 512), _MM_HINT_T0 );
    // x =  (float) ((i * 47) % 256);
    _VECTOR item = _MM_SET1(x);
    idx = tst = curidx = 0;
#ifdef DEBUG_BIN_SEARCH
    printf("%dth input %f\n",i,x);
#endif
    for(j = 0; j < HEIGHT / H_OFFSET; j++)
    {
      _VECTOR xmm_range = _MM_LOAD(simd_pack + tst + idx * VLEN);
      //printf("hello world %f %f %f %f\n", simd_pack[0], simd_pack[1], simd_pack[2], simd_pack[3]);
      //printf("tree idx : %d, data %f, idx=%d, %f\n", tst + idx*VLEN, x,idx, simd_pack[tst + idx*VLEN]);
#ifdef SSE
      curidx = _mm_movemask_ps( _mm_cmpge_ps(item, xmm_range));
#elif AVX
      curidx = _mm256_movemask_ps( _mm256_cmp_ps(item, xmm_range, _CMP_GE_OS));
#else
      curidx = _mm512_cmp_ps_mask(item, xmm_range, _CMP_GE_OS);
#endif
#ifdef DEBUG_BIN_SEARCH
      printf("%d : cmpresult=%d\n",j,curidx);
#endif
      /*
#ifdef __MIC__
curidx = table[curidx>>8] + table[curidx % 256];
#else
*/
      curidx = table[curidx];
      //#endif
#ifdef DEBUG_BIN_SEARCH
      printf("%d : localidx=%d\n",j,curidx);
#endif

#ifdef PRECALCULATED_IDX
      tst = tsts[j];
#else
      tst = VLEN * tst + VLEN;
#endif
      idx = VLEN * idx + curidx;
#ifdef DEBUG_BIN_SEARCH
      printf("%d : globalidx=%d\n",j,idx);
      fflush(stdout);
#endif
    }
#ifdef NO_UPDATE
    tmpForReduction += idx;
#else
    //printf("idx : %d\n", idx);
    bin[idx]++;
#endif
    //if(i==65536)break;
  }
#ifdef NO_UPDATE
  bin[0] = tmpForReduction;
#endif
#endif


#ifdef MANUAL_UNROLL
  for(; i < count; i++)
  {
    unsigned int h = _bin_count;
    unsigned  l = 0;
    float d = data[i];
    while(h - l != 1) // binary search
    {
      int middle = (h - l) / 2 + l;
      if(boundary[middle] > d) h = middle;
      else l = middle;
    }
    bin[l]++;
  }
#endif

  //	printf("last element %d\n", bin[0]);

  _mm_free(boundary);
  return 0;
}

#endif

#ifdef SIMD
#ifdef __MIC__
int hist_binary_float_simd_sg 
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_count,
 float * tree
 )
{
  size_t remainder;
  size_t start = 0;
  unsigned int i;
  int j;

  for(i = 0, j = 1; j < bin_count; j*=2, i++);

  int HEIGHT = i;

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
  i = start;
  int mid = (bin_count + 1) / 2;
  int offset = mid/2;
  __m512i all1 = _mm512_set1_epi32(1);

  for(i = start; i + VLEN - 1 < count; i+=VLEN)
  {
    __m512 d =  _MM_LOAD(data + i);
    __m512i idxs = _mm512_set1_epi32(mid);
    __m512i offsets = _mm512_set1_epi32(offset);
    for(j = 1; j < HEIGHT; j++, offsets = _mm512_srli_epi32(offsets, 1))
    {
      __m512 bound = _mm512_i32gather_ps(idxs,  tree,  _MM_SCALE_4);
      __mmask16 cr = _mm512_cmp_ps_mask(d, bound, _MM_CMPINT_LT); 
      idxs = _mm512_mask_sub_epi32(idxs, cr, idxs, offsets);
      idxs = _mm512_mask_add_epi32(idxs, _mm512_knot(cr), idxs, offsets);
    }

    __m512 bound = _mm512_i32gather_ps(idxs,  tree,  _MM_SCALE_4);
    __mmask16 cr = _mm512_cmp_ps_mask(d, bound, _MM_CMPINT_LT); 
    idxs = _mm512_mask_sub_epi32(idxs, cr, idxs, all1); 
    __declspec(align(64)) int buf[16];
    _mm512_store_epi32(buf, idxs);
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
  }


  unsigned low = 0, high = bin_count;
  for(; i < count; i++)
  {
    unsigned int h = high;
    unsigned  l = low;
    float d = data[i];
    while(h - l != 1) // binary search
    {
      int middle = (h - l) / 2 + l;
      if(boundary[middle] > d) h = middle;
      else l = middle;
    }
    bin[l]++;
  }
  return 0;
}
#endif
#endif

int hist_binary_float 
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_count,
 float * tree
 )
{
  size_t remainder;
  size_t start = 0;
  unsigned int i;
  int j;
#ifndef SIMD
  int VLEN = 8;
#endif

  for(i = 0, j = 1; j < bin_count; j*=2, i++);

  int HEIGHT = i;

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
  i = start;
  int mid = (bin_count + 1) / 2;
  for(i = start; i + VLEN - 1 < count; i+=VLEN)
  {
    float d1 = data[i];
    float d2 = data[i+1];
    float d3 = data[i+2];
    float d4 = data[i+3];
    float d5 = data[i+4];
    float d6 = data[i+5];
    float d7 = data[i+6];
    float d8 = data[i+7];

    int i1, i2, i3, i4, i5, i6, i7, i8;
    i1=i2=i3=i4=i5=i6=i7=i8=mid;

    int offset = mid/2;
    for(j = 1; j < HEIGHT; j++, offset /= 2)
    {
      //printf("%d : %d\n", j, i1);
      float c1 = boundary[i1];
      float c2 = boundary[i2];
      float c3 = boundary[i3];
      float c4 = boundary[i4];
      float c5 = boundary[i5];
      float c6 = boundary[i6];
      float c7 = boundary[i7];
      float c8 = boundary[i8];
      int x1 = d1 >= c1;
      int x2 = d2 >= c2;
      int x3 = d3 >= c3;
      int x4 = d4 >= c4;
      int x5 = d5 >= c5;
      int x6 = d6 >= c6;
      int x7 = d7 >= c7;
      int x8 = d8 >= c8;
#if 0
      //printf("%d : %d\n", j, x1);
      x1 <<= 1;
      x2 <<= 1;
      x3 <<= 1;
      x4 <<= 1;
      x5 <<= 1;
      x6 <<= 1;
      x7 <<= 1;
      x8 <<= 1;
      //printf("%d : %d\n", j, x1);
      x1 -= 1;
      x2 -= 1;
      x3 -= 1;
      x4 -= 1;
      x5 -= 1;
      x6 -= 1;
      x7 -= 1;
      x8 -= 1;
      //printf("%d : %d\n", j, x1);
      x1 *= offset; 
      x2 *= offset; 
      x3 *= offset; 
      x4 *= offset; 
      x5 *= offset; 
      x6 *= offset; 
      x7 *= offset; 
      x8 *= offset; 
      //printf("%d : %d\n", j, x1);
      i1 += x1;
      i2 += x2;
      i3 += x3;
      i4 += x4;
      i5 += x5;
      i6 += x6;
      i7 += x7;
      i8 += x8;
#else
      i1 += x1?offset:-offset;
      i2 += x2?offset:-offset;
      i3 += x3?offset:-offset;
      i4 += x4?offset:-offset;
      i5 += x5?offset:-offset;
      i6 += x6?offset:-offset;
      i7 += x7?offset:-offset;
      i8 += x8?offset:-offset;

#endif
    }

    float c1 = boundary[i1];
    float c2 = boundary[i2];
    float c3 = boundary[i3];
    float c4 = boundary[i4];
    float c5 = boundary[i5];
    float c6 = boundary[i6];
    float c7 = boundary[i7];
    float c8 = boundary[i8];
    int x1 = d1 >= c1;
    int x2 = d2 >= c2;
    int x3 = d3 >= c3;
    int x4 = d4 >= c4;
    int x5 = d5 >= c5;
    int x6 = d6 >= c6;
    int x7 = d7 >= c7;
    int x8 = d8 >= c8;
    x1 -= 1;
    x2 -= 1;
    x3 -= 1;
    x4 -= 1;
    x5 -= 1;
    x6 -= 1;
    x7 -= 1;
    x8 -= 1;
    i1 += x1;
    i2 += x2;
    i3 += x3;
    i4 += x4;
    i5 += x5;
    i6 += x6;
    i7 += x7;
    i8 += x8;
    bin[i1]++;
    bin[i2]++;
    bin[i3]++;
    bin[i4]++;
    bin[i5]++;
    bin[i6]++;
    bin[i7]++;
    bin[i8]++;
  }


  unsigned low = 0, high = bin_count;
  for(; i < count; i++)
  {
    unsigned int h = high;
    unsigned  l = low;
    float d = data[i];
    while(h - l != 1) // binary search
    {
      int middle = (h - l) / 2 + l;
      if(boundary[middle] > d) h = middle;
      else l = middle;
    }
    bin[l]++;
  }
  return 0;
}

int hist_ipp_float
(
 float * data, //data should be aligned
 float * boundary,
 unsigned int count,
 unsigned int * bin,
 unsigned int bin_count 
 )
{
  return 0;
}
