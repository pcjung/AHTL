#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hist_method.h"
#include "timer.h"
#include <assert.h>
#ifdef SIMD
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#endif

static void check_result(unsigned *a, unsigned *b, unsigned count, const char * funct)
{
				unsigned i;
				for(i = 0; i < count; i++)
				{
								if(a[i] != b[i]) break;
				}
				if(i != count) // error
				{
								printf("%s wrong result\n", funct);
								printf("bin[%d] : %d -> %d\n", i, a[i], b[i]);
								exit(-1);
				}
				fprintf(stderr, "%s passed\n", funct);
				fflush(stderr);
				return;
}
int main()
{
				unsigned int count = 65536;
				unsigned int bin_count = 512;
				float * data = _mm_malloc(count * sizeof(float), 4096);
				float * boundary = _mm_malloc((bin_count + 1) * sizeof(float), 4096);
				float base = 0.0;
				float width = 10.0;
				float max = base + width * bin_count;
				timer t_;
				timer * t = &t_;
				timer_init(t);

				unsigned int i;
				float temp;

				fprintf(stderr, "init bin boundary values..\n");
				for(i = 0; i < bin_count + 1; i++) boundary[i] = base + width * i;
				
				fprintf(stderr, "init input data....\n");
				for(temp = 0.5, i = 0; i < count; i++) 
				//psuedo random input geneartion for test
				{
								temp += 17;
								if( temp > max ) temp -= max;
							  data[i] = temp;
				}
				
				fprintf(stderr, "check if the input is valid..\n");
				for(i = 0; i < count; i++)
				{
								assert(data[i] > boundary[0]);
								assert(data[i] < boundary[bin_count]);
				}
				fflush(stderr);

				unsigned int * bin = _mm_malloc(bin_count * sizeof(unsigned int), 4096);
				unsigned int * bin_compare = _mm_malloc(bin_count * sizeof(unsigned int), 4096);
				memset(bin, 0, sizeof(int) * bin_count); 
				memset(bin_compare, 0, sizeof(int) * bin_count); 
				

				hist_uniform_float(data, base, width, count, bin_compare, bin_count);

				hist_uniform_float_atomic(data, base, width, count, bin, bin_count);
				check_result(bin_compare, bin, bin_count, "hist_uniform_float_atomic");
				memset(bin, 0, sizeof(int) * bin_count); 

				hist_linear_float(data, boundary, count, bin, bin_count);
				check_result(bin_compare, bin, bin_count, "hist_linear_float");
				memset(bin, 0, sizeof(int) * bin_count); 

				hist_binary_float(data, boundary, count, bin, bin_count);
				check_result(bin_compare, bin, bin_count, "hist_binary_float");
				memset(bin, 0, sizeof(int) * bin_count); 
#ifdef SIMD
				
				hist_uniform_float_simd(data, base, width, count, bin, bin_count);
				check_result(bin_compare, bin, bin_count, "hist_uniform_float_simd");
				memset(bin, 0, sizeof(int) * bin_count); 

				hist_linear_float_simd(data, boundary, count, bin, bin_count);
				check_result(bin_compare, bin, bin_count, "hist_linear_float_simd");
				memset(bin, 0, sizeof(int) * bin_count); 

				//hist_binary_float_simd(data, boundary, count, bin, bin_count);
				check_result(bin_compare, bin, bin_count, "hist_binary_float_simd");
				memset(bin, 0, sizeof(int) * bin_count); 

				hist_sorting_float_simd(data, boundary, count, bin, bin_count);
				check_result(bin_compare, bin, bin_count, "hist_sorting_float_simd");
				memset(bin, 0, sizeof(int) * bin_count); 

				unsigned int opcode;
				unsigned int thread_num = 1;
				for(thread_num = 1; thread_num <= 32; thread_num*=2)
				{
								for(opcode = 1; opcode < 6; opcode++)
								{
												hist_float_omp(
																				data, 
																				count, //assume 2^n
																				boundary, 
																				bin_count, //assume 2^n for sorting, 8^n binary
																				base, 
																				width, 
																				bin,
																				thread_num, 
																				opcode
#ifdef GET_TIME
																				,t  
#endif
																			);
												char buffer[128];
												sprintf(buffer, "omp opcode=%d, %d threads", opcode, thread_num);
												check_result(bin_compare, bin, bin_count, buffer);
												memset(bin, 0, sizeof(int) * bin_count); 
								}
				}


#endif
				return 0;
}
