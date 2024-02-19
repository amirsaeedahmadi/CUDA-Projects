#part a
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmp.c"
#include <assert.h> 
#include <immintrin.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;


__m128i compare(__m128i a, __m128i b) {
    __m128i z = _mm_set1_epi8(0);
    __m128i ff = _mm_set1_epi8(0xff);

    
    __m128i a_c = _mm_cmpgt_epi8(a, z);
    __m128i b_c = _mm_cmpgt_epi8(b, z);

    __m128i and_a_b_c = _mm_and_si128(a_c, b_c);
    __m128i nor_a_b_c = _mm_or_si128(a_c, b_c);
    nor_a_b_c = _mm_xor_si128(nor_a_b_c, ff);

    __m128i na_has_alamat = _mm_xor_si128(a_c, b_c);

    __m128i not_xor_a_b_c = _mm_xor_si128(_mm_xor_si128(a_c, b_c), ff); 

    
    __m128i compare_result = _mm_cmpgt_epi8(a, b);


    __m128i not_compare_result = _mm_xor_si128(compare_result, ff);

    compare_result = _mm_and_si128(compare_result, not_xor_a_b_c);
    compare_result = _mm_or_si128(compare_result, _mm_and_si128(_mm_cmpgt_epi8(z, a) , na_has_alamat));
    compare_result = _mm_or_si128(compare_result, _mm_cmpeq_epi8(z, b));


    return compare_result;
}

int main()
{

  byte *pixels_top_1, *pixels_bg_1;
  int32 width_top, width_bg;
  int32 height_top, height_bg;
  int32 bytesPerPixel_top, bytesPerPixel_bg;
  ReadImage("dino.bmp", &pixels_top_1, &width_top, &height_top, &bytesPerPixel_top);
  ReadImage("parking.bmp", &pixels_bg_1, &width_bg, &height_bg, &bytesPerPixel_bg);
  

  /* images should have color and be of the same size */
  assert(bytesPerPixel_top == 3);
  assert(width_top == width_bg);
  assert(height_top == height_bg); 
  assert(bytesPerPixel_top == bytesPerPixel_bg); 

  /* we can now work with one size */
  int32 width = width_top, height = height_top, bytesPerPixel = bytesPerPixel_top; 

  
  // Allocate memory for RGBA image data
  byte * pixels_top = (byte*)malloc(width * height * 4);

  for (int i = 0; i < width * height; i++) {
      pixels_top[i * 4] = pixels_top_1[i * 3];
      pixels_top[i * 4 + 1] = pixels_top_1[i * 3 + 1]; 
      pixels_top[i * 4 + 2] = pixels_top_1[i * 3 + 2]; 
      pixels_top[i * 4 + 3] = 0;                
  }

  byte * pixels_bg = (byte*)malloc(width * height * 4);

  for (int i = 0; i < width * height; i++) {
      pixels_bg[i * 4] = pixels_bg_1[i * 3];        
      pixels_bg[i * 4 + 1] = pixels_bg_1[i * 3 + 1]; 
      pixels_bg[i * 4 + 2] = pixels_bg_1[i * 3 + 2]; 
      pixels_bg[i * 4 + 3] = 0;                
  }

  
  /* start replacing green screen using SIMD */
  int32 pixelCount = width * height * 4;
  int32 vectorSize = 16; // Process 4 pixels at a time

  for (int i = 0; i < pixelCount; i += vectorSize)
  {
    __m128i pixels_top_vec = _mm_loadu_si128((__m128i*)&pixels_top[i]);
    __m128i pixels_bg_vec = _mm_loadu_si128((__m128i*)&pixels_bg[i]);


    __m128i shuffle_mask = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 4, 8, 12);
    __m128i r_shuffled_vec = _mm_shuffle_epi8(pixels_top_vec, shuffle_mask);

    shuffle_mask = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 5, 9, 13);
    __m128i g_shuffled_vec = _mm_shuffle_epi8(pixels_top_vec, shuffle_mask);


    shuffle_mask = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2, 6, 10, 14);
    __m128i b_shuffled_vec = _mm_shuffle_epi8(pixels_top_vec, shuffle_mask);


    __m128i m1 = compare(g_shuffled_vec, r_shuffled_vec);
    __m128i m2 = compare(g_shuffled_vec, b_shuffled_vec);
    __m128i green_masked = _mm_and_si128(m1,m2);

    shuffle_mask = _mm_set_epi8(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3);
    green_masked = _mm_shuffle_epi8(green_masked, shuffle_mask);


    __m128i new_pixels_vec = _mm_or_si128(
        _mm_andnot_si128(green_masked, pixels_top_vec), 
        _mm_and_si128(green_masked, pixels_bg_vec));

    _mm_storeu_si128((__m128i*)&pixels_top[i], new_pixels_vec);
  }

  for (int i = 0; i < width * height; i++) {
      pixels_top_1[i * 3] = pixels_top[i * 4];
      pixels_top_1[i * 3 + 1] = pixels_top[i * 4 + 1]; 
      pixels_top_1[i * 3 + 2] = pixels_top[i * 4 + 2]; 
  }


  /* write new image */
  WriteImage("r.bmp", pixels_top_1, width, height, bytesPerPixel);
}
