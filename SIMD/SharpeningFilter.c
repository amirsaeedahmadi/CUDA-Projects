#part b
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmp.c"
#include <assert.h> 
#include <immintrin.h>


int main()
{
  /* start reading the file and its information*/
  byte *pixels_top, *pixels_bg;
  int32 width_top, width_bg;
  int32 height_top, height_bg;
  int32 bytesPerPixel_top, bytesPerPixel_bg;
  ReadImage("r.bmp", &pixels_top, &width_top, &height_top, &bytesPerPixel_top);
  ReadImage("r.bmp", &pixels_bg, &width_bg, &height_bg, &bytesPerPixel_bg);
  

  /* images should have color and be of the same size */
  assert(bytesPerPixel_top == 3);
  assert(width_top == width_bg);
  assert(height_top == height_bg); 
  assert(bytesPerPixel_top == bytesPerPixel_bg); 

  /* we can now work with one size */
  int32 width = width_top, height = height_top, bytesPerPixel = bytesPerPixel_top; 

  
  /* start replacing green screen using SIMD */
  int32 stride = 1; // Process 16 pixels at a time

  for (int i = 1; i < height; i += 1)
  {
    for (int j = 1; j < width; j += 1){
        int center = i * width + j;
        center *= 3;

        int start = center - 3*width;

        __m128i v2 = _mm_setr_epi32(pixels_top[start], pixels_top[start+1],
                                pixels_top[start+2], 0);
        

        start += 3*width;

        __m128i v4 = _mm_setr_epi32(pixels_top[start-3], pixels_top[start-2], 
                                pixels_top[start-1], 0);
        __m128i v5 = _mm_setr_epi32(pixels_top[start], pixels_top[start+1],
                                pixels_top[start+2], 0);
  
        __m128i v6 = _mm_setr_epi32(pixels_top[start+3], pixels_top[start+4],
                                pixels_top[start+5], 0);
        start += 3*width;

        __m128i v8 = _mm_setr_epi32(pixels_top[start], pixels_top[start+1],
                                pixels_top[start+2], 0);
   
        __m128i s10 = _mm_add_epi32(v5, v5);
        s10 = _mm_add_epi32(s10, s10);
        s10 = _mm_add_epi32(s10, s10);
        s10 = _mm_sub_epi32(s10, v2);
        s10 = _mm_sub_epi32(s10, v4);
        s10 = _mm_sub_epi32(s10, v6);
        s10 = _mm_sub_epi32(s10, v8);

        int* ptr = (int*)&s10;

        pixels_bg[center] = ptr[0]/4;
        pixels_bg[center+1] = ptr[1]/4;
        pixels_bg[center+2] = ptr[2]/4;

    }
  }
    /* write new image */
    WriteImage("r_sharpen.bmp", pixels_bg, width, height, bytesPerPixel);
}
