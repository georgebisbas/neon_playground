%% DEVITO_JIT_BACKDOOR=1 DEVITO_PLATFORM=cortexa76 DEVITO_ARCH=arm DEVITO_COMPILER=gcc DEVITO_LOGGING=DEBUG DEVITO_LANGUAGE=openmp python examples/diff_2d.py 

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S .tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;
#define uL0(t,x,y) u[(t)*x_stride0 + (x)*y_stride0 + (y)]

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "omp.h"
#include "arm_neon.h"

struct dataobj
{
  void *restrict data;
  unsigned long * size;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
};

struct profiler
{
  double section0;
};

int Kernel(struct dataobj *restrict u_vec, const float h_x, const float h_y, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, const int nthreads, struct profiler * timers)
{
  float *u __attribute__ ((aligned (64))) = (float *) u_vec->data;

  const int x_fsz0 = u_vec->size[1];
  const int y_fsz0 = u_vec->size[2];

  const int x_stride0 = x_fsz0 * y_fsz0;
  const int y_stride0 = y_fsz0;

  float32x4_t r0_vec = vdupq_n_f32(1.0F / (h_x * h_x));
  float32x4_t r1_vec = vdupq_n_f32(1.0F / (h_y * h_y));
  float32x4_t factor1 = vdupq_n_f32(-2.0e-2F);
  float32x4_t factor2 = vdupq_n_f32(1.0e-2F);

  for (int time = time_m, t0 = (time) % 2, t1 = (time + 1) % 2; time <= time_M; time += 1, t0 = (time) % 2, t1 = (time + 1) % 2)
  {
    START(section0)
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for schedule(dynamic,1)
      for (int x = x_m; x <= x_M; x += 1)
      {
        int y;
        for (y = y_m; y <= y_M - 3; y += 4)
        {
          float32x4_t u_c = vld1q_f32(&uL0(t0, x + 2, y + 2));
          float32x4_t u_xm1 = vld1q_f32(&uL0(t0, x + 1, y + 2));
          float32x4_t u_xp1 = vld1q_f32(&uL0(t0, x + 3, y + 2));
          float32x4_t u_ym1 = vld1q_f32(&uL0(t0, x + 2, y + 1));
          float32x4_t u_yp1 = vld1q_f32(&uL0(t0, x + 2, y + 3));
          
          float32x4_t term1 = vmulq_f32(factor1, vaddq_f32(vmulq_f32(r0_vec, u_c), vmulq_f32(r1_vec, u_c)));
          float32x4_t term2 = vmulq_f32(factor2, vaddq_f32(vaddq_f32(vmulq_f32(r0_vec, u_xm1), vmulq_f32(r0_vec, u_xp1)), vaddq_f32(vmulq_f32(r1_vec, u_ym1), vmulq_f32(r1_vec, u_yp1))));
          float32x4_t result = vaddq_f32(vaddq_f32(term1, term2), u_c);
          
          vst1q_f32(&uL0(t1, x + 2, y + 2), result);
        }
        
        for (; y <= y_M; y++)
        {
          uL0(t1, x + 2, y + 2) = -2.0e-2F * (r0_vec[0] * uL0(t0, x + 2, y + 2) + r1_vec[0] * uL0(t0, x + 2, y + 2)) + 
                                  1.0e-2F * (r0_vec[0] * uL0(t0, x + 1, y + 2) + r0_vec[0] * uL0(t0, x + 3, y + 2) + 
                                             r1_vec[0] * uL0(t0, x + 2, y + 1) + r1_vec[0] * uL0(t0, x + 2, y + 3)) + 
                                  uL0(t0, x + 2, y + 2);
        }
      }
    }
    STOP(section0, timers)
  }

  return 0;
}
/* Backdoor edit at Thu Feb  6 19:50:52 2025*/ 
/* Backdoor edit at Thu Feb  6 19:50:57 2025*/ 
/* Backdoor edit at Thu Feb  6 19:51:07 2025*/ 
/* Backdoor edit at Thu Feb  6 20:00:12 2025*/ 
/* Backdoor edit at Thu Feb  6 20:00:17 2025*/ 
/* Backdoor edit at Thu Feb  6 20:00:39 2025*/ 
