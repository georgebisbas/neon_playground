// Using Github copilot, correct results, 3d diffusion


#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;
#define uL0(t,x,y,z) u[(t)*x_stride0 + (x)*y_stride0 + (y)*z_stride0 + (z)]
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "omp.h"
#include <arm_neon.h>

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
} ;

struct profiler
{
  double section0;
} ;


int Kernel(struct dataobj *restrict u_vec, const float h_x, const float h_y, const float h_z, const int time_M, const int time_m, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, struct profiler * timers)
{
  float *u __attribute__ ((aligned (64))) = (float *) u_vec->data;

  const int x_fsz0 = u_vec->size[1];
  const int y_fsz0 = u_vec->size[2];
  const int z_fsz0 = u_vec->size[3];

  const int x_stride0 = x_fsz0*y_fsz0*z_fsz0;
  const int y_stride0 = y_fsz0*z_fsz0;
  const int z_stride0 = z_fsz0;

  float r0 = 1.0F/(h_x*h_x);
  float r1 = 1.0F/(h_y*h_y);
  float r2 = 1.0F/(h_z*h_z);

  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    START(section0)
    #pragma omp parallel num_threads(nthreads)
    {
      #pragma omp for collapse(2) schedule(dynamic,1)
      for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
        {
          for (int x = x0_blk0; x <= MIN(x_M, x0_blk0 + x0_blk0_size - 1); x += 1)
          {
            for (int y = y0_blk0; y <= MIN(y_M, y0_blk0 + y0_blk0_size - 1); y += 1)
            {
              int z;
              for (z = z_m; z <= z_M - 3; z += 4)
              {
                float32x4_t uL0_t0_x2_y2_z2 = vld1q_f32(&uL0(t0, x + 2, y + 2, z + 2));
                float32x4_t uL0_t0_x1_y2_z2 = vld1q_f32(&uL0(t0, x + 1, y + 2, z + 2));
                float32x4_t uL0_t0_x3_y2_z2 = vld1q_f32(&uL0(t0, x + 3, y + 2, z + 2));
                float32x4_t uL0_t0_x2_y1_z2 = vld1q_f32(&uL0(t0, x + 2, y + 1, z + 2));
                float32x4_t uL0_t0_x2_y3_z2 = vld1q_f32(&uL0(t0, x + 2, y + 3, z + 2));
                float32x4_t uL0_t0_x2_y2_z1 = vld1q_f32(&uL0(t0, x + 2, y + 2, z + 1));
                float32x4_t uL0_t0_x2_y2_z3 = vld1q_f32(&uL0(t0, x + 2, y + 2, z + 3));

                float32x4_t result = vmulq_n_f32(uL0_t0_x2_y2_z2, -2.0e-2F * (r0 + r1 + r2));
                result = vmlaq_n_f32(result, uL0_t0_x1_y2_z2, 1.0e-2F * r0);
                result = vmlaq_n_f32(result, uL0_t0_x3_y2_z2, 1.0e-2F * r0);
                result = vmlaq_n_f32(result, uL0_t0_x2_y1_z2, 1.0e-2F * r1);
                result = vmlaq_n_f32(result, uL0_t0_x2_y3_z2, 1.0e-2F * r1);
                result = vmlaq_n_f32(result, uL0_t0_x2_y2_z1, 1.0e-2F * r2);
                result = vmlaq_n_f32(result, uL0_t0_x2_y2_z3, 1.0e-2F * r2);
                result = vaddq_f32(result, uL0_t0_x2_y2_z2);

                vst1q_f32(&uL0(t1, x + 2, y + 2, z + 2), result);
              }

              // Handle remaining elements
              for (; z <= z_M; z += 1)
              {
                uL0(t1, x + 2, y + 2, z + 2) = -2.0e-2F*(r0*uL0(t0, x + 2, y + 2, z + 2) + r1*uL0(t0, x + 2, y + 2, z + 2) + r2*uL0(t0, x + 2, y + 2, z + 2)) 
                + 1.0e-2F*(r0*uL0(t0, x + 1, y + 2, z + 2) + r0*uL0(t0, x + 3, y + 2, z + 2) + r1*uL0(t0, x + 2, y + 1, z + 2) + r1*uL0(t0, x + 2, y + 3, z + 2) + r2*uL0(t0, x + 2, y + 2, z + 1) + r2*uL0(t0, x + 2, y + 2, z + 3)) 
                + uL0(t0, x + 2, y + 2, z + 2);
              }
            }
          }
        }
      }
    }
    STOP(section0,timers)
  }

  return 0;
}
