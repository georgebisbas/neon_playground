#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "omp.h"

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
  double section1;
  double section2;
} ;


int Forward(struct dataobj *restrict damp_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const float dt, const float o_x, const float o_y, const float o_z, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x0_blk0_size, const int y0_blk0_size, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  float r1 = 1.0F/(dt*dt);
  float r2 = 1.0F/dt;

  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
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
              #pragma omp simd aligned(damp,u,vp:16)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r3 = 1.0F/(vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4]);
                u[t2][x + 4][y + 4][z + 4] = (-r3*(-2.0F*r1*u[t0][x + 4][y + 4][z + 4] + r1*u[t1][x + 4][y + 4][z + 4]) + r2*damp[x + 4][y + 4][z + 4]*u[t0][x + 4][y + 4][z + 4] + 3.70370379e-4F*(-u[t0][x + 2][y + 4][z + 4] - u[t0][x + 4][y + 2][z + 4] - u[t0][x + 4][y + 4][z + 2] - u[t0][x + 4][y + 4][z + 6] - u[t0][x + 4][y + 6][z + 4] - u[t0][x + 6][y + 4][z + 4]) + 5.92592607e-3F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5] + u[t0][x + 4][y + 5][z + 4] + u[t0][x + 5][y + 4][z + 4]) - 3.33333341e-2F*u[t0][x + 4][y + 4][z + 4])/(r3*r1 + r2*damp[x + 4][y + 4][z + 4]);
              }
            }
          }
        }
      }
    }
    STOP(section0,timers)

    START(section1)
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_src_M - p_src_m + 1)/nthreads_nonaffine));
      #pragma omp for schedule(dynamic,chunk_size)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        for (int rsrcx = 0; rsrcx <= 1; rsrcx += 1)
        {
          for (int rsrcy = 0; rsrcy <= 1; rsrcy += 1)
          {
            for (int rsrcz = 0; rsrcz <= 1; rsrcz += 1)
            {
              int posx = (int)(floorf(6.66667e-2*(-o_x + src_coords[p_src][0])));
              int posy = (int)(floorf(6.66667e-2*(-o_y + src_coords[p_src][1])));
              int posz = (int)(floorf(6.66667e-2*(-o_z + src_coords[p_src][2])));
              float px = 6.66667e-2F*(-o_x + src_coords[p_src][0]) - floorf(6.66667e-2F*(-o_x + src_coords[p_src][0]));
              float py = 6.66667e-2F*(-o_y + src_coords[p_src][1]) - floorf(6.66667e-2F*(-o_y + src_coords[p_src][1]));
              float pz = 6.66667e-2F*(-o_z + src_coords[p_src][2]) - floorf(6.66667e-2F*(-o_z + src_coords[p_src][2]));
              if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcz + posz >= z_m - 1 && rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1 && rsrcz + posz <= z_M + 1)
              {
                float r0 = (dt*dt)*(vp[posx + 4][posy + 4][posz + 4]*vp[posx + 4][posy + 4][posz + 4])*(rsrcx*px + (1 - rsrcx)*(1 - px))*(rsrcy*py + (1 - rsrcy)*(1 - py))*(rsrcz*pz + (1 - rsrcz)*(1 - pz))*src[time][p_src];
                #pragma omp atomic update
                u[t2][rsrcx + posx + 4][rsrcy + posy + 4][rsrcz + posz + 4] += r0;
              }
            }
          }
        }
      }
    }
    STOP(section1,timers)

    START(section2)
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_rec_M - p_rec_m + 1)/nthreads_nonaffine));
      #pragma omp for schedule(dynamic,chunk_size)
      for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
      {
        float r7 = 6.66667e-2F*(-o_x + rec_coords[p_rec][0]);
        float r4 = floorf(r7);
        int posx = (int)r4;
        float r8 = 6.66667e-2F*(-o_y + rec_coords[p_rec][1]);
        float r5 = floorf(r8);
        int posy = (int)r5;
        float r9 = 6.66667e-2F*(-o_z + rec_coords[p_rec][2]);
        float r6 = floorf(r9);
        int posz = (int)r6;
        float px = -r4 + r7;
        float py = -r5 + r8;
        float pz = -r6 + r9;
        float sum = 0.0F;

        for (int rrecx = 0; rrecx <= 1; rrecx += 1)
        {
          for (int rrecy = 0; rrecy <= 1; rrecy += 1)
          {
            for (int rrecz = 0; rrecz <= 1; rrecz += 1)
            {
              if (rrecx + posx >= x_m - 1 && rrecy + posy >= y_m - 1 && rrecz + posz >= z_m - 1 && rrecx + posx <= x_M + 1 && rrecy + posy <= y_M + 1 && rrecz + posz <= z_M + 1)
              {
                sum += (rrecx*px + (1 - rrecx)*(1 - px))*(rrecy*py + (1 - rrecy)*(1 - py))*(rrecz*pz + (1 - rrecz)*(1 - pz))*u[t0][rrecx + posx + 4][rrecy + posy + 4][rrecz + posz + 4];
              }
            }
          }
        }

        rec[time][p_rec] = sum;
      }
    }
    STOP(section2,timers)
  }

  return 0;
}
