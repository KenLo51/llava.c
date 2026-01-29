#ifndef NN_OPS_H
#define NN_OPS_H

#include <math.h>
#include <stdlib.h>
#include <omp.h>

void rmsnorm_inplace(float* x, float* weight, int size);
void rmsnorm(float* out, float* in, float* weight, int size);
void softmax_inplace(float* x, int size);
void softmax(float* out, float* in, int size);
void matmul_inplace(float* x, float* weight, int in_channel, int out_channel);
void matmul(float* out, float* in, float* weight, int in_channel, int out_channel);

#endif // NN_OPS_H