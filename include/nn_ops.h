#ifndef NN_OPS_H
#define NN_OPS_H

#include <math.h>
#include <stdlib.h>
#include <omp.h>

void rmsnorm_inplace(float* x, float* weight, int size);
void rmsnorm(float* out, float* in, float* weight, int size);
void layernorm_inplace(float* x, float* weight, float* bias, int size, float epsilon);
void layernorm(float* out, float* in, float* weight, float* bias, int size, float epsilon);

void matmul_inplace(float* x, float* weight, int in_channel, int out_channel);
void matmul(float* out, float* in, float* weight, int in_channel, int out_channel);
void linear_inplace(float* x, float* weight, float* bias, int in_channel, int out_channel);
void linear(float* out, float* in, float* weight, float* bias, int in_channel, int out_channel);

void softmax_inplace(float* x, int size);
void softmax(float* out, float* in, int size);

void gelu_inplace(float* x, int size);
void gelu(float* out, float* in, int size);
void quick_gelu_inplace(float* x, int size);
void quick_gelu(float* out, float* in, int size);
void silu_inplace(float* x, int size);
void silu(float* out, float* in, int size);

#endif // NN_OPS_H