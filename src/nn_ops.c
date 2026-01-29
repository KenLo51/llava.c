#include "nn_ops.h"

void rmsnorm_inplace(float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        x[j] = weight[j] * (ss * x[j]);
    }
}

void rmsnorm(float* out, float* in, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += in[j] * in[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        out[j] = weight[j] * (ss * in[j]);
    }
}

void softmax_inplace(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void softmax(float* out, float* in, int size) {
    // find max value (for numerical stability)
    float max_val = in[0];
    for (int i = 1; i < size; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        out[i] /= sum;
    }
}

void matmul_inplace(float* x, float* weight, int in_channel, int out_channel) {
    // W (d,n) @ x (n,) -> x (d,)
    // Need temporary storage since we're overwriting x
    float* temp = (float*)malloc(out_channel * sizeof(float));
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < out_channel; i++) {
        float val = 0.0f;
        for (int j = 0; j < in_channel; j++) {
            val += weight[i * in_channel + j] * x[j];
        }
        temp[i] = val;
    }
    // Copy result back to x
    for (i = 0; i < out_channel; i++) {
        x[i] = temp[i];
    }
    free(temp);
}

void matmul(float* out, float* in, float* weight, int in_channel, int out_channel) {
    // W (d,n) @ in (n,) -> out (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < out_channel; i++) {
        float val = 0.0f;
        for (int j = 0; j < in_channel; j++) {
            val += weight[i * in_channel + j] * in[j];
        }
        out[i] = val;
    }
}
