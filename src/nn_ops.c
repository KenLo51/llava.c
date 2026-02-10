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

void linear_inplace(float* x, float* weight, float* bias, int in_channel, int out_channel) {
    // W (d,n) @ x (n,) + b (d,) -> x (d,)
    // Need temporary storage since we're overwriting x
    float* temp = (float*)malloc(out_channel * sizeof(float));
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < out_channel; i++) {
        float val = bias[i];
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

void linear(float* out, float* in, float* weight, float* bias, int in_channel, int out_channel) {
    // W (d,n) @ in (n,) + b (d,) -> out (d,)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < out_channel; i++) {
        float val = bias[i];
        for (int j = 0; j < in_channel; j++) {
            val += weight[i * in_channel + j] * in[j];
        }
        out[i] = val;
    }
}


void layernorm_inplace(float* x, float* weight, float* bias, int size, float epsilon) {
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += x[i];
    }
    mean /= size;
    
    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    // Normalize, scale and shift
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        x[i] = weight[i] * ((x[i] - mean) * inv_std) + bias[i];
    }
}

void layernorm(float* out, float* in, float* weight, float* bias, int size, float epsilon) {
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += in[i];
    }
    mean /= size;
    
    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = in[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    // Normalize, scale and shift
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        out[i] = weight[i] * ((in[i] - mean) * inv_std) + bias[i];
    }
}

void gelu_inplace(float* x, int size) {
    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        float x_val = x[i];
        float x_cubed = x_val * x_val * x_val;
        float inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        x[i] = 0.5f * x_val * (1.0f + tanhf(inner));
    }
}

void gelu(float* out, float* in, int size) {
    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        float x_val = in[i];
        float x_cubed = x_val * x_val * x_val;
        float inner = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        out[i] = 0.5f * x_val * (1.0f + tanhf(inner));
    }
}

void quick_gelu_inplace(float* x, int size) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        float x_val = x[i];
        // Sigmoid Approximation: x * sigmoid(1.702 * x)
        // sigmoid(z) = 1 / (1 + exp(-z))
        float val = x_val * 1.702f;
        x[i] = x_val / (1.0f + expf(-val));
    }
}
void quick_gelu(float* out, float* in, int size) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        float x_val = in[i];
        // Sigmoid Approximation: x * sigmoid(1.702 * x)
        // sigmoid(z) = 1 / (1 + exp(-z))
        float val = x_val * 1.702f;
        out[i] = x_val / (1.0f + expf(-val));
    }
}

void silu_inplace(float* x, int size) {
    // SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        float x_val = x[i];
        x[i] = x_val / (1.0f + expf(-x_val));
    }
}

void silu(float* out, float* in, int size) {
    // SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        float x_val = in[i];
        out[i] = x_val / (1.0f + expf(-x_val));
    }
}