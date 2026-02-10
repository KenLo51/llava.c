#ifndef GGUF_READER_H
#define GGUF_READER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

// GGUF Constants
#define GGUF_MAGIC 0x46554747  // "GGUF" in little endian
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32

// GGUF Value Types
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

// GGML Quantization Types
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8   = 16,
    GGML_TYPE_I16  = 17,
    GGML_TYPE_I32  = 18,
    GGML_TYPE_I64  = 19,
    GGML_TYPE_F64  = 20,
} ggml_type;

// String structure for GGUF
typedef struct {
    uint64_t len;
    char* data;
} gguf_string;

// Forward declaration for circular dependency
typedef struct gguf_array gguf_array;

// Value union for different types
typedef union {
    uint8_t   uint8;
    int8_t    int8;
    uint16_t  uint16;
    int16_t   int16;
    uint32_t  uint32;
    int32_t   int32;
    float     float32;
    bool      bool_;
    uint64_t  uint64;
    int64_t   int64;
    double    float64;
    gguf_string string;
    gguf_array*     arr;  // pointer to array data
} gguf_value;

// Array structure
struct gguf_array {
    gguf_type type;
    uint64_t len;
    gguf_value* data;  // array of values
};

// Metadata key-value pair
typedef struct {
    char* key;
    gguf_type type;
    gguf_value value;
} gguf_metadata_kv;

// Tensor information
typedef struct {
    char* name;
    uint32_t n_dims;
    uint64_t* dims;
    ggml_type type;
    uint64_t offset;
    size_t n_elements;
    size_t n_bytes;
    void* data;  // pointer to tensor data
} gguf_tensor;

// Main GGUF context
typedef struct {
    // File data
    uint8_t* data;
    size_t size;
    FILE* file;  // Keep file handle open for on-demand tensor loading
    
    // Header
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    
    // Metadata
    gguf_metadata_kv* metadata;
    
    // Tensors
    gguf_tensor* tensors;
    
    // Alignment and offset
    uint32_t alignment;
    size_t data_offset;
    
    // Byte order ('I' for native, 'S' for swapped)
    char byte_order;
    
    // Flags
    bool tensors_loaded;
} gguf_context;

// Helper function to get GGML type name
size_t ggml_type_size(ggml_type type);
float ggml_type_to_float(ggml_type type, uint8_t* data);

// Function declarations
gguf_context* gguf_init_from_file(const char* filename);
gguf_context* gguf_init_from_file_ex(const char* filename, bool load_tensors);
void gguf_free(gguf_context* ctx);

// Helper functions
gguf_metadata_kv* gguf_get_metadata(gguf_context* ctx, const char* key);
gguf_tensor* gguf_get_tensor(gguf_context* ctx, uint64_t index);
gguf_tensor* gguf_find_tensor(gguf_context* ctx, const char* name);

// Tensor data loading
bool gguf_load_tensor_data(gguf_context* ctx, gguf_tensor* tensor);
void gguf_free_tensor_data(gguf_tensor* tensor);

// Print functions for debugging
void gguf_print_metadata(gguf_context* ctx);
void gguf_print_tensors(gguf_context* ctx);

// Data conversion
void copy_tensor_data_to_float_array(gguf_tensor* tensor, float* dest_array);

#endif // GGUF_READER_H