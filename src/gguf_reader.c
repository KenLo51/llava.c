#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
    #include <windows.h>
    #include "win.h"
    #define FSEEK_64 _fseeki64
#else
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
    #define FSEEK_64 fseeko
#endif



// Quantization block sizes and type sizes
static const size_t GGML_QUANT_SIZES[][2] = {
    [GGML_TYPE_F32]  = {1, 4},
    [GGML_TYPE_F16]  = {1, 2},
    [GGML_TYPE_Q4_0] = {32, 18},
    [GGML_TYPE_Q4_1] = {32, 20},
    [GGML_TYPE_Q5_0] = {32, 22},
    [GGML_TYPE_Q5_1] = {32, 24},
    [GGML_TYPE_Q8_0] = {32, 34},
    [GGML_TYPE_Q8_1] = {32, 36},
    [GGML_TYPE_Q2_K] = {256, 82},
    [GGML_TYPE_Q3_K] = {256, 110},
    [GGML_TYPE_Q4_K] = {256, 144},
    [GGML_TYPE_Q5_K] = {256, 176},
    [GGML_TYPE_Q6_K] = {256, 210},
    [GGML_TYPE_Q8_K] = {256, 292},
    [GGML_TYPE_I8]   = {1, 1},
    [GGML_TYPE_I16]  = {1, 2},
    [GGML_TYPE_I32]  = {1, 4},
    [GGML_TYPE_I64]  = {1, 8},
    [GGML_TYPE_F64]  = {1, 8},
};

// Helper function to swap bytes based on byte order
static void maybe_swap_16(uint16_t* val, char byte_order) {
    if (byte_order == 'S') {
        *val = (*val >> 8) | (*val << 8);
    }
}

static void maybe_swap_32(uint32_t* val, char byte_order) {
    if (byte_order == 'S') {
        *val = ((*val >> 24) & 0xff) | 
               ((*val >> 8) & 0xff00) | 
               ((*val << 8) & 0xff0000) | 
               ((*val << 24) & 0xff000000);
    }
}

static void maybe_swap_64(uint64_t* val, char byte_order) {
    if (byte_order == 'S') {
        *val = ((*val >> 56) & 0xffULL) | 
               ((*val >> 40) & 0xff00ULL) | 
               ((*val >> 24) & 0xff0000ULL) | 
               ((*val >> 8) & 0xff000000ULL) |
               ((*val << 8) & 0xff00000000ULL) |
               ((*val << 24) & 0xff0000000000ULL) |
               ((*val << 40) & 0xff000000000000ULL) |
               ((*val << 56) & 0xff00000000000000ULL);
    }
}

// Read string from GGUF file
static size_t read_string(uint8_t* data, size_t offset, gguf_string* str, char byte_order) {
    size_t orig_offset = offset;
    
    // Read string length
    uint64_t len = *(uint64_t*)(data + offset);
    maybe_swap_64(&len, byte_order);
    offset += 8;
    
    // Read string data
    str->len = len;
    str->data = (char*)malloc(len + 1);
    memcpy(str->data, data + offset, len);
    str->data[len] = '\0';
    offset += len;
    
    return offset - orig_offset;
}

// Free string
static void free_string(gguf_string* str) {
    if (str->data) {
        free(str->data);
        str->data = NULL;
    }
}

// Read array from GGUF file
static size_t read_array(uint8_t* data, size_t offset, gguf_array* arr, char byte_order);

// Read value based on type
static size_t read_value(uint8_t* data, size_t offset, gguf_type type, gguf_value* val, char byte_order) {
    size_t size = 0;
    
    switch (type) {
        case GGUF_TYPE_UINT8:
            val->uint8 = *(uint8_t*)(data + offset);
            size = 1;
            break;
        case GGUF_TYPE_INT8:
            val->int8 = *(int8_t*)(data + offset);
            size = 1;
            break;
        case GGUF_TYPE_UINT16:
            val->uint16 = *(uint16_t*)(data + offset);
            maybe_swap_16(&val->uint16, byte_order);
            size = 2;
            break;
        case GGUF_TYPE_INT16:
            val->int16 = *(int16_t*)(data + offset);
            maybe_swap_16((uint16_t*)&val->int16, byte_order);
            size = 2;
            break;
        case GGUF_TYPE_UINT32:
            val->uint32 = *(uint32_t*)(data + offset);
            maybe_swap_32(&val->uint32, byte_order);
            size = 4;
            break;
        case GGUF_TYPE_INT32:
            val->int32 = *(int32_t*)(data + offset);
            maybe_swap_32((uint32_t*)&val->int32, byte_order);
            size = 4;
            break;
        case GGUF_TYPE_FLOAT32:
            val->float32 = *(float*)(data + offset);
            maybe_swap_32((uint32_t*)&val->float32, byte_order);
            size = 4;
            break;
        case GGUF_TYPE_BOOL:
            val->bool_ = *(bool*)(data + offset);
            size = 1;
            break;
        case GGUF_TYPE_UINT64:
            val->uint64 = *(uint64_t*)(data + offset);
            maybe_swap_64(&val->uint64, byte_order);
            size = 8;
            break;
        case GGUF_TYPE_INT64:
            val->int64 = *(int64_t*)(data + offset);
            maybe_swap_64((uint64_t*)&val->int64, byte_order);
            size = 8;
            break;
        case GGUF_TYPE_FLOAT64:
            val->float64 = *(double*)(data + offset);
            maybe_swap_64((uint64_t*)&val->float64, byte_order);
            size = 8;
            break;
        case GGUF_TYPE_STRING:
            size = read_string(data, offset, &val->string, byte_order);
            break;
        case GGUF_TYPE_ARRAY: {
            gguf_array* arr = (gguf_array*)malloc(sizeof(gguf_array));
            size = read_array(data, offset, arr, byte_order);
            val->arr = arr;
            break;
        }
        default:
            fprintf(stderr, "Unknown type: %d\n", type);
            break;
    }
    
    return size;
}

// Read array from GGUF file
static size_t read_array(uint8_t* data, size_t offset, gguf_array* arr, char byte_order) {
    size_t orig_offset = offset;
    
    // Read array type
    uint32_t type = *(uint32_t*)(data + offset);
    maybe_swap_32(&type, byte_order);
    arr->type = (gguf_type)type;
    offset += 4;
    
    // Read array length
    uint64_t len = *(uint64_t*)(data + offset);
    maybe_swap_64(&len, byte_order);
    arr->len = len;
    offset += 8;
    
    // Allocate array
    gguf_value* values = (gguf_value*)calloc(len, sizeof(gguf_value));
    
    // Read array elements
    for (uint64_t i = 0; i < len; i++) {
        size_t value_size = read_value(data, offset, arr->type, &values[i], byte_order);
        offset += value_size;
    }
    
    arr->data = values;
    return offset - orig_offset;
}

// Free value based on type
static void free_value(gguf_type type, gguf_value* val) {
    if (type == GGUF_TYPE_STRING) {
        free_string(&val->string);
    } else if (type == GGUF_TYPE_ARRAY) {
        gguf_array* arr = (gguf_array*)val->arr;
        if (arr) {
            if (arr->data) {
                gguf_value* values = (gguf_value*)arr->data;
                for (uint64_t i = 0; i < arr->len; i++) {
                    free_value(arr->type, &values[i]);
                }
                free(arr->data);
            }
            free(arr);
        }
    }
}

// Read data from file at specific offset
static size_t read_data_at_offset(FILE* file, size_t offset, void* buffer, size_t size) {
    if (FSEEK_64(file, (long long)offset, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to offset %zu\n", offset);
        return 0;
    }
    size_t read = fread(buffer, 1, size, file);
    if (read != size) {
        fprintf(stderr, "Failed to read %zu bytes at offset %zu (got %zu)\n", size, offset, read);
        return read;
    }
    return read;
}

// Get file size
static size_t get_file_size(FILE* file) {
    long current = ftell(file);
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, current, SEEK_SET);
    return size;
}


// Helper function to get GGML type name
size_t ggml_type_size(ggml_type type){
    if(type < 0 || type > GGML_TYPE_F64){
        fprintf(stderr, "Unknown GGML type: %d\n", type);
        return 0;
    }
    return GGML_QUANT_SIZES[type][1];
}

float ggml_type_to_float(ggml_type type, uint8_t* data) {
    switch (type) {
        case GGML_TYPE_F32:
            return *((float*)data);
            
        case GGML_TYPE_F16: {
            uint16_t half = *((uint16_t*)data);
            
            // 提取各個欄位
            uint32_t sign = (half >> 15) & 0x00000001;
            uint32_t exp  = (half >> 10) & 0x0000001F;
            uint32_t frac = half & 0x000003FF;
            uint32_t result;

            // 情況 1: Zero (Exp=0, Frac=0)
            if (exp == 0 && frac == 0) {
                // 正確處理負零：將符號位移至 FP32 的最高位
                result = (sign << 31);
                return *((float*)&result);
            }

            // 情況 2: Subnormal (Exp=0, Frac!=0)
            else if (exp == 0) {
                // FP16 subnormal 的數值是: 0.frac * 2^-14
                // 我們需要將其正規化為: 1.xxxx * 2^Y
                
                // 只要第 10 bit (0x400) 是 0，就左移 frac 並減少指數
                // 這裡我們將 frac 正規化，直到隱含的 1 出現在第 10 bit
                // 由於 FP16 bias 是 15，正規化數最小 exp 是 1 (代表 2^-14)
                // Subnormal 其實也是 2^-14，但沒有隱含的 1
                
                // 先把 frac 移到正常位置，讓後續處理統一
                // 當前 frac 是 0.xxxx...
                // 我們透過迴圈找到第一個 1
                int shift_count = 0;
                while ((frac & 0x0400) == 0) {
                    frac <<= 1;
                    shift_count++;
                }
                
                // 移除隱含的 1 (bit 10)
                frac &= 0x03FF;
                
                // 計算新的指數
                // FP16 bias 15, FP32 bias 127. Diff = 112.
                // Subnormal 原始 effective exp 是 -14 (即 1 - 15)
                // 我們左移了 (shift_count + 1) 次才把 1 移出去 (因為原本沒有 hidden bit)
                // 這裡用更直觀的算法：
                // FP32 exp = 127 - 15 + 1 - shift_count = 113 - shift_count
                // (注意：這裡邏輯比較複雜，簡單的做法是下面的 result 組合)
                
                exp = 113 - shift_count; 
                
                // 組合 FP32
                // Frac 需要從 10 bits 擴展到 23 bits (左移 13)
                result = (sign << 31) | (exp << 23) | (frac << 13);
            }

            // 情況 3: Infinity or NaN (Exp=31)
            else if (exp == 31) {
                if (frac == 0) {
                    // Infinity
                    result = (sign << 31) | 0x7F800000;
                } else {
                    // NaN (保留 sign 和 frac 的高位以維持 NaN payload，或直接回傳標準 NaN)
                    result = (sign << 31) | 0x7F800000 | (frac << 13) | 0x400000; // 確保是 Quiet NaN
                }
            }

            // 情況 4: Normalized (Exp != 0 && Exp != 31)
            else {
                // 調整 Bias: +127 - 15 = +112
                exp = exp + 112;
                // 組合: Frac 左移 13 位填滿 FP32 的尾數部分
                result = (sign << 31) | (exp << 23) | (frac << 13);
            }

            return *((float*)&result);
        }
        
        default:
            fprintf(stderr, "Unsupported type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

// Initialize GGUF context from file with options
gguf_context* gguf_init_from_file_ex(const char* filename, bool load_tensors) {
    // Open file
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }
    
    // Get file size
    size_t file_size = get_file_size(file);
    
    // Allocate context
    gguf_context* ctx = (gguf_context*)calloc(1, sizeof(gguf_context));
    ctx->data = NULL;  // We won't map the entire file
    ctx->size = file_size;
    ctx->file = file;  // Keep file handle open
    ctx->byte_order = 'I';  // Assume native byte order initially
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
    ctx->tensors_loaded = false;
    
    // Read header into memory
    size_t header_size = 4 + 4 + 8 + 8;  // magic + version + tensor_count + kv_count
    uint8_t header[24];
    if (fread(header, 1, header_size, file) != header_size) {
        fprintf(stderr, "Failed to read header\n");
        fclose(file);
        free(ctx);
        return NULL;
    }
    
    size_t offset = 0;
    
    // Read and check magic
    uint32_t magic = *(uint32_t*)(header + offset);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        fclose(file);
        free(ctx);
        return NULL;
    }
    offset += 4;
    
    // Read version
    uint32_t version = *(uint32_t*)(header + offset);
    // Check if byte order needs to be swapped
    if ((version & 0xFFFF) == 0) {
        ctx->byte_order = 'S';
        maybe_swap_32(&version, ctx->byte_order);
    }
    ctx->version = version;
    offset += 4;
    
    if (version != 2 && version != 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        fclose(file);
        free(ctx);
        return NULL;
    }
    
    // Read tensor count and KV count
    uint64_t tensor_count = *(uint64_t*)(header + offset);
    maybe_swap_64(&tensor_count, ctx->byte_order);
    ctx->tensor_count = tensor_count;
    offset += 8;
    
    uint64_t kv_count = *(uint64_t*)(header + offset);
    maybe_swap_64(&kv_count, ctx->byte_order);
    ctx->kv_count = kv_count;
    offset += 8;
    
    printf("GGUF version: %u, tensors: %llu, metadata: %llu\n", 
           version, (unsigned long long)tensor_count, (unsigned long long)kv_count);
    
    // Read all metadata and tensor info into a large buffer
    // Estimate size: assume average 100 bytes per metadata entry and 200 bytes per tensor
    size_t estimated_header_size = kv_count * 100 + tensor_count * 200 + 1024 * 1024;  // +1MB safety
    uint8_t* data = (uint8_t*)malloc(estimated_header_size);
    if (!data) {
        fprintf(stderr, "Failed to allocate buffer for headers\n");
        fclose(file);
        free(ctx);
        return NULL;
    }
    
    size_t bytes_read = fread(data, 1, estimated_header_size, file);
    if (bytes_read == 0 && ferror(file)) {
        fprintf(stderr, "Failed to read file data\n");
        free(data);
        fclose(file);
        free(ctx);
        return NULL;
    }
    
    printf("Read %zu bytes of metadata and tensor info\n", bytes_read);
    
    // Read metadata key-value pairs
    ctx->metadata = (gguf_metadata_kv*)calloc(kv_count, sizeof(gguf_metadata_kv));
    offset = 0;
    for (uint64_t i = 0; i < kv_count; i++) {
        // Read key
        gguf_string key;
        size_t key_size = read_string(data, offset, &key, ctx->byte_order);
        offset += key_size;
        ctx->metadata[i].key = key.data;
        
        // Read value type
        uint32_t value_type = *(uint32_t*)(data + offset);
        maybe_swap_32(&value_type, ctx->byte_order);
        ctx->metadata[i].type = (gguf_type)value_type;
        offset += 4;
        
        // Read value
        size_t value_size = read_value(data, offset, ctx->metadata[i].type, &ctx->metadata[i].value, ctx->byte_order);
        offset += value_size;
    }
    
    // Check for alignment override in metadata
    for (uint64_t i = 0; i < kv_count; i++) {
        if (strcmp(ctx->metadata[i].key, "general.alignment") == 0) {
            if (ctx->metadata[i].type == GGUF_TYPE_UINT32) {
                ctx->alignment = ctx->metadata[i].value.uint32;
                printf("Using alignment from metadata: %u\n", ctx->alignment);
            }
            break;
        }
    }
    
    // Read tensor info
    ctx->tensors = (gguf_tensor*)calloc(tensor_count, sizeof(gguf_tensor));
    for (uint64_t i = 0; i < tensor_count; i++) {
        // Read tensor name
        gguf_string name;
        size_t name_size = read_string(data, offset, &name, ctx->byte_order);
        offset += name_size;
        ctx->tensors[i].name = name.data;
        
        // Read number of dimensions
        uint32_t n_dims = *(uint32_t*)(data + offset);
        maybe_swap_32(&n_dims, ctx->byte_order);
        ctx->tensors[i].n_dims = n_dims;
        offset += 4;
        
        // Read dimensions
        ctx->tensors[i].dims = (uint64_t*)malloc(n_dims * sizeof(uint64_t));
        size_t n_elements = 1;
        for (uint32_t j = 0; j < n_dims; j++) {
            uint64_t dim = *(uint64_t*)(data + offset);
            maybe_swap_64(&dim, ctx->byte_order);
            ctx->tensors[i].dims[j] = dim;
            n_elements *= dim;
            offset += 8;
        }
        ctx->tensors[i].n_elements = n_elements;
        
        // Read tensor type
        uint32_t tensor_type = *(uint32_t*)(data + offset);
        maybe_swap_32(&tensor_type, ctx->byte_order);
        ctx->tensors[i].type = (ggml_type)tensor_type;
        offset += 4;
        
        // Read tensor offset
        uint64_t tensor_offset = *(uint64_t*)(data + offset);
        maybe_swap_64(&tensor_offset, ctx->byte_order);
        ctx->tensors[i].offset = tensor_offset;
        offset += 8;
        
        // Calculate tensor size
        size_t block_size = GGML_QUANT_SIZES[tensor_type][0];
        size_t type_size = GGML_QUANT_SIZES[tensor_type][1];
        ctx->tensors[i].n_bytes = n_elements * type_size / block_size;
    }
    
    // Free metadata buffer
    free(data);
    
    // Calculate data offset with alignment
    size_t file_offset = header_size + offset;
    size_t padding = file_offset % ctx->alignment;
    if (padding != 0) {
        file_offset += ctx->alignment - padding;
    }
    ctx->data_offset = file_offset;
    
    printf("Data offset: %zu, file size: %zu\n", ctx->data_offset, file_size);
    
    printf("Data offset: %zu, file size: %zu\n", ctx->data_offset, file_size);
    
    // Load all tensors if requested
    if (load_tensors) {
        printf("Reading %llu tensors...\n", (unsigned long long)tensor_count);
        for (uint64_t i = 0; i < tensor_count; i++) {
            // Allocate memory for this tensor
            ctx->tensors[i].data = malloc(ctx->tensors[i].n_bytes);
            if (!ctx->tensors[i].data) {
                fprintf(stderr, "Failed to allocate %zu bytes for tensor %llu: %s\n", 
                        ctx->tensors[i].n_bytes, (unsigned long long)i, ctx->tensors[i].name);
                gguf_free(ctx);
                return NULL;
            }
            
            // Read tensor data from file
            size_t tensor_file_offset = ctx->data_offset + ctx->tensors[i].offset;
            if (read_data_at_offset(file, tensor_file_offset, ctx->tensors[i].data, ctx->tensors[i].n_bytes) != ctx->tensors[i].n_bytes) {
                fprintf(stderr, "Failed to read tensor %llu: %s\n", (unsigned long long)i, ctx->tensors[i].name);
                gguf_free(ctx);
                return NULL;
            }
            
            if ((i + 1) % 10 == 0 || i == tensor_count - 1) {
                printf("\rLoaded %llu/%llu tensors", (unsigned long long)(i + 1), (unsigned long long)tensor_count);
                fflush(stdout);
            }
        }
        printf("\n");
        ctx->tensors_loaded = true;
        fclose(file);
        ctx->file = NULL;
    } else {
        printf("Tensor data not loaded (metadata only mode)\n");
        // Keep file open for on-demand loading
    }
    
    return ctx;
}

// Initialize GGUF context from file (default: don't load tensor data)
gguf_context* gguf_init_from_file(const char* filename) {
    return gguf_init_from_file_ex(filename, true);
}

// Load individual tensor data on demand
bool gguf_load_tensor_data(gguf_context* ctx, gguf_tensor* tensor) {
    if (!ctx || !tensor) return false;
    
    // Check if already loaded
    if (tensor->data != NULL) return true;
    
    // Check if we have a file handle
    if (!ctx->file) {
        fprintf(stderr, "Cannot load tensor data: file is closed\n");
        return false;
    }
    
    // Allocate memory
    tensor->data = malloc(tensor->n_bytes);
    if (!tensor->data) {
        fprintf(stderr, "Failed to allocate %zu bytes for tensor: %s\n", 
                tensor->n_bytes, tensor->name);
        return false;
    }
    
    // Read from file
    size_t tensor_file_offset = ctx->data_offset + tensor->offset;
    if (read_data_at_offset(ctx->file, tensor_file_offset, tensor->data, tensor->n_bytes) != tensor->n_bytes) {
        fprintf(stderr, "Failed to read tensor: %s\n", tensor->name);
        free(tensor->data);
        tensor->data = NULL;
        return false;
    }
    
    return true;
}

// Free individual tensor data
void gguf_free_tensor_data(gguf_tensor* tensor) {
    if (tensor && tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
    }
}

// Free GGUF context
void gguf_free(gguf_context* ctx) {
    if (!ctx) return;
    
    // Free metadata
    if (ctx->metadata) {
        for (uint64_t i = 0; i < ctx->kv_count; i++) {
            if (ctx->metadata[i].key) {
                free(ctx->metadata[i].key);
            }
            free_value(ctx->metadata[i].type, &ctx->metadata[i].value);
        }
        free(ctx->metadata);
    }
    
    // Free tensors
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->tensor_count; i++) {
            if (ctx->tensors[i].name) {
                free(ctx->tensors[i].name);
            }
            if (ctx->tensors[i].dims) {
                free(ctx->tensors[i].dims);
            }
            if (ctx->tensors[i].data) {
                free(ctx->tensors[i].data);
            }
        }
        free(ctx->tensors);
    }
    
    // Close file if still open
    if (ctx->file) {
        fclose(ctx->file);
    }
    
    free(ctx);
}

// Get metadata by key
gguf_metadata_kv* gguf_get_metadata(gguf_context* ctx, const char* key) {
    if (!ctx || !key) return NULL;
    
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (strcmp(ctx->metadata[i].key, key) == 0) {
            return &ctx->metadata[i];
        }
    }
    
    return NULL;
}

// Get tensor by index
gguf_tensor* gguf_get_tensor(gguf_context* ctx, uint64_t index) {
    if (!ctx || index >= ctx->tensor_count) return NULL;
    return &ctx->tensors[index];
}

// Find tensor by name
gguf_tensor* gguf_find_tensor(gguf_context* ctx, const char* name) {
    if (!ctx || !name) return NULL;
    
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    
    return NULL;
}

// Helper function to get type name
static const char* get_type_name(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return "UINT8";
        case GGUF_TYPE_INT8:    return "INT8";
        case GGUF_TYPE_UINT16:  return "UINT16";
        case GGUF_TYPE_INT16:   return "INT16";
        case GGUF_TYPE_UINT32:  return "UINT32";
        case GGUF_TYPE_INT32:   return "INT32";
        case GGUF_TYPE_FLOAT32: return "FLOAT32";
        case GGUF_TYPE_BOOL:    return "BOOL";
        case GGUF_TYPE_STRING:  return "STRING";
        case GGUF_TYPE_ARRAY:   return "ARRAY";
        case GGUF_TYPE_UINT64:  return "UINT64";
        case GGUF_TYPE_INT64:   return "INT64";
        case GGUF_TYPE_FLOAT64: return "FLOAT64";
        default: return "UNKNOWN";
    }
}

static const char* get_ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_I8:   return "I8";
        case GGML_TYPE_I16:  return "I16";
        case GGML_TYPE_I32:  return "I32";
        case GGML_TYPE_I64:  return "I64";
        case GGML_TYPE_F64:  return "F64";
        default: return "UNKNOWN";
    }
}

// Print metadata value
static void print_metadata_value(gguf_type type, gguf_value* val) {
    switch (type) {
        case GGUF_TYPE_UINT8:
            printf("%u", val->uint8);
            break;
        case GGUF_TYPE_INT8:
            printf("%d", val->int8);
            break;
        case GGUF_TYPE_UINT16:
            printf("%u", val->uint16);
            break;
        case GGUF_TYPE_INT16:
            printf("%d", val->int16);
            break;
        case GGUF_TYPE_UINT32:
            printf("%u", val->uint32);
            break;
        case GGUF_TYPE_INT32:
            printf("%d", val->int32);
            break;
        case GGUF_TYPE_FLOAT32:
            printf("%f", val->float32);
            break;
        case GGUF_TYPE_BOOL:
            printf("%s", val->bool_ ? "true" : "false");
            break;
        case GGUF_TYPE_UINT64:
            printf("%llu", (unsigned long long)val->uint64);
            break;
        case GGUF_TYPE_INT64:
            printf("%lld", (long long)val->int64);
            break;
        case GGUF_TYPE_FLOAT64:
            printf("%f", val->float64);
            break;
        case GGUF_TYPE_STRING:
            printf("\"%s\"", val->string.data);
            break;
        case GGUF_TYPE_ARRAY: {
            gguf_array* arr = (gguf_array*)val->arr;
            printf("[");
            gguf_value* values = (gguf_value*)arr->data;
            for (uint64_t i = 0; i < arr->len && i < 10; i++) {  // Limit to first 10 elements
                if (i > 0) printf(", ");
                print_metadata_value(arr->type, &values[i]);
            }
            if (arr->len > 10) {
                printf(", ... (%llu total)", (unsigned long long)arr->len);
            }
            printf("]");
            break;
        }
        default:
            printf("<unknown>");
            break;
    }
}

// Print metadata
void gguf_print_metadata(gguf_context* ctx) {
    if (!ctx) return;
    
    printf("\n=== GGUF Metadata (%llu entries) ===\n", (unsigned long long)ctx->kv_count);
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        printf("  %-40s [%s] = ", ctx->metadata[i].key, get_type_name(ctx->metadata[i].type));
        print_metadata_value(ctx->metadata[i].type, &ctx->metadata[i].value);
        printf("\n");
    }
}

// Print tensors
void gguf_print_tensors(gguf_context* ctx) {
    if (!ctx) return;
    
    printf("\n=== GGUF Tensors (%llu tensors) ===\n", (unsigned long long)ctx->tensor_count);
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        gguf_tensor* tensor = &ctx->tensors[i];
        printf("  [%llu] %-50s ", (unsigned long long)i, tensor->name);
        printf("type=%-6s shape=[", get_ggml_type_name(tensor->type));
        for (uint32_t j = 0; j < tensor->n_dims; j++) {
            if (j > 0) printf(", ");
            printf("%llu", (unsigned long long)tensor->dims[j]);
        }
        printf("] n_elements=%zu n_bytes=%zu offset=%llu\n", 
               tensor->n_elements, tensor->n_bytes, (unsigned long long)tensor->offset);
    }
}
