#include "inference.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <android/log.h>
#include <string>

#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, "ONNX_INFERENCE", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ONNX_INFERENCE", __VA_ARGS__)

struct ModelContext {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_node_dims;
    size_t input_width;
    size_t input_height;
    std::string input_name_str;
    std::string output_name_str;
    
    // Frame skipping logic
    int skip_mode;        // 0 = no skip, 1 = every 2nd frame, 2 = skip N consecutive
    int skip_count;       // number of frames to skip (for mode 2)
    int frame_counter;    // internal frame counter
    
    ModelContext(const char* model_path) : env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference"), session(nullptr) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session = Ort::Session(env, model_path, session_options);
        
        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        input_name_str = input_name_alloc.get();
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
        output_name_str = output_name_alloc.get();

        Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims = tensor_info.GetShape();

        // Deal with dynamic batch size
        if(input_node_dims[0] < 0) {
            input_node_dims[0] = 1;
        }

        input_width = input_node_dims[3];
        input_height = input_node_dims[2];
        
        // Initialize frame skip settings
        skip_mode = 0;
        skip_count = 1;
        frame_counter = 0;

        ALOGI("Loaded model. Input [%d,%d,%d,%d]", 
            (int)input_node_dims[0], (int)input_node_dims[1], (int)input_node_dims[2], (int)input_node_dims[3]);
    }
};

void* init_model(const char* model_path) {
    try {
        ModelContext* ctx = new ModelContext(model_path);
        return ctx;
    } catch (const std::exception& e) {
        ALOGE("Failed to load model: %s", e.what());
        return nullptr;
    }
}

void free_model(void* context) {
    if (context) {
        delete static_cast<ModelContext*>(context);
    }
}

void set_frame_skip(void* context, int skip_mode, int skip_count) {
    if (!context) return;
    ModelContext* ctx = static_cast<ModelContext*>(context);
    
    ctx->skip_mode = skip_mode;
    ctx->skip_count = skip_count > 0 ? skip_count : 1;
    ctx->frame_counter = 0; // Reset counter when mode changes
    
    ALOGI("Frame skip mode set to: %d, skip_count: %d", skip_mode, skip_count);
}

int should_process_frame(void* context) {
    if (!context) return 1; // Process by default
    ModelContext* ctx = static_cast<ModelContext*>(context);
    
    bool should_process = false;
    
    switch (ctx->skip_mode) {
        case 0: // No skip - process every frame
            should_process = true;
            break;
            
        case 1: // Skip every 2nd frame (process frame 0, 2, 4, 6...)
            should_process = (ctx->frame_counter % 2 == 0);
            break;
            
        case 2: // Skip N consecutive frames
            // Process frame, then skip N frames, then process again
            // Example: skip_count=2 -> process 0, skip 1,2, process 3, skip 4,5, process 6...
            should_process = (ctx->frame_counter % (ctx->skip_count + 1) == 0);
            break;
            
        default:
            should_process = true;
            break;
    }
    
    ctx->frame_counter++;
    
    return should_process ? 1 : 0;
}

void preprocess_yuv(const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                    int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                    int img_w, int img_h,
                    float* out_tensor, size_t out_w, size_t out_h,
                    bool rotate_90, float& scale, float& pad_x, float& pad_y) {
    
    size_t channel_stride = out_w * out_h;
    float* r_plane = out_tensor;
    float* g_plane = out_tensor + channel_stride;
    float* b_plane = out_tensor + 2 * channel_stride;

    // Fill with gray (114.0/255.0) for letterbox padding
    std::fill_n(r_plane, channel_stride, 114.0f / 255.0f);
    std::fill_n(g_plane, channel_stride, 114.0f / 255.0f);
    std::fill_n(b_plane, channel_stride, 114.0f / 255.0f);

    int src_w = img_w;
    int src_h = img_h;
    int logical_w = rotate_90 ? img_h : img_w;
    int logical_h = rotate_90 ? img_w : img_h;

    scale = std::min((float)out_w / (float)logical_w, (float)out_h / (float)logical_h);
    int new_w = (int)(logical_w * scale);
    int new_h = (int)(logical_h * scale);

    pad_x = (out_w - new_w) / 2.0f;
    pad_y = (out_h - new_h) / 2.0f;

    // Resize with CORRECT YUV to RGB conversion + Rotation + Letterboxing
    for(int dy = 0; dy < new_h; dy++) {
        for(int dx = 0; dx < new_w; dx++) {
            // Map to logical image coords
            int logical_x = dx / scale;
            int logical_y = dy / scale;

            // Map to original source image coords
            int sx, sy;
            if (rotate_90) {
                sx = logical_y; // 90 degree clockwise
                sy = src_h - 1 - logical_x;
            } else {
                sx = logical_x;
                sy = logical_y;
            }

            sx = std::max(0, std::min(sx, src_w - 1));
            sy = std::max(0, std::min(sy, src_h - 1));

            // Read raw YUV values
            int y_raw = y_plane[sy * y_row_stride + sx];
            int uv_idx = (sy / 2) * uv_row_stride + (sx / 2) * uv_pixel_stride;
            int u_raw = u_plane[uv_idx];
            int v_raw = v_plane[uv_idx];

            // Scale from LIMITED range to FULL range
            float y = (y_raw - 16.0f) * 255.0f / 219.0f;
            float u = (u_raw - 128.0f) * 255.0f / 224.0f;
            float v = (v_raw - 128.0f) * 255.0f / 224.0f;

            // BT.601 YUV to RGB conversion
            float r = y + 1.402f * v;
            float g = y - 0.344136f * u - 0.714136f * v;
            float b = y + 1.772f * u;

            // Clamp to [0, 255]
            r = std::max(0.0f, std::min(255.0f, r));
            g = std::max(0.0f, std::min(255.0f, g));
            b = std::max(0.0f, std::min(255.0f, b));

            size_t out_idx = (dy + (int)pad_y) * out_w + (dx + (int)pad_x);
            
            // YOLO normalization (0-1)
            r_plane[out_idx] = r / 255.0f;
            g_plane[out_idx] = g / 255.0f;
            b_plane[out_idx] = b / 255.0f;
        }
    }
}

static float iou(const float* a, const float* b) {
    float x1 = std::max(a[0] - a[2]/2.0f, b[0] - b[2]/2.0f);
    float y1 = std::max(a[1] - a[3]/2.0f, b[1] - b[3]/2.0f);
    float x2 = std::min(a[0] + a[2]/2.0f, b[0] + b[2]/2.0f);
    float y2 = std::min(a[1] + a[3]/2.0f, b[1] + b[3]/2.0f);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float a_area = a[2] * a[3];
    float b_area = b[2] * b[3];
    
    return inter_area / (a_area + b_area - inter_area + 1e-6);
}

float* run_inference_yuv(void* context, 
                         const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                         int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                         int img_w, int img_h, 
                         int* out_count) {
    *out_count = 0;
    if (!context) return nullptr;
    ModelContext* ctx = static_cast<ModelContext*>(context);

    std::vector<float> input_tensor_values(1 * 3 * ctx->input_width * ctx->input_height);
    
    bool rotate_90 = (img_w > img_h);
    float scale, pad_x, pad_y;

    preprocess_yuv(y_plane, u_plane, v_plane, 
                   y_row_stride, uv_row_stride, uv_pixel_stride, 
                   img_w, img_h, 
                   input_tensor_values.data(), ctx->input_width, ctx->input_height,
                   rotate_90, scale, pad_x, pad_y);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), ctx->input_node_dims.data(), ctx->input_node_dims.size());

    const char* input_names[] = {ctx->input_name_str.c_str()};
    const char* output_names[] = {ctx->output_name_str.c_str()};

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = ctx->session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    } catch(const std::exception& e) {
        ALOGE("Inference error: %s", e.what());
        return nullptr;
    }

    if(output_tensors.empty()) return nullptr;

    float* raw_output = output_tensors.front().GetTensorMutableData<float>();
    auto out_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();

    int dim1 = out_shape.size() > 1 ? out_shape[1] : 1;
    int dim2 = out_shape.size() > 2 ? out_shape[2] : 1;

    int num_anchors, k_elements;
    bool transpose = false;

    if (dim2 > dim1) {
        num_anchors = dim2;
        k_elements = dim1;
        transpose = true; 
    } else {
        num_anchors = dim1;
        k_elements = dim2;
    }

    // Heuristically determine YOLOv5 vs YOLOv8
    // YOLOv5 COCO output is 85, YOLOv8 COCO is 84
    bool has_objectness = false;
    if (k_elements == 85 || (k_elements > 4 && k_elements != 6 && k_elements != 84 && k_elements % 5 == 0)) { 
        has_objectness = true; // Likely YOLOv5
    }

    float conf_thresh = 0.25f;
    std::vector<std::vector<float>> valid_boxes;
    int logical_w = rotate_90 ? img_h : img_w;
    int logical_h = rotate_90 ? img_w : img_h;

    for (int i = 0; i < num_anchors; ++i) {
        std::vector<float> row(k_elements);
        for(int j=0; j<k_elements; ++j) {
            row[j] = transpose ? raw_output[j * num_anchors + i] : raw_output[i * k_elements + j];
        }

        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];

        float objectness = 1.0f;
        int class_start_idx = 4;

        if (has_objectness && k_elements > 4) {
            objectness = row[4];
            class_start_idx = 5;
        }

        float max_class_score = -1.0f;
        int class_id = -1;

        for (int c = class_start_idx; c < k_elements; ++c) {
            if (row[c] > max_class_score) {
                max_class_score = row[c];
                class_id = c - class_start_idx;
            }
        }

        float conf = has_objectness ? (objectness * max_class_score) : max_class_score;

        if (conf > conf_thresh) {
            float box_x = x;
            float box_y = y;
            float box_w = w;
            float box_h = h;
            
            // Remove letterbox padding and scaling
            if (x <= 2.0f && y <= 2.0f && w <= 2.0f && h <= 2.0f) { // If model outputs normalized
                box_x *= ctx->input_width;
                box_y *= ctx->input_height;
                box_w *= ctx->input_width;
                box_h *= ctx->input_height;
            }
            
            box_x = (box_x - pad_x) / scale;
            box_y = (box_y - pad_y) / scale;
            box_w = box_w / scale;
            box_h = box_h / scale;

            // Normalize relative to the logical orientation the image is currently in
            float nx = box_x / logical_w;
            float ny = box_y / logical_h;
            float nw = box_w / logical_w;
            float nh = box_h / logical_h;
            
            valid_boxes.push_back({nx, ny, nw, nh, conf, (float)class_id});
        }
    }

    // NMS
    float nms_thresh = 0.45f;
    std::sort(valid_boxes.begin(), valid_boxes.end(), [](const std::vector<float>& a, const std::vector<float>& b) {
        return a[4] > b[4];
    });

    std::vector<std::vector<float>> final_boxes;
    for (const auto& box : valid_boxes) {
        bool keep = true;
        for (const auto& f_box : final_boxes) {
            if (box[5] == f_box[5]) { // Same class
                if (iou(box.data(), f_box.data()) > nms_thresh) {
                    keep = false;
                    break;
                }
            }
        }
        if (keep) {
            final_boxes.push_back(box);
        }
    }

    *out_count = (int)final_boxes.size();
    if (final_boxes.empty()) return nullptr;

    float* result_array = new float[final_boxes.size() * 6];
    for (size_t i = 0; i < final_boxes.size(); ++i) {
        result_array[i * 6 + 0] = final_boxes[i][0];
        result_array[i * 6 + 1] = final_boxes[i][1];
        result_array[i * 6 + 2] = final_boxes[i][2];
        result_array[i * 6 + 3] = final_boxes[i][3];
        result_array[i * 6 + 4] = final_boxes[i][4];
        result_array[i * 6 + 5] = final_boxes[i][5];
    }

    return result_array;
}

void free_results(float* results) {
    if(results) {
        delete[] results;
    }
}