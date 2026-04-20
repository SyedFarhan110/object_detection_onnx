#include "inference.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <android/log.h>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>

#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, "ONNX_INFERENCE", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ONNX_INFERENCE", __VA_ARGS__)

// Frame data structure for the queue
struct FrameData {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> u_plane;
    std::vector<uint8_t> v_plane;
    int y_row_stride;
    int uv_row_stride;
    int uv_pixel_stride;
    int img_width;
    int img_height;
    int64_t timestamp_ms;
    
    FrameData() = default;
    
    FrameData(const uint8_t* y, const uint8_t* u, const uint8_t* v,
              int y_stride, int uv_stride, int uv_pixel_stride,
              int width, int height)
        : y_row_stride(y_stride)
        , uv_row_stride(uv_stride)
        , uv_pixel_stride(uv_pixel_stride)
        , img_width(width)
        , img_height(height)
    {
        // Copy Y plane
        size_t y_size = y_stride * height;
        y_plane.resize(y_size);
        std::copy(y, y + y_size, y_plane.begin());
        
        // Copy U plane
        size_t uv_height = (height + 1) / 2;
        size_t u_size = uv_stride * uv_height;
        u_plane.resize(u_size);
        std::copy(u, u + u_size, u_plane.begin());
        
        // Copy V plane
        v_plane.resize(u_size);
        std::copy(v, v + u_size, v_plane.begin());
        
        // Timestamp
        auto now = std::chrono::system_clock::now();
        timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
    }
};

// Detection results structure
struct DetectionResults {
    std::vector<float> boxes; // Flat array: [x, y, w, h, conf, class_id] * N
    int count;
    int64_t timestamp_ms;
    
    DetectionResults() : count(0), timestamp_ms(0) {}
};

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
    int skip_mode;
    int skip_count;
    int frame_counter;
    
    // Producer-Consumer Queue
    std::queue<FrameData> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    int max_queue_size;
    std::atomic<bool> worker_running;
    std::thread worker_thread;
    
    // Latest detection results (double buffering)
    DetectionResults latest_results;
    std::mutex results_mutex;
    
    // Statistics
    std::atomic<int> frames_dropped;
    std::atomic<int> frames_processed;
    
    ModelContext(const char* model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference")
        , session(nullptr)
        , max_queue_size(2)
        , worker_running(false)
        , frames_dropped(0)
        , frames_processed(0)
    {
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

        if(input_node_dims[0] < 0) {
            input_node_dims[0] = 1;
        }

        input_width = input_node_dims[3];
        input_height = input_node_dims[2];
        
        skip_mode = 0;
        skip_count = 1;
        frame_counter = 0;

        ALOGI("Loaded model. Input [%d,%d,%d,%d]", 
            (int)input_node_dims[0], (int)input_node_dims[1], 
            (int)input_node_dims[2], (int)input_node_dims[3]);
    }
    
    ~ModelContext() {
        // Stop worker if running
        if (worker_running) {
            stop_worker();
        }
    }
    
    void start_worker(int queue_size) {
        if (worker_running) {
            ALOGE("Worker already running");
            return;
        }
        
        max_queue_size = queue_size;
        worker_running = true;
        frames_dropped = 0;
        frames_processed = 0;
        
        worker_thread = std::thread(&ModelContext::worker_loop, this);
        ALOGI("Inference worker started with queue size %d", max_queue_size);
    }
    
    void stop_worker() {
        if (!worker_running) return;
        
        worker_running = false;
        queue_cv.notify_all();
        
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        
        // Clear queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            std::queue<FrameData> empty;
            std::swap(frame_queue, empty);
        }
        
        ALOGI("Inference worker stopped. Processed: %d, Dropped: %d", 
              frames_processed.load(), frames_dropped.load());
    }
    
    void worker_loop() {
        ALOGI("Worker thread started");
        
        while (worker_running) {
            FrameData frame;
            
            // Wait for frame
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { 
                    return !frame_queue.empty() || !worker_running; 
                });
                
                if (!worker_running && frame_queue.empty()) {
                    break;
                }
                
                if (!frame_queue.empty()) {
                    frame = std::move(frame_queue.front());
                    frame_queue.pop();
                }
            }
            
            // Process frame
            if (!frame.y_plane.empty()) {
                int count = 0;
                float* results = run_inference_internal(
                    frame.y_plane.data(),
                    frame.u_plane.data(),
                    frame.v_plane.data(),
                    frame.y_row_stride,
                    frame.uv_row_stride,
                    frame.uv_pixel_stride,
                    frame.img_width,
                    frame.img_height,
                    &count
                );
                
                // Update latest results
                {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    latest_results.count = count;
                    latest_results.timestamp_ms = frame.timestamp_ms;
                    
                    if (results && count > 0) {
                        latest_results.boxes.assign(results, results + (count * 6));
                        delete[] results;
                    } else {
                        latest_results.boxes.clear();
                    }
                }
                
                frames_processed++;
            }
        }
        
        ALOGI("Worker thread exiting");
    }
    
    float* run_inference_internal(
        const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
        int y_row_stride, int uv_row_stride, int uv_pixel_stride,
        int img_w, int img_h,
        int* out_count);
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
    ctx->frame_counter = 0;
    
    ALOGI("Frame skip mode set to: %d, skip_count: %d", skip_mode, skip_count);
}

int should_process_frame(void* context) {
    if (!context) return 1;
    ModelContext* ctx = static_cast<ModelContext*>(context);
    
    bool should_process = false;
    
    switch (ctx->skip_mode) {
        case 0:
            should_process = true;
            break;
        case 1:
            should_process = (ctx->frame_counter % 2 == 0);
            break;
        case 2:
            should_process = (ctx->frame_counter % (ctx->skip_count + 1) == 0);
            break;
        default:
            should_process = true;
            break;
    }
    
    ctx->frame_counter++;
    return should_process ? 1 : 0;
}

// ============ Producer-Consumer API ============

void start_inference_worker(void* context, int max_queue_size) {
    if (!context) return;
    ModelContext* ctx = static_cast<ModelContext*>(context);
    ctx->start_worker(max_queue_size);
}

void stop_inference_worker(void* context) {
    if (!context) return;
    ModelContext* ctx = static_cast<ModelContext*>(context);
    ctx->stop_worker();
}

int push_frame_to_queue(void* context,
                        const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                        int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                        int img_width, int img_height) {
    if (!context) return 0;
    ModelContext* ctx = static_cast<ModelContext*>(context);
    
    if (!ctx->worker_running) {
        ALOGE("Worker not running - call start_inference_worker first");
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(ctx->queue_mutex);
    
    // Drop frame if queue is full
    if ((int)ctx->frame_queue.size() >= ctx->max_queue_size) {
        ctx->frames_dropped++;
        return 0; // Queue full
    }
    
    // Create and push frame
    ctx->frame_queue.emplace(
        y_plane, u_plane, v_plane,
        y_row_stride, uv_row_stride, uv_pixel_stride,
        img_width, img_height
    );
    
    ctx->queue_cv.notify_one();
    return 1; // Success
}

const float* get_latest_results(void* context, int* out_count, int64_t* out_timestamp) {
    if (!context) {
        *out_count = 0;
        *out_timestamp = 0;
        return nullptr;
    }
    
    ModelContext* ctx = static_cast<ModelContext*>(context);
    std::lock_guard<std::mutex> lock(ctx->results_mutex);
    
    *out_count = ctx->latest_results.count;
    *out_timestamp = ctx->latest_results.timestamp_ms;
    
    if (ctx->latest_results.boxes.empty()) {
        return nullptr;
    }
    
    return ctx->latest_results.boxes.data();
}

// ============ Preprocessing ============

void preprocess_yuv(const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                    int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                    int img_w, int img_h,
                    float* out_tensor, size_t out_w, size_t out_h,
                    bool rotate_90, float& scale, float& pad_x, float& pad_y) {
    
    size_t channel_stride = out_w * out_h;
    float* r_plane = out_tensor;
    float* g_plane = out_tensor + channel_stride;
    float* b_plane = out_tensor + 2 * channel_stride;

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

    for(int dy = 0; dy < new_h; dy++) {
        for(int dx = 0; dx < new_w; dx++) {
            int logical_x = dx / scale;
            int logical_y = dy / scale;

            int sx, sy;
            if (rotate_90) {
                sx = logical_y;
                sy = src_h - 1 - logical_x;
            } else {
                sx = logical_x;
                sy = logical_y;
            }

            sx = std::max(0, std::min(sx, src_w - 1));
            sy = std::max(0, std::min(sy, src_h - 1));

            int y_raw = y_plane[sy * y_row_stride + sx];
            int uv_idx = (sy / 2) * uv_row_stride + (sx / 2) * uv_pixel_stride;
            int u_raw = u_plane[uv_idx];
            int v_raw = v_plane[uv_idx];

            float y = (y_raw - 16.0f) * 255.0f / 219.0f;
            float u = (u_raw - 128.0f) * 255.0f / 224.0f;
            float v = (v_raw - 128.0f) * 255.0f / 224.0f;

            float r = y + 1.402f * v;
            float g = y - 0.344136f * u - 0.714136f * v;
            float b = y + 1.772f * u;

            r = std::max(0.0f, std::min(255.0f, r));
            g = std::max(0.0f, std::min(255.0f, g));
            b = std::max(0.0f, std::min(255.0f, b));

            size_t out_idx = (dy + (int)pad_y) * out_w + (dx + (int)pad_x);
            
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

float* ModelContext::run_inference_internal(
    const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
    int y_row_stride, int uv_row_stride, int uv_pixel_stride,
    int img_w, int img_h,
    int* out_count) {
    
    *out_count = 0;

    std::vector<float> input_tensor_values(1 * 3 * input_width * input_height);
    
    bool rotate_90 = (img_w > img_h);
    float scale, pad_x, pad_y;

    preprocess_yuv(y_plane, u_plane, v_plane, 
                   y_row_stride, uv_row_stride, uv_pixel_stride, 
                   img_w, img_h, 
                   input_tensor_values.data(), input_width, input_height,
                   rotate_90, scale, pad_x, pad_y);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), 
        input_node_dims.data(), input_node_dims.size());

    const char* input_names[] = {input_name_str.c_str()};
    const char* output_names[] = {output_name_str.c_str()};

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
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

    bool has_objectness = false;
    if (k_elements == 85 || (k_elements > 4 && k_elements != 6 && k_elements != 84 && k_elements % 5 == 0)) { 
        has_objectness = true;
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
            
            if (x <= 2.0f && y <= 2.0f && w <= 2.0f && h <= 2.0f) {
                box_x *= input_width;
                box_y *= input_height;
                box_w *= input_width;
                box_h *= input_height;
            }
            
            box_x = (box_x - pad_x) / scale;
            box_y = (box_y - pad_y) / scale;
            box_w = box_w / scale;
            box_h = box_h / scale;

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
            if (box[5] == f_box[5]) {
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

// Legacy synchronous API
float* run_inference_yuv(void* context, 
                         const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                         int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                         int img_w, int img_h, 
                         int* out_count) {
    *out_count = 0;
    if (!context) return nullptr;
    ModelContext* ctx = static_cast<ModelContext*>(context);
    
    return ctx->run_inference_internal(
        y_plane, u_plane, v_plane,
        y_row_stride, uv_row_stride, uv_pixel_stride,
        img_w, img_h, out_count
    );
}

void free_results(float* results) {
    if(results) {
        delete[] results;
    }
}