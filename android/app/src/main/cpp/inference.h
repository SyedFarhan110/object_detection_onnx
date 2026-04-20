#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the ONNX model and return a context pointer
void* init_model(const char* model_path);

// Free the model context
void free_model(void* context);

// Set frame skip mode
// skip_mode: 0 = no skip, 1 = every 2nd frame, 2 = skip N consecutive frames
// skip_count: only used when skip_mode = 2 (number of frames to skip)
void set_frame_skip(void* context, int skip_mode, int skip_count);

// Check if the current frame should be processed
// Returns 1 if should process, 0 if should skip
// This increments the internal frame counter
int should_process_frame(void* context);

// ============ Producer-Consumer Queue API ============

// Start the inference worker thread (consumer)
// max_queue_size: maximum number of frames to buffer (recommended: 2-3)
void start_inference_worker(void* context, int max_queue_size);

// Stop the inference worker thread
void stop_inference_worker(void* context);

// Push a frame to the inference queue (producer - called from camera thread)
// Returns 1 if frame was queued, 0 if queue is full (frame dropped)
// The data is copied internally, so caller can free the buffers immediately
int push_frame_to_queue(void* context,
                        const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                        int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                        int img_width, int img_height);

// Get the latest detection results (non-blocking)
// Returns pointer to results array, or nullptr if no new results
// out_count: number of detections
// out_timestamp: timestamp of when this frame was captured (milliseconds)
// Caller should NOT free the returned pointer - it's managed internally
const float* get_latest_results(void* context, int* out_count, int64_t* out_timestamp);

// ============ Legacy Synchronous API (for backward compatibility) ============

// Run inference using YUV420 image buffers from Android Camera (synchronous)
// Returns a pointer to a flat float array. 
// The results are structured as [N * 6], where N is the number of boxes.
// Each box has: [x_center, y_center, width, height, confidence, class_id]
// The x, y, width, height are normalized [0, 1].
// Caller must free the returned pointer using free_results()
float* run_inference_yuv(void* context, 
                         const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                         int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                         int img_width, int img_height, 
                         int* out_count);

// Free the results pointer returned by run_inference_yuv (legacy API only)
void free_results(float* results);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H