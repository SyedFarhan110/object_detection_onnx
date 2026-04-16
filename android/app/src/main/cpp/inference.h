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

// Run inference using YUV420 image buffers from Android Camera
// Returns a pointer to a flat float array. 
// The results are structured as [N * 6], where N is the number of boxes.
// Each box has: [x_center, y_center, width, height, confidence, class_id]
// The x, y, width, height are normalized [0, 1].
float* run_inference_yuv(void* context, 
                         const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                         int y_row_stride, int uv_row_stride, int uv_pixel_stride,
                         int img_width, int img_height, 
                         int* out_count);

// Free the results pointer returned by run_inference
void free_results(float* results);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H