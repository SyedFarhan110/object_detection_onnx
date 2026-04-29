#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <android/log.h>

#define TAG "CV_DETECTOR"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct CVContext {
    // thresholds and parameters
    float horizontal_edge_thresh = 5.0f;
    float vertical_edge_thresh = 5.0f;
    int min_shelves = 2;
    bool use_adaptive_threshold = true;
};

extern "C" {

void* create_cv_detector() {
    try {
        CVContext* ctx = new CVContext();
        ALOGI("create_cv_detector: created context %p", ctx);
        return ctx;
    } catch (...) {
        ALOGE("create_cv_detector: allocation failed");
        return nullptr;
    }
}

void free_cv_detector(void* ctx) {
    if (!ctx) return;
    ALOGI("free_cv_detector: freeing context %p", ctx);
    delete static_cast<CVContext*>(ctx);
}

// Non-maximum suppression helper
std::vector<int> apply_nms(const std::vector<int>& lines, 
                           const std::vector<float>& strengths, 
                           int min_distance) {
    std::vector<int> result;
    std::vector<bool> suppressed(lines.size(), false);
    
    // Sort indices by strength (descending)
    std::vector<size_t> indices(lines.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), 
              [&](size_t a, size_t b) { return strengths[lines[a]] > strengths[lines[b]]; });
    
    for (size_t idx : indices) {
        if (suppressed[idx]) continue;
        
        result.push_back(lines[idx]);
        
        // Suppress nearby lines
        for (size_t j = 0; j < lines.size(); ++j) {
            if (!suppressed[j] && std::abs(lines[idx] - lines[j]) < min_distance) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

// Run a lightweight rule-based detector on the Y (luma) plane.
// Returns a newly allocated float array [N*6] (x,y,w,h,conf,class_id) normalized to [0..1].
// Caller must free with free_cv_results.
float* run_cv_detector_yuv(void* ctx,
                           const uint8_t* y_plane, const uint8_t* /*u_plane*/, const uint8_t* /*v_plane*/,
                           int y_row_stride, int /*uv_row_stride*/, int /*uv_pixel_stride*/,
                           int img_w, int img_h,
                           int* out_count) {
    if (!ctx || !y_plane || img_w <= 0 || img_h <= 0 || !out_count) {
        if (out_count) *out_count = 0;
        return nullptr;
    }

    CVContext* c = static_cast<CVContext*>(ctx);

    ALOGI("run_cv_detector_yuv: START w=%d h=%d stride=%d", img_w, img_h, y_row_stride);

    // Compute per-row horizontal gradient magnitude
    std::vector<float> row_strength(img_h, 0.0f);
    float row_min = 999.0f, row_max = 0.0f, row_sum = 0.0f;
    
    for (int r = 0; r < img_h; ++r) {
        const uint8_t* row_ptr = y_plane + r * y_row_stride;
        float acc = 0.0f;
        for (int x = 1; x < img_w; ++x) {
            acc += std::abs((int)row_ptr[x] - (int)row_ptr[x - 1]);
        }
        row_strength[r] = acc / (float)img_w;
        row_min = std::min(row_min, row_strength[r]);
        row_max = std::max(row_max, row_strength[r]);
        row_sum += row_strength[r];
    }
    float row_mean = row_sum / (float)img_h;
    ALOGI("run_cv_detector_yuv: row gradients min=%.2f max=%.2f mean=%.2f", row_min, row_max, row_mean);

    // Compute per-column vertical gradient magnitude
    std::vector<float> col_strength(img_w, 0.0f);
    float col_min = 999.0f, col_max = 0.0f, col_sum = 0.0f;
    
    for (int x = 0; x < img_w; ++x) {
        float acc = 0.0f;
        for (int r = 1; r < img_h; ++r) {
            const uint8_t* row_ptr = y_plane + r * y_row_stride;
            const uint8_t* row_prev = y_plane + (r - 1) * y_row_stride;
            acc += std::abs((int)row_ptr[x] - (int)row_prev[x]);
        }
        col_strength[x] = acc / (float)img_h;
        col_min = std::min(col_min, col_strength[x]);
        col_max = std::max(col_max, col_strength[x]);
        col_sum += col_strength[x];
    }
    float col_mean = col_sum / (float)img_w;
    ALOGI("run_cv_detector_yuv: col gradients min=%.2f max=%.2f mean=%.2f", col_min, col_max, col_mean);

    // Adaptive thresholding using 75th percentile
    float h_thresh = c->horizontal_edge_thresh;
    float v_thresh = c->vertical_edge_thresh;
    
    if (c->use_adaptive_threshold) {
        std::vector<float> row_sorted = row_strength;
        std::vector<float> col_sorted = col_strength;
        std::sort(row_sorted.begin(), row_sorted.end());
        std::sort(col_sorted.begin(), col_sorted.end());
        
        size_t p75_idx_row = (size_t)(row_sorted.size() * 0.75f);
        size_t p75_idx_col = (size_t)(col_sorted.size() * 0.75f);
        
        // Use 75th percentile with a minimum threshold
        h_thresh = std::max(6.0f, row_sorted[p75_idx_row] * 0.9f);
        v_thresh = std::max(6.0f, col_sorted[p75_idx_col] * 0.9f);
        
        ALOGI("run_cv_detector_yuv: adaptive h_thresh=%.2f v_thresh=%.2f (75th percentile)", 
              h_thresh, v_thresh);
    }

    // Detect peaks in row_strength -> candidate horizontal lines
    std::vector<int> horiz_lines_raw;
    for (int r = 2; r < img_h - 2; ++r) {
        if (row_strength[r] > h_thresh &&
            row_strength[r] >= row_strength[r-1] &&
            row_strength[r] >= row_strength[r+1]) {
            horiz_lines_raw.push_back(r);
        }
    }

    // Apply NMS to horizontal lines (merge lines within 15 pixels)
    std::vector<int> horiz_lines = apply_nms(horiz_lines_raw, row_strength, 15);
    ALOGI("run_cv_detector_yuv: found %zu horiz line candidates (after NMS from %zu)", 
          horiz_lines.size(), horiz_lines_raw.size());

    // Detect peaks in col_strength -> candidate vertical lines
    std::vector<int> vert_lines_raw;
    for (int x = 2; x < img_w - 2; ++x) {
        if (col_strength[x] > v_thresh &&
            col_strength[x] >= col_strength[x-1] &&
            col_strength[x] >= col_strength[x+1]) {
            vert_lines_raw.push_back(x);
        }
    }

    // Apply NMS to vertical lines (merge lines within 15 pixels)
    std::vector<int> vert_lines = apply_nms(vert_lines_raw, col_strength, 15);
    ALOGI("run_cv_detector_yuv: found %zu vert line candidates (after NMS from %zu)", 
          vert_lines.size(), vert_lines_raw.size());

    // Check line counts are reasonable
    if (horiz_lines.size() < 3 || horiz_lines.size() > 25) {
        *out_count = 0;
        ALOGI("Invalid horiz line count: %zu (valid: 3-25)", horiz_lines.size());
        return nullptr;
    }

    if (vert_lines.size() < 2 || vert_lines.size() > 25) {
        *out_count = 0;
        ALOGI("Invalid vert line count: %zu (valid: 2-25)", vert_lines.size());
        return nullptr;
    }

    // Find repeated horizontal layer spacing
    bool repetition_ok = false;
    float avg_spacing = 0.0f;
    if (horiz_lines.size() >= 4) {
        std::vector<int> diffs;
        for (size_t i = 1; i < horiz_lines.size(); ++i) {
            diffs.push_back(horiz_lines[i] - horiz_lines[i-1]);
        }
        
        if (diffs.size() >= 3) {
            float sum = 0.0f;
            for (int d : diffs) sum += d;
            avg_spacing = sum / (float)diffs.size();
            
            float var = 0.0f;
            for (int d : diffs) var += (d - avg_spacing) * (d - avg_spacing);
            var /= diffs.size();
            float std_dev = std::sqrt(std::max(0.0f, var));
            
            // Relaxed: spacing variance < 25% AND spacing > 15 pixels
            repetition_ok = (avg_spacing > 15.0f && std_dev < (avg_spacing * 0.25f));
            ALOGI("run_cv_detector_yuv: repetition check - spacing=%.1f std_dev=%.1f ok=%d", 
                  avg_spacing, std_dev, repetition_ok ? 1 : 0);
        }
    }

    ALOGI("run_cv_detector_yuv: repetition_ok=%d avg_spacing=%.2f horiz_count=%zu", 
          repetition_ok ? 1 : 0, avg_spacing, horiz_lines.size());

    // Rectangular structure check
    bool rectangle_ok = (vert_lines.size() >= 2 && horiz_lines.size() >= 3);
    ALOGI("run_cv_detector_yuv: rectangle_ok=%d (v=%zu h=%zu)", 
          rectangle_ok ? 1 : 0, vert_lines.size(), horiz_lines.size());

    // Count grid intersections (with reasonable limits)
    int intersection_count = 0;
    const int CHECK_RADIUS = 5;
    for (int h : horiz_lines) {
        for (int v : vert_lines) {
            if (h > CHECK_RADIUS && h < img_h - CHECK_RADIUS && 
                v > CHECK_RADIUS && v < img_w - CHECK_RADIUS) {
                
                // Check local variance in 5x5 region around intersection
                float local_variance = 0.0f;
                const uint8_t* center_row = y_plane + h * y_row_stride;
                int center_val = center_row[v];
                
                for (int dy = -CHECK_RADIUS; dy <= CHECK_RADIUS; dy += 2) {
                    for (int dx = -CHECK_RADIUS; dx <= CHECK_RADIUS; dx += 2) {
                        int y = h + dy;
                        int x = v + dx;
                        if (y >= 0 && y < img_h && x >= 0 && x < img_w) {
                            const uint8_t* r = y_plane + y * y_row_stride;
                            local_variance += std::abs((int)r[x] - center_val);
                        }
                    }
                }
                
                if (local_variance > 200.0f) {
                    intersection_count++;
                }
            }
        }
    }

    ALOGI("run_cv_detector_yuv: grid intersections=%d", intersection_count);

    // Check for valid grid structure (not too few, not too many)
    bool has_grid_structure = (intersection_count >= 4 && intersection_count <= 1000);
    if (!has_grid_structure) {
        *out_count = 0;
        ALOGI("Invalid grid: %d intersections (need 4-1000)", intersection_count);
        return nullptr;
    }

    // Need either repetition OR rectangle structure
    if (!repetition_ok && !rectangle_ok) {
        *out_count = 0;
        ALOGI("run_cv_detector_yuv: no rack-like structure found");
        return nullptr;
    }

    // Compute bounding box from detected lines
    int left = vert_lines.front();
    int right = vert_lines.back();
    int top = horiz_lines.front();
    int bottom = horiz_lines.back();

    ALOGI("run_cv_detector_yuv: bbox before padding: l=%d r=%d t=%d b=%d", 
          left, right, top, bottom);

    // Expand margins slightly
    int pad_x = std::min(20, img_w / 20);
    int pad_y = std::min(20, img_h / 20);
    left = std::max(0, left - pad_x);
    right = std::min(img_w - 1, right + pad_x);
    top = std::max(0, top - pad_y);
    bottom = std::min(img_h - 1, bottom + pad_y);

    float cx = (left + right) / 2.0f;
    float cy = (top + bottom) / 2.0f;
    float bw = (right - left);
    float bh = (bottom - top);

    // Normalize to [0..1]
    float nx = cx / (float)img_w;
    float ny = cy / (float)img_h;
    float nw = bw / (float)img_w;
    float nh = bh / (float)img_h;

    // Confidence scoring
    float score = 0.0f;
    if (rectangle_ok) score += 0.35f;
    if (repetition_ok) score += 0.45f;
    
    // Bonus for reasonable intersection count (10-50 is ideal)
    if (intersection_count >= 10 && intersection_count <= 50) {
        score += 0.20f;
    }
    
    ALOGI("run_cv_detector_yuv: score=%.3f (rect=%d rep=%d inter=%d)", 
          score, rectangle_ok ? 1 : 0, repetition_ok ? 1 : 0, intersection_count);

    if (score < 0.35f) {
        *out_count = 0;
        ALOGI("run_cv_detector_yuv: score too low %.3f (min=0.35)", score);
        return nullptr;
    }

    // Prepare result array (single detection)
    *out_count = 1;
    float* res = new float[6];
    res[0] = nx; // x (center)
    res[1] = ny; // y (center)
    res[2] = nw; // w
    res[3] = nh; // h
    res[4] = score; // confidence
    res[5] = 0.0f; // class id = 0

    ALOGI("run_cv_detector_yuv: ✓✓✓ RACK DETECTED ✓✓✓ x=%.3f y=%.3f w=%.3f h=%.3f conf=%.3f", 
          nx, ny, nw, nh, score);

    return res;
}

void free_cv_results(float* results) {
    if (results) delete[] results;
}

} // extern C