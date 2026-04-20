#ifndef POLYGON_VALIDATOR_H
#define POLYGON_VALIDATOR_H

#include <vector>
#include <memory>
#include <mutex>

// Point structure for 2D coordinates
struct Point2D {
    float x;
    float y;
    
    Point2D() : x(0.0f), y(0.0f) {}
    Point2D(float _x, float _y) : x(_x), y(_y) {}
};

// Bounding box representation
struct BoundingBox {
    float x;      // center x
    float y;      // center y
    float width;
    float height;
    
    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(float _x, float _y, float _w, float _h) 
        : x(_x), y(_y), width(_w), height(_h) {}
    
    // Get corners of bounding box
    std::vector<Point2D> getCorners() const;
    
    // Get center point
    Point2D getCenter() const;
};

// Validation result
enum class ValidationStatus {
    FULLY_INSIDE,
    PARTIALLY_INSIDE,
    OUTSIDE
};

struct ValidationResult {
    ValidationStatus status;
    float overlapPercentage;  // 0.0 to 1.0
    int pointsInside;
    int totalPoints;
    
    ValidationResult() 
        : status(ValidationStatus::OUTSIDE), 
          overlapPercentage(0.0f),
          pointsInside(0),
          totalPoints(0) {}
};

// Main polygon validator class
class PolygonValidator {
public:
    PolygonValidator();
    ~PolygonValidator();
    
    // Update the polygon region (thread-safe)
    void setPolygon(const std::vector<Point2D>& polygon);
    
    // Get current polygon (thread-safe)
    std::vector<Point2D> getPolygon() const;
    
    // Clear the polygon
    void clearPolygon();
    
    // Check if polygon is set
    bool hasPolygon() const;
    
    // Validate a bounding box against the polygon
    ValidationResult validateBoundingBox(const BoundingBox& bbox) const;
    
    // Validate a set of points against the polygon
    ValidationResult validatePoints(const std::vector<Point2D>& points) const;
    
    // Set validation threshold (0.0 to 1.0)
    // If percentage of points inside >= threshold, status is FULLY_INSIDE
    // If percentage > 0 but < threshold, status is PARTIALLY_INSIDE
    void setValidationThreshold(float threshold);
    
    float getValidationThreshold() const;
    
    // Point-in-polygon test (static utility)
    static bool isPointInPolygon(const Point2D& point, 
                                  const std::vector<Point2D>& polygon);
    
private:
    std::vector<Point2D> polygon_;
    mutable std::mutex mutex_;
    float validationThreshold_;  // default 1.0 (all points must be inside)
    
    // Internal validation logic
    ValidationResult performValidation(const std::vector<Point2D>& points) const;
};

// C interface for FFI
extern "C" {
    // Create validator instance
    void* create_polygon_validator();
    
    // Destroy validator instance
    void free_polygon_validator(void* validator);
    
    // Set polygon points
    // points array format: [x1, y1, x2, y2, ..., xn, yn]
    void set_polygon(void* validator, const float* points, int32_t pointCount);
    
    // Clear polygon
    void clear_polygon(void* validator);
    
    // Check if polygon is set
    int32_t has_polygon(void* validator);
    
    // Get polygon points count
    int32_t get_polygon_point_count(void* validator);
    
    // Get polygon points
    // outPoints array format: [x1, y1, x2, y2, ..., xn, yn]
    void get_polygon_points(void* validator, float* outPoints);
    
    // Validate bounding box
    // Returns status: 0=OUTSIDE, 1=PARTIALLY_INSIDE, 2=FULLY_INSIDE
    int32_t validate_bounding_box(void* validator, 
                                   float x, float y, 
                                   float width, float height,
                                   float* outOverlapPercentage);
    
    // Validate custom points
    // points array format: [x1, y1, x2, y2, ..., xn, yn]
    int32_t validate_points(void* validator,
                           const float* points, 
                           int32_t pointCount,
                           float* outOverlapPercentage);
    
    // Set validation threshold
    void set_validation_threshold(void* validator, float threshold);
    
    // Get validation threshold
    float get_validation_threshold(void* validator);
}

#endif // POLYGON_VALIDATOR_H