#include "polygon_validator.h"
#include <cmath>
#include <algorithm>
#include <limits>

// BoundingBox methods
std::vector<Point2D> BoundingBox::getCorners() const {
    float halfW = width / 2.0f;
    float halfH = height / 2.0f;
    
    return {
        Point2D(x - halfW, y - halfH),  // Top-left
        Point2D(x + halfW, y - halfH),  // Top-right
        Point2D(x + halfW, y + halfH),  // Bottom-right
        Point2D(x - halfW, y + halfH)   // Bottom-left
    };
}

Point2D BoundingBox::getCenter() const {
    return Point2D(x, y);
}

// PolygonValidator implementation
PolygonValidator::PolygonValidator() 
    : validationThreshold_(1.0f) {
}

PolygonValidator::~PolygonValidator() {
}

void PolygonValidator::setPolygon(const std::vector<Point2D>& polygon) {
    std::lock_guard<std::mutex> lock(mutex_);
    polygon_ = polygon;
}

std::vector<Point2D> PolygonValidator::getPolygon() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return polygon_;
}

void PolygonValidator::clearPolygon() {
    std::lock_guard<std::mutex> lock(mutex_);
    polygon_.clear();
}

bool PolygonValidator::hasPolygon() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return polygon_.size() >= 3;  // Minimum 3 points for a polygon
}

void PolygonValidator::setValidationThreshold(float threshold) {
    threshold = std::max(0.0f, std::min(1.0f, threshold));
    validationThreshold_ = threshold;
}

float PolygonValidator::getValidationThreshold() const {
    return validationThreshold_;
}

ValidationResult PolygonValidator::validateBoundingBox(const BoundingBox& bbox) const {
    // Get corners and center of bounding box
    std::vector<Point2D> points = bbox.getCorners();
    points.push_back(bbox.getCenter());  // Also check center point
    
    return performValidation(points);
}

ValidationResult PolygonValidator::validatePoints(const std::vector<Point2D>& points) const {
    return performValidation(points);
}

ValidationResult PolygonValidator::performValidation(const std::vector<Point2D>& points) const {
    ValidationResult result;
    
    if (points.empty()) {
        return result;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (polygon_.size() < 3) {
        // No valid polygon set
        return result;
    }
    
    result.totalPoints = static_cast<int>(points.size());
    result.pointsInside = 0;
    
    // Check each point
    for (const auto& point : points) {
        if (isPointInPolygon(point, polygon_)) {
            result.pointsInside++;
        }
    }
    
    // Calculate overlap percentage
    result.overlapPercentage = static_cast<float>(result.pointsInside) / 
                                static_cast<float>(result.totalPoints);
    
    // Determine status based on threshold
    if (result.overlapPercentage >= validationThreshold_) {
        result.status = ValidationStatus::FULLY_INSIDE;
    } else if (result.overlapPercentage > 0.0f) {
        result.status = ValidationStatus::PARTIALLY_INSIDE;
    } else {
        result.status = ValidationStatus::OUTSIDE;
    }
    
    return result;
}

// Ray casting algorithm for point-in-polygon test
bool PolygonValidator::isPointInPolygon(const Point2D& point, 
                                        const std::vector<Point2D>& polygon) {
    if (polygon.size() < 3) {
        return false;
    }
    
    bool inside = false;
    size_t n = polygon.size();
    
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const Point2D& vi = polygon[i];
        const Point2D& vj = polygon[j];
        
        // Ray casting: check if horizontal ray from point crosses edge
        if (((vi.y > point.y) != (vj.y > point.y)) &&
            (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)) {
            inside = !inside;
        }
    }
    
    return inside;
}

// C Interface Implementation
extern "C" {

void* create_polygon_validator() {
    try {
        return new PolygonValidator();
    } catch (...) {
        return nullptr;
    }
}

void free_polygon_validator(void* validator) {
    if (validator) {
        delete static_cast<PolygonValidator*>(validator);
    }
}

void set_polygon(void* validator, const float* points, int32_t pointCount) {
    if (!validator || !points || pointCount < 3) {
        return;
    }
    
    PolygonValidator* pv = static_cast<PolygonValidator*>(validator);
    std::vector<Point2D> polygon;
    polygon.reserve(pointCount);
    
    for (int32_t i = 0; i < pointCount; i++) {
        polygon.emplace_back(points[i * 2], points[i * 2 + 1]);
    }
    
    pv->setPolygon(polygon);
}

void clear_polygon(void* validator) {
    if (!validator) {
        return;
    }
    
    static_cast<PolygonValidator*>(validator)->clearPolygon();
}

int32_t has_polygon(void* validator) {
    if (!validator) {
        return 0;
    }
    
    return static_cast<PolygonValidator*>(validator)->hasPolygon() ? 1 : 0;
}

int32_t get_polygon_point_count(void* validator) {
    if (!validator) {
        return 0;
    }
    
    PolygonValidator* pv = static_cast<PolygonValidator*>(validator);
    return static_cast<int32_t>(pv->getPolygon().size());
}

void get_polygon_points(void* validator, float* outPoints) {
    if (!validator || !outPoints) {
        return;
    }
    
    PolygonValidator* pv = static_cast<PolygonValidator*>(validator);
    std::vector<Point2D> polygon = pv->getPolygon();
    
    for (size_t i = 0; i < polygon.size(); i++) {
        outPoints[i * 2] = polygon[i].x;
        outPoints[i * 2 + 1] = polygon[i].y;
    }
}

int32_t validate_bounding_box(void* validator, 
                               float x, float y, 
                               float width, float height,
                               float* outOverlapPercentage) {
    if (!validator) {
        if (outOverlapPercentage) {
            *outOverlapPercentage = 0.0f;
        }
        return 0;  // OUTSIDE
    }
    
    PolygonValidator* pv = static_cast<PolygonValidator*>(validator);
    BoundingBox bbox(x, y, width, height);
    ValidationResult result = pv->validateBoundingBox(bbox);
    
    if (outOverlapPercentage) {
        *outOverlapPercentage = result.overlapPercentage;
    }
    
    return static_cast<int32_t>(result.status);
}

int32_t validate_points(void* validator,
                       const float* points, 
                       int32_t pointCount,
                       float* outOverlapPercentage) {
    if (!validator || !points || pointCount <= 0) {
        if (outOverlapPercentage) {
            *outOverlapPercentage = 0.0f;
        }
        return 0;  // OUTSIDE
    }
    
    PolygonValidator* pv = static_cast<PolygonValidator*>(validator);
    std::vector<Point2D> testPoints;
    testPoints.reserve(pointCount);
    
    for (int32_t i = 0; i < pointCount; i++) {
        testPoints.emplace_back(points[i * 2], points[i * 2 + 1]);
    }
    
    ValidationResult result = pv->validatePoints(testPoints);
    
    if (outOverlapPercentage) {
        *outOverlapPercentage = result.overlapPercentage;
    }
    
    return static_cast<int32_t>(result.status);
}

void set_validation_threshold(void* validator, float threshold) {
    if (!validator) {
        return;
    }
    
    static_cast<PolygonValidator*>(validator)->setValidationThreshold(threshold);
}

float get_validation_threshold(void* validator) {
    if (!validator) {
        return 1.0f;
    }
    
    return static_cast<PolygonValidator*>(validator)->getValidationThreshold();
}

}  // extern "C"