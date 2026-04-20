import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

// Native function signatures
typedef CreatePolygonValidatorNative = Pointer<Void> Function();
typedef CreatePolygonValidatorDart = Pointer<Void> Function();

typedef FreePolygonValidatorNative = Void Function(Pointer<Void> validator);
typedef FreePolygonValidatorDart = void Function(Pointer<Void> validator);

typedef SetPolygonNative =
    Void Function(
      Pointer<Void> validator,
      Pointer<Float> points,
      Int32 pointCount,
    );
typedef SetPolygonDart =
    void Function(
      Pointer<Void> validator,
      Pointer<Float> points,
      int pointCount,
    );

typedef ClearPolygonNative = Void Function(Pointer<Void> validator);
typedef ClearPolygonDart = void Function(Pointer<Void> validator);

typedef HasPolygonNative = Int32 Function(Pointer<Void> validator);
typedef HasPolygonDart = int Function(Pointer<Void> validator);

typedef GetPolygonPointCountNative = Int32 Function(Pointer<Void> validator);
typedef GetPolygonPointCountDart = int Function(Pointer<Void> validator);

typedef GetPolygonPointsNative =
    Void Function(Pointer<Void> validator, Pointer<Float> outPoints);
typedef GetPolygonPointsDart =
    void Function(Pointer<Void> validator, Pointer<Float> outPoints);

typedef ValidateBoundingBoxNative =
    Int32 Function(
      Pointer<Void> validator,
      Float x,
      Float y,
      Float width,
      Float height,
      Pointer<Float> outOverlapPercentage,
    );
typedef ValidateBoundingBoxDart =
    int Function(
      Pointer<Void> validator,
      double x,
      double y,
      double width,
      double height,
      Pointer<Float> outOverlapPercentage,
    );

typedef ValidatePointsNative =
    Int32 Function(
      Pointer<Void> validator,
      Pointer<Float> points,
      Int32 pointCount,
      Pointer<Float> outOverlapPercentage,
    );
typedef ValidatePointsDart =
    int Function(
      Pointer<Void> validator,
      Pointer<Float> points,
      int pointCount,
      Pointer<Float> outOverlapPercentage,
    );

typedef SetValidationThresholdNative =
    Void Function(Pointer<Void> validator, Float threshold);
typedef SetValidationThresholdDart =
    void Function(Pointer<Void> validator, double threshold);

typedef GetValidationThresholdNative = Float Function(Pointer<Void> validator);
typedef GetValidationThresholdDart = double Function(Pointer<Void> validator);

/// Represents a 2D point with x and y coordinates
class Point2D {
  final double x;
  final double y;

  const Point2D(this.x, this.y);

  @override
  String toString() => 'Point2D($x, $y)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Point2D &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y;

  @override
  int get hashCode => x.hashCode ^ y.hashCode;
}

/// Validation status enum
enum ValidationStatus {
  outside(0),
  partiallyInside(1),
  fullyInside(2);

  final int value;
  const ValidationStatus(this.value);

  static ValidationStatus fromValue(int value) {
    switch (value) {
      case 0:
        return ValidationStatus.outside;
      case 1:
        return ValidationStatus.partiallyInside;
      case 2:
        return ValidationStatus.fullyInside;
      default:
        return ValidationStatus.outside;
    }
  }
}

/// Result of polygon validation
class PolygonValidationResult {
  final ValidationStatus status;
  final double overlapPercentage;

  const PolygonValidationResult({
    required this.status,
    required this.overlapPercentage,
  });

  bool get isFullyInside => status == ValidationStatus.fullyInside;
  bool get isPartiallyInside => status == ValidationStatus.partiallyInside;
  bool get isOutside => status == ValidationStatus.outside;

  @override
  String toString() =>
      'PolygonValidationResult(status: $status, overlap: ${(overlapPercentage * 100).toStringAsFixed(1)}%)';
}

/// Bridge to native polygon validator
class PolygonValidatorBridge {
  late final DynamicLibrary _lib;
  late final CreatePolygonValidatorDart _createValidator;
  late final FreePolygonValidatorDart _freeValidator;
  late final SetPolygonDart _setPolygon;
  late final ClearPolygonDart _clearPolygon;
  late final HasPolygonDart _hasPolygon;
  late final GetPolygonPointCountDart _getPolygonPointCount;
  late final GetPolygonPointsDart _getPolygonPoints;
  late final ValidateBoundingBoxDart _validateBoundingBox;
  late final ValidatePointsDart _validatePoints;
  late final SetValidationThresholdDart _setValidationThreshold;
  late final GetValidationThresholdDart _getValidationThreshold;

  Pointer<Void>? _validatorContext;

  PolygonValidatorBridge() {
    // Load the native library
    if (Platform.isAndroid) {
      _lib = DynamicLibrary.open('libpolygon_validator.so');
    } else if (Platform.isIOS) {
      _lib = DynamicLibrary.process();
    } else {
      throw UnsupportedError('Unsupported platform');
    }

    // Lookup functions
    _createValidator = _lib
        .lookup<NativeFunction<CreatePolygonValidatorNative>>(
          'create_polygon_validator',
        )
        .asFunction();

    _freeValidator = _lib
        .lookup<NativeFunction<FreePolygonValidatorNative>>(
          'free_polygon_validator',
        )
        .asFunction();

    _setPolygon = _lib
        .lookup<NativeFunction<SetPolygonNative>>('set_polygon')
        .asFunction();

    _clearPolygon = _lib
        .lookup<NativeFunction<ClearPolygonNative>>('clear_polygon')
        .asFunction();

    _hasPolygon = _lib
        .lookup<NativeFunction<HasPolygonNative>>('has_polygon')
        .asFunction();

    _getPolygonPointCount = _lib
        .lookup<NativeFunction<GetPolygonPointCountNative>>(
          'get_polygon_point_count',
        )
        .asFunction();

    _getPolygonPoints = _lib
        .lookup<NativeFunction<GetPolygonPointsNative>>('get_polygon_points')
        .asFunction();

    _validateBoundingBox = _lib
        .lookup<NativeFunction<ValidateBoundingBoxNative>>(
          'validate_bounding_box',
        )
        .asFunction();

    _validatePoints = _lib
        .lookup<NativeFunction<ValidatePointsNative>>('validate_points')
        .asFunction();

    _setValidationThreshold = _lib
        .lookup<NativeFunction<SetValidationThresholdNative>>(
          'set_validation_threshold',
        )
        .asFunction();

    _getValidationThreshold = _lib
        .lookup<NativeFunction<GetValidationThresholdNative>>(
          'get_validation_threshold',
        )
        .asFunction();

    // Initialize validator
    _validatorContext = _createValidator();
  }

  /// Set the polygon region of interest
  /// [points] - List of points defining the polygon (minimum 3 points)
  void setPolygon(List<Point2D> points) {
    if (_validatorContext == null || _validatorContext == nullptr) {
      throw StateError('Validator not initialized');
    }

    if (points.length < 3) {
      throw ArgumentError('Polygon must have at least 3 points');
    }

    // Flatten points array: [x1, y1, x2, y2, ...]
    final pointsArray = malloc<Float>(points.length * 2);
    try {
      for (int i = 0; i < points.length; i++) {
        pointsArray[i * 2] = points[i].x;
        pointsArray[i * 2 + 1] = points[i].y;
      }

      _setPolygon(_validatorContext!, pointsArray, points.length);
    } finally {
      malloc.free(pointsArray);
    }
  }

  /// Clear the current polygon
  void clearPolygon() {
    if (_validatorContext == null || _validatorContext == nullptr) return;
    _clearPolygon(_validatorContext!);
  }

  /// Check if a polygon is currently set
  bool hasPolygon() {
    if (_validatorContext == null || _validatorContext == nullptr) {
      return false;
    }
    return _hasPolygon(_validatorContext!) == 1;
  }

  /// Get the current polygon points
  List<Point2D> getPolygon() {
    if (_validatorContext == null || _validatorContext == nullptr) {
      return [];
    }

    final count = _getPolygonPointCount(_validatorContext!);
    if (count == 0) return [];

    final pointsArray = malloc<Float>(count * 2);
    try {
      _getPolygonPoints(_validatorContext!, pointsArray);

      final points = <Point2D>[];
      for (int i = 0; i < count; i++) {
        points.add(Point2D(pointsArray[i * 2], pointsArray[i * 2 + 1]));
      }

      return points;
    } finally {
      malloc.free(pointsArray);
    }
  }

  /// Validate a bounding box against the polygon
  /// [x, y] - Center coordinates of the bounding box
  /// [width, height] - Dimensions of the bounding box
  PolygonValidationResult validateBoundingBox({
    required double x,
    required double y,
    required double width,
    required double height,
  }) {
    if (_validatorContext == null || _validatorContext == nullptr) {
      return const PolygonValidationResult(
        status: ValidationStatus.outside,
        overlapPercentage: 0.0,
      );
    }

    final overlapPtr = malloc<Float>();
    try {
      final statusValue = _validateBoundingBox(
        _validatorContext!,
        x,
        y,
        width,
        height,
        overlapPtr,
      );

      return PolygonValidationResult(
        status: ValidationStatus.fromValue(statusValue),
        overlapPercentage: overlapPtr.value,
      );
    } finally {
      malloc.free(overlapPtr);
    }
  }

  /// Validate a set of custom points against the polygon
  /// [points] - List of points to validate
  PolygonValidationResult validatePoints(List<Point2D> points) {
    if (_validatorContext == null || _validatorContext == nullptr) {
      return const PolygonValidationResult(
        status: ValidationStatus.outside,
        overlapPercentage: 0.0,
      );
    }

    if (points.isEmpty) {
      return const PolygonValidationResult(
        status: ValidationStatus.outside,
        overlapPercentage: 0.0,
      );
    }

    final pointsArray = malloc<Float>(points.length * 2);
    final overlapPtr = malloc<Float>();

    try {
      // Flatten points array
      for (int i = 0; i < points.length; i++) {
        pointsArray[i * 2] = points[i].x;
        pointsArray[i * 2 + 1] = points[i].y;
      }

      final statusValue = _validatePoints(
        _validatorContext!,
        pointsArray,
        points.length,
        overlapPtr,
      );

      return PolygonValidationResult(
        status: ValidationStatus.fromValue(statusValue),
        overlapPercentage: overlapPtr.value,
      );
    } finally {
      malloc.free(pointsArray);
      malloc.free(overlapPtr);
    }
  }

  /// Set the validation threshold (0.0 to 1.0)
  /// Percentage of points that must be inside for status to be FULLY_INSIDE
  /// Default is 1.0 (all points must be inside)
  void setValidationThreshold(double threshold) {
    if (_validatorContext == null || _validatorContext == nullptr) return;
    _setValidationThreshold(_validatorContext!, threshold);
  }

  /// Get the current validation threshold
  double getValidationThreshold() {
    if (_validatorContext == null || _validatorContext == nullptr) {
      return 1.0;
    }
    return _getValidationThreshold(_validatorContext!);
  }

  /// Dispose the validator and free native resources
  void dispose() {
    if (_validatorContext != null && _validatorContext != nullptr) {
      _freeValidator(_validatorContext!);
      _validatorContext = null;
    }
  }
}
