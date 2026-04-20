import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:object_detection_app/services/polygon_validator_bridge.dart';
import 'package:object_detection_app/services/inference_bridge.dart';

/// Real-time polygon-based spatial validation for rack detection
class RackValidationExample extends StatefulWidget {
  final CameraDescription camera;

  const RackValidationExample({Key? key, required this.camera})
    : super(key: key);

  @override
  State<RackValidationExample> createState() => _RackValidationExampleState();
}

class _RackValidationExampleState extends State<RackValidationExample> {
  static const bool _enableVerboseLogs = true;
  static const int _personClassId = 0;

  // Bridges
  late final InferenceBridge _inferenceBridge;
  late final PolygonValidatorBridge _polygonValidator;

  // Camera
  CameraController? _cameraController;
  bool _isCameraInitialized = false;

  // UI state
  List<Point2D> _polygonPoints = [];
  bool _isDrawingPolygon = false;
  ValidationStatus? _lastValidationStatus;
  double _validationThreshold = 1.0;

  // Performance optimization
  bool _isProcessing = false;
  int _frameCounter = 0;

  void _log(String message) {
    if (!_enableVerboseLogs) return;
    final ts = DateTime.now().toIso8601String();
    debugPrint('[RackValidation][$ts] $message');
  }

  void _logPolygonStats({required String context}) {
    if (!_enableVerboseLogs) return;

    if (_polygonPoints.isEmpty) {
      _log('$context | Polygon: empty');
      return;
    }

    double minX = _polygonPoints.first.x;
    double maxX = _polygonPoints.first.x;
    double minY = _polygonPoints.first.y;
    double maxY = _polygonPoints.first.y;

    for (final p in _polygonPoints) {
      minX = math.min(minX, p.x);
      maxX = math.max(maxX, p.x);
      minY = math.min(minY, p.y);
      maxY = math.max(maxY, p.y);
    }

    final bboxWidth = maxX - minX;
    final bboxHeight = maxY - minY;

    double area = 0.0;
    double perimeter = 0.0;
    if (_polygonPoints.length >= 3) {
      for (int i = 0; i < _polygonPoints.length; i++) {
        final current = _polygonPoints[i];
        final next = _polygonPoints[(i + 1) % _polygonPoints.length];
        area += (current.x * next.y) - (next.x * current.y);

        final dx = next.x - current.x;
        final dy = next.y - current.y;
        perimeter += math.sqrt((dx * dx) + (dy * dy));
      }
      area = area.abs() / 2.0;
    }

    final pointsStr = _polygonPoints
        .map((p) => '(${p.x.toStringAsFixed(1)}, ${p.y.toStringAsFixed(1)})')
        .join(', ');

    _log(
      '$context | Polygon points=${_polygonPoints.length} | '
      'bbox[minX=${minX.toStringAsFixed(1)}, minY=${minY.toStringAsFixed(1)}, '
      'maxX=${maxX.toStringAsFixed(1)}, maxY=${maxY.toStringAsFixed(1)}, '
      'width=${bboxWidth.toStringAsFixed(1)}, height=${bboxHeight.toStringAsFixed(1)}] | '
      'area=${area.toStringAsFixed(1)} | perimeter=${perimeter.toStringAsFixed(1)} | '
      'points=[$pointsStr]',
    );
  }

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeBridges();
  }

  Future<void> _initializeCamera() async {
    _log(
      'Initializing camera: name=${widget.camera.name}, '
      'lensDirection=${widget.camera.lensDirection}, '
      'sensorOrientation=${widget.camera.sensorOrientation}',
    );

    _cameraController = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _cameraController!.initialize();

      _log(
        'Camera initialized: previewSize=${_cameraController!.value.previewSize}, '
        'isStreamingImages=${_cameraController!.value.isStreamingImages}',
      );

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });

        // Start image stream
        _log('Starting camera image stream...');
        _cameraController!.startImageStream(_processCameraImage);
        _log('Camera image stream started.');
      }
    } catch (e) {
      _log('Error initializing camera: $e');
    }
  }

  Future<String> _loadModelFromAssets(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final bytes = byteData.buffer.asUint8List();
    final tempFile = File('${Directory.systemTemp.path}/best.onnx');
    await tempFile.writeAsBytes(bytes, flush: true);
    return tempFile.path;
  }

  Future<void> _initializeBridges() async {
    _log('Initializing inference and polygon validator bridges...');
    _inferenceBridge = InferenceBridge();
    _polygonValidator = PolygonValidatorBridge();

    // Load model from assets
    try {
      final modelPath = await _loadModelFromAssets('assets/models/model.onnx');
      _log('Model copied to temp path: $modelPath');
      final success = _inferenceBridge.loadModel(modelPath);

      if (success) {
        _log('Model loaded successfully');
        // Set frame skip for better performance
        _inferenceBridge.setFrameSkip(1, 3); // Process every 3rd frame
        _log('Frame skip configured: mode=1 (every second), skipCount=3');
      } else {
        _log('Failed to load model');
        // _showError('Failed to load detection model');
      }
    } catch (e) {
      _log('Error loading model: $e');
      // _showError('Error: $e');
    }

    _polygonValidator.setValidationThreshold(_validationThreshold);
    _log(
      'Validation threshold set to ${_validationThreshold.toStringAsFixed(2)}',
    );
  }

  void _processCameraImage(CameraImage image) {
    _frameCounter++;
    _log(
      'Frame #$_frameCounter received: '
      'size=${image.width}x${image.height}, '
      'format=${image.format.group}, planes=${image.planes.length}',
    );

    // Skip if already processing or no polygon set
    if (_isProcessing) {
      _log('Frame #$_frameCounter skipped: processing in progress');
      return;
    }

    if (!_polygonValidator.hasPolygon()) {
      _log('Frame #$_frameCounter skipped: no polygon defined');
      return;
    }

    _isProcessing = true;

    try {
      // Check if should process this frame (frame skipping)
      if (!_inferenceBridge.shouldProcessFrame()) {
        _log('Frame #$_frameCounter skipped by frame-skip policy');
        return;
      }

      // Extract YUV planes
      final yPlane = image.planes[0].bytes;
      final uPlane = image.planes[1].bytes;
      final vPlane = image.planes[2].bytes;

      _log(
        'Frame #$_frameCounter planes: '
        'y(len=${yPlane.length}, rowStride=${image.planes[0].bytesPerRow}), '
        'u(len=${uPlane.length}, rowStride=${image.planes[1].bytesPerRow}, pixelStride=${image.planes[1].bytesPerPixel}), '
        'v(len=${vPlane.length}, rowStride=${image.planes[2].bytesPerRow}, pixelStride=${image.planes[2].bytesPerPixel})',
      );

      // Run inference
      final detections = _inferenceBridge.runInference(
        yPlane,
        uPlane,
        vPlane,
        image.planes[0].bytesPerRow,
        image.planes[1].bytesPerRow,
        image.planes[1].bytesPerPixel ?? 1,
        image.width,
        image.height,
      );

      _log(
        'Frame #$_frameCounter inference results: count=${detections.length}',
      );

      for (int i = 0; i < detections.length; i++) {
        final d = detections[i];
        _log(
          'Frame #$_frameCounter detection[$i]: '
          'classId=${d.classId}, confidence=${d.confidence.toStringAsFixed(4)}, '
          'center=(${d.x.toStringAsFixed(4)}, ${d.y.toStringAsFixed(4)}), '
          'size=(${d.w.toStringAsFixed(4)}, ${d.h.toStringAsFixed(4)})',
        );
      }

      DetectionResult? personDetection;
      for (final detection in detections) {
        if (detection.classId == _personClassId) {
          personDetection = detection;
          break;
        }
      }

      if (personDetection != null) {
        final detection = personDetection;

        _log(
          'Frame #$_frameCounter selected person detection for validation: '
          'classId=${detection.classId}, confidence=${detection.confidence.toStringAsFixed(4)}',
        );

        // Scale coordinates to screen size
        final screenSize = MediaQuery.of(context).size;
        final scaleX = screenSize.width / image.width;
        final scaleY = screenSize.height / image.height;

        _log(
          'Frame #$_frameCounter scaling: '
          'screen=${screenSize.width.toStringAsFixed(1)}x${screenSize.height.toStringAsFixed(1)}, '
          'scaleX=${scaleX.toStringAsFixed(4)}, scaleY=${scaleY.toStringAsFixed(4)}',
        );

        _logPolygonStats(context: 'Frame #$_frameCounter');

        final result = _polygonValidator.validateBoundingBox(
          x: detection.x * scaleX,
          y: detection.y * scaleY,
          width: detection.w * scaleX,
          height: detection.h * scaleY,
        );

        _log(
          'Frame #$_frameCounter validation result: '
          'status=${result.status.name}, '
          'overlap=${(result.overlapPercentage * 100).toStringAsFixed(2)}%',
        );

        // Update UI only if status changed
        if (result.status != _lastValidationStatus) {
          if (mounted) {
            setState(() {
              _lastValidationStatus = result.status;
            });
          }
          _log(
            'Frame #$_frameCounter UI status changed to ${result.status.name}',
          );
        }

        _handleValidationResult(detection, result);
      } else {
        _log(
          'Frame #$_frameCounter no person detection found, skipping polygon validation',
        );
      }
    } catch (e) {
      _log('Error processing frame #$_frameCounter: $e');
    } finally {
      _isProcessing = false;
      _log('Frame #$_frameCounter processing complete');
    }
  }

  void _handleValidationResult(
    DetectionResult detection,
    PolygonValidationResult result,
  ) {
    if (detection.classId != _personClassId) {
      _log(
        'Ignoring non-person detection: classId=${detection.classId}, confidence=${detection.confidence.toStringAsFixed(4)}',
      );
      return;
    }

    switch (result.status) {
      case ValidationStatus.fullyInside:
        _log(
          'PERSON MATCH 100%: classId=${detection.classId}, '
          'confidence=${detection.confidence.toStringAsFixed(4)}',
        );
        break;

      case ValidationStatus.partiallyInside:
        _log(
          'PERSON NOT FULLY INSIDE: classId=${detection.classId}, '
          'confidence=${detection.confidence.toStringAsFixed(4)}',
        );
        break;

      case ValidationStatus.outside:
        _log(
          'PERSON OUTSIDE POLYGON: classId=${detection.classId}, '
          'confidence=${detection.confidence.toStringAsFixed(4)}',
        );
        break;
    }
  }

  void _addPolygonPoint(Offset position) {
    _log(
      'Adding polygon point: x=${position.dx.toStringAsFixed(1)}, '
      'y=${position.dy.toStringAsFixed(1)}',
    );

    // Use local list to avoid multiple setState calls
    final updatedPoints = List<Point2D>.from(_polygonPoints)
      ..add(Point2D(position.dx, position.dy));

    setState(() {
      _polygonPoints = updatedPoints;
    });

    _logPolygonStats(context: 'Polygon update (point added)');
  }

  void _completePolygon() {
    if (_polygonPoints.length >= 3) {
      _polygonValidator.setPolygon(_polygonPoints);
      setState(() {
        _isDrawingPolygon = false;
      });
      _log('Polygon completed and pushed to native validator');
      _logPolygonStats(context: 'Polygon completed');
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Polygon needs at least 3 points'),
          duration: Duration(seconds: 2),
        ),
      );
      _log('Polygon completion failed: points=${_polygonPoints.length} (<3)');
    }
  }

  void _clearPolygon() {
    _log('Clearing polygon and resetting validation state');
    _polygonValidator.clearPolygon();
    setState(() {
      _polygonPoints.clear();
      _isDrawingPolygon = false;
      _lastValidationStatus = null;
    });
    _log('Polygon cleared');
  }

  void _updateThreshold(double value) {
    setState(() {
      _validationThreshold = value;
    });
    _polygonValidator.setValidationThreshold(value);
    _log(
      'Validation threshold updated to ${(value * 100).toStringAsFixed(0)}%',
    );
  }

  @override
  void dispose() {
    _log('Disposing resources...');
    _cameraController?.dispose();
    _inferenceBridge.dispose();
    _polygonValidator.dispose();
    _log('Resources disposed');
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Rack Validation'),
        backgroundColor: Colors.black87,
        actions: [
          if (_isDrawingPolygon)
            IconButton(
              icon: const Icon(Icons.check),
              onPressed: _completePolygon,
              tooltip: 'Complete Polygon',
            ),
          IconButton(
            icon: const Icon(Icons.clear),
            onPressed: _clearPolygon,
            tooltip: 'Clear Polygon',
          ),
        ],
      ),
      body: Column(
        children: [
          // Validation status indicator
          _buildStatusIndicator(),

          // Threshold slider
          _buildThresholdSlider(),

          // Camera preview with polygon overlay
          Expanded(child: _buildCameraPreview()),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            _isDrawingPolygon = true;
            _polygonPoints.clear();
          });
        },
        backgroundColor: _isDrawingPolygon ? Colors.orange : Colors.blue,
        child: Icon(_isDrawingPolygon ? Icons.edit : Icons.edit_outlined),
        tooltip: 'Draw Polygon',
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isCameraInitialized || _cameraController == null) {
      return Container(
        color: Colors.black,
        child: const Center(
          child: CircularProgressIndicator(color: Colors.white),
        ),
      );
    }

    return GestureDetector(
      onTapDown: (details) {
        if (_isDrawingPolygon) {
          _addPolygonPoint(details.localPosition);
        }
      },
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview
          CameraPreview(_cameraController!),

          // Polygon overlay
          if (_polygonPoints.isNotEmpty)
            CustomPaint(
              painter: PolygonPainter(
                points: _polygonPoints,
                isComplete: !_isDrawingPolygon,
              ),
              size: Size.infinite,
            ),

          // Drawing instructions
          if (_isDrawingPolygon)
            Positioned(
              top: 16,
              left: 0,
              right: 0,
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 16),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Tap to add points (${_polygonPoints.length} points)',
                  style: const TextStyle(color: Colors.white, fontSize: 14),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildStatusIndicator() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      color: _getStatusColor(),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(_getStatusIcon(), color: Colors.white, size: 20),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              _getStatusText(),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
              overflow: TextOverflow.ellipsis,
              maxLines: 1,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildThresholdSlider() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: Colors.grey[100],
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Validation Threshold',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
              ),
              Text(
                '${(_validationThreshold * 100).toInt()}%',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                  color: Colors.blue[700],
                ),
              ),
            ],
          ),
          Slider(
            value: _validationThreshold,
            min: 0.0,
            max: 1.0,
            divisions: 10,
            label: '${(_validationThreshold * 100).toInt()}%',
            onChanged: _updateThreshold,
            activeColor: Colors.blue[700],
          ),
        ],
      ),
    );
  }

  Color _getStatusColor() {
    switch (_lastValidationStatus) {
      case ValidationStatus.fullyInside:
        return Colors.green[700]!;
      case ValidationStatus.partiallyInside:
        return Colors.orange[700]!;
      case ValidationStatus.outside:
        return Colors.red[700]!;
      case null:
        return Colors.grey[700]!;
    }
  }

  IconData _getStatusIcon() {
    switch (_lastValidationStatus) {
      case ValidationStatus.fullyInside:
        return Icons.check_circle;
      case ValidationStatus.partiallyInside:
        return Icons.warning;
      case ValidationStatus.outside:
        return Icons.cancel;
      case null:
        return Icons.help_outline;
    }
  }

  String _getStatusText() {
    if (!_polygonValidator.hasPolygon()) {
      return 'Draw polygon to start';
    }

    switch (_lastValidationStatus) {
      case ValidationStatus.fullyInside:
        return 'VALID - Inside polygon';
      case ValidationStatus.partiallyInside:
        return 'PARTIAL - Partially inside';
      case ValidationStatus.outside:
        return 'INVALID - Outside polygon';
      case null:
        return 'Waiting for detection...';
    }
  }
}

/// Optimized custom painter for drawing polygons
class PolygonPainter extends CustomPainter {
  final List<Point2D> points;
  final bool isComplete;

  PolygonPainter({required this.points, required this.isComplete});

  @override
  void paint(Canvas canvas, Size size) {
    if (points.isEmpty) return;

    final color = isComplete ? Colors.green : Colors.blue;

    final strokePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final fillPaint = Paint()
      ..color = color.withOpacity(0.2)
      ..style = PaintingStyle.fill;

    final path = Path();
    path.moveTo(points.first.x, points.first.y);

    for (int i = 1; i < points.length; i++) {
      path.lineTo(points[i].x, points[i].y);
    }

    if (isComplete) {
      path.close();
      // Draw filled polygon first
      canvas.drawPath(path, fillPaint);
    }

    // Draw outline
    canvas.drawPath(path, strokePaint);

    // Draw points
    final pointPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;

    final pointBorderPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (final point in points) {
      final center = Offset(point.x, point.y);
      // White center
      canvas.drawCircle(center, 6.0, pointPaint);
      // Colored border
      canvas.drawCircle(center, 6.0, pointBorderPaint);
    }
  }

  @override
  bool shouldRepaint(PolygonPainter oldDelegate) {
    return oldDelegate.points.length != points.length ||
        oldDelegate.isComplete != isComplete;
  }
}
