import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:object_detection_app/services/inference_bridge.dart';
import 'package:object_detection_app/services/cv_bridge.dart';
import 'package:object_detection_app/services/polygon_validator_bridge.dart';
import 'package:object_detection_app/widgets/polygon_painter.dart';

class ValidationCaptureScreen extends StatefulWidget {
  final CameraDescription camera;
  final String polygonName;
  final List<Point2D> points;

  const ValidationCaptureScreen({
    Key? key,
    required this.camera,
    required this.polygonName,
    required this.points,
  }) : super(key: key);

  @override
  State<ValidationCaptureScreen> createState() =>
      _ValidationCaptureScreenState();
}

class _ValidationCaptureScreenState extends State<ValidationCaptureScreen>
    with SingleTickerProviderStateMixin {
  CameraController? _cameraController;
  late InferenceBridge _inferenceBridge;
  late CvBridge _cvBridge;
  late PolygonValidatorBridge _polygonValidator;
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  bool _isProcessing = false;
  ValidationStatus? _status;
  static const int _rackClassId = 0;
  bool _useMl = true; // toggle: true=ML, false=CV

  @override
  void initState() {
    super.initState();
    _inferenceBridge = InferenceBridge();
    _cvBridge = CvBridge();
    _polygonValidator = PolygonValidatorBridge();

    _polygonValidator.setPolygon(widget.points);
    _polygonValidator.setValidationThreshold(1.0);

    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);

    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.1).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _initModelAndCamera();
  }

  Future<void> _initModelAndCamera() async {
    final byteData = await rootBundle.load('assets/models/model.onnx');
    final bytes = byteData.buffer.asUint8List();
    final tempFile = File('${Directory.systemTemp.path}/best.onnx');
    await tempFile.writeAsBytes(bytes, flush: true);
    _inferenceBridge.loadModel(tempFile.path);

    _cameraController = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await _cameraController!.initialize();

    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _cameraController?.dispose();
    _inferenceBridge.dispose();
    _cvBridge.dispose();
    _polygonValidator.dispose();
    super.dispose();
  }

  Future<void> _captureAndValidate() async {
    if (_isProcessing || _cameraController == null) return;

    setState(() {
      _isProcessing = true;
      _status = null;
    });

    try {
      _cameraController!.startImageStream((CameraImage camImage) {
        _cameraController!.stopImageStream();
        _processFrame(camImage);
      });
    } catch (e) {
      setState(() => _isProcessing = false);
      debugPrint('Error: $e');
    }
  }

  void _processFrame(CameraImage image) {
    try {
      final yPlane = image.planes[0].bytes;
      final uPlane = image.planes[1].bytes;
      final vPlane = image.planes[2].bytes;

      final detections = _useMl
          ? _inferenceBridge.runInference(
              yPlane,
              uPlane,
              vPlane,
              image.planes[0].bytesPerRow,
              image.planes[1].bytesPerRow,
              image.planes[1].bytesPerPixel ?? 1,
              image.width,
              image.height,
            )
          : _cvBridge.runSync(
              yPlane,
              uPlane,
              vPlane,
              image.planes[0].bytesPerRow,
              image.planes[1].bytesPerRow,
              image.planes[1].bytesPerPixel ?? 1,
              image.width,
              image.height,
            );

      DetectionResult? bestDetection;
      for (final detection in detections) {
        if (detection.classId == _rackClassId) {
          bestDetection = detection;
          break;
        }
      }

      if (bestDetection != null) {
        final screenSize = MediaQuery.of(context).size;
        final scaleX = screenSize.width / image.width;
        final scaleY = screenSize.height / image.height;

        final result = _polygonValidator.validateBoundingBox(
          x: bestDetection.x * scaleX,
          y: bestDetection.y * scaleY,
          width: bestDetection.w * scaleX,
          height: bestDetection.h * scaleY,
        );

        setState(() {
          _status = result.status;
        });
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Row(
                children: [
                  Icon(Icons.warning, color: Colors.white),
                  SizedBox(width: 12),
                  Text('No rack detected in frame'),
                ],
              ),
              backgroundColor: Colors.orange,
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          );
        }
      }
    } catch (e) {
      debugPrint("Frame processing error: $e");
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.black.withOpacity(0.3),
        elevation: 0,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Validate Rack',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                shadows: [
                  Shadow(
                    offset: Offset(0, 1),
                    blurRadius: 8.0,
                    color: Colors.black45,
                  ),
                ],
              ),
            ),
            Text(
              widget.polygonName,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.normal,
                shadows: [
                  Shadow(
                    offset: Offset(0, 1),
                    blurRadius: 8.0,
                    color: Colors.black45,
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          Row(
            children: [
              const Text('ML', style: TextStyle(fontSize: 12)),
              Switch(
                value: _useMl,
                onChanged: (v) {
                  setState(() {
                    _useMl = v;
                  });
                },
              ),
              const Text('CV', style: TextStyle(fontSize: 12)),
              const SizedBox(width: 8),
            ],
          ),
        ],
        iconTheme: const IconThemeData(
          shadows: [
            Shadow(
              offset: Offset(0, 1),
              blurRadius: 8.0,
              color: Colors.black45,
            ),
          ],
        ),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview
          if (_cameraController != null &&
              _cameraController!.value.isInitialized)
            CameraPreview(_cameraController!)
          else
            Container(
              color: Colors.black,
              child: const Center(
                child: CircularProgressIndicator(color: Colors.white),
              ),
            ),

          // Polygon Overlay
          CustomPaint(
            painter: PolygonPainter(points: widget.points, isComplete: true),
          ),

          // Validation Result Banner
          if (_status != null)
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: SafeArea(
                bottom: false,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  margin: const EdgeInsets.all(16),
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: _status == ValidationStatus.fullyInside
                          ? [Colors.green[700]!, Colors.green[500]!]
                          : _status == ValidationStatus.partiallyInside
                          ? [Colors.orange[700]!, Colors.orange[500]!]
                          : [Colors.red[700]!, Colors.red[500]!],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color:
                            (_status == ValidationStatus.fullyInside
                                    ? Colors.green
                                    : _status ==
                                          ValidationStatus.partiallyInside
                                    ? Colors.orange
                                    : Colors.red)
                                .withOpacity(0.4),
                        blurRadius: 20,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      Icon(
                        _status == ValidationStatus.fullyInside
                            ? Icons.check_circle
                            : _status == ValidationStatus.partiallyInside
                            ? Icons.warning
                            : Icons.cancel,
                        color: Colors.white,
                        size: 48,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        _status == ValidationStatus.fullyInside
                            ? 'RACK INSIDE POLYGON'
                            : (_status == ValidationStatus.partiallyInside
                                  ? 'RACK PARTIALLY INSIDE'
                                  : 'RACK OUTSIDE POLYGON'),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.2,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _status == ValidationStatus.fullyInside
                            ? 'Position verified ✓'
                            : 'Adjust rack position',
                        style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 14,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
            ),

          // Instructions
          if (_status == null && !_isProcessing)
            Positioned(
              top: 100,
              left: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.white24),
                ),
                child: const Column(
                  children: [
                    Icon(
                      Icons.center_focus_strong,
                      color: Colors.white,
                      size: 32,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Position rack within the polygon',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 4),
                    Text(
                      'Tap capture when ready',
                      style: TextStyle(color: Colors.white70, fontSize: 14),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),

          // Capture Button
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.transparent, Colors.black.withOpacity(0.8)],
                ),
              ),
              child: SafeArea(
                top: false,
                child: ScaleTransition(
                  scale: _isProcessing
                      ? _pulseAnimation
                      : const AlwaysStoppedAnimation(1.0),
                  child: SizedBox(
                    width: double.infinity,
                    height: 65,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _isProcessing
                            ? Colors.grey
                            : Theme.of(context).primaryColor,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        elevation: 8,
                        shadowColor: Theme.of(
                          context,
                        ).primaryColor.withOpacity(0.5),
                      ),
                      onPressed: _isProcessing ? null : _captureAndValidate,
                      child: _isProcessing
                          ? const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                SizedBox(
                                  width: 24,
                                  height: 24,
                                  child: CircularProgressIndicator(
                                    color: Colors.white,
                                    strokeWidth: 3,
                                  ),
                                ),
                                SizedBox(width: 16),
                                Text(
                                  'Processing...',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            )
                          : const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.camera_alt, size: 28),
                                SizedBox(width: 12),
                                Text(
                                  'Capture & Validate',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
