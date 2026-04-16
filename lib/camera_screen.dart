import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:object_detection_app/services/ffi_bridge.dart';
import 'package:object_detection_app/main.dart';
import 'package:object_detection_app/widgets/bbox_painter.dart';
import 'package:path_provider/path_provider.dart';

enum FrameSkipMode {
  noSkip('No Skip', 0),
  everySecond('Every 2nd Frame', 1),
  skipN('Skip N Consecutive', 2);

  const FrameSkipMode(this.label, this.value);
  final String label;
  final int value;
}

class ObjectDetectionScreen extends StatefulWidget {
  const ObjectDetectionScreen({super.key});

  @override
  State<ObjectDetectionScreen> createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  CameraController? _controller;
  bool _isDetecting = false;
  bool _isModelLoaded = false;
  List<Map<String, dynamic>> _results = [];
  List<Map<String, dynamic>> _lastValidResults =
      []; // Store last inference results
  List<String> _labels = [];
  final InferenceBridge _bridge = InferenceBridge();
  Size _imageSize = Size.zero;
  DateTime? _lastInferenceTime;
  double _fps = 0.0;

  // Frame skip settings
  FrameSkipMode _selectedSkipMode = FrameSkipMode.noSkip;
  int _skipCount = 1;

  @override
  void initState() {
    super.initState();
    _initModel();
    if (cameras.isNotEmpty) {
      _initCamera(cameras[0]);
    }
  }

  Future<void> _initModel() async {
    try {
      // Load model from assets to local file
      const modelName = 'model.onnx';
      final byteData = await rootBundle.load('assets/models/$modelName');
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/$modelName');
      await file.writeAsBytes(byteData.buffer.asUint8List());

      bool loaded = _bridge.loadModel(file.path);

      // Try loading labels (optional)
      try {
        final labelsData = await rootBundle.loadString(
          'assets/models/labels.txt',
        );
        _labels = labelsData
            .split('\n')
            .map((e) => e.trim())
            .where((e) => e.isNotEmpty)
            .toList();
        debugPrint("Loaded ${_labels.length} labels successfully.");
      } catch (e) {
        debugPrint("No labels.txt found. Falling back to numeric Class IDs.");
      }

      if (loaded) {
        setState(() {
          _isModelLoaded = true;
        });
        debugPrint("ONNX Model loaded successfully!");
      } else {
        debugPrint("Failed to load ONNX Model!");
      }
    } catch (e) {
      debugPrint("Error loading model: $e");
    }
  }

  void _initCamera(CameraDescription description) async {
    _controller = CameraController(
      description,
      ResolutionPreset.medium, // keep low/medium for faster inference
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() {});

      _controller!.startImageStream((CameraImage image) {
        if (_isDetecting || !_isModelLoaded) return;
        _isDetecting = true;
        _runInference(image);
      });
    } catch (e) {
      debugPrint("Camera Error: $e");
    }
  }

  Future<void> _runInference(CameraImage image) async {
    if (image.planes.length != 3) {
      _isDetecting = false;
      return;
    }

    try {
      final yPlane = image.planes[0].bytes;
      final uPlane = image.planes[1].bytes;
      final vPlane = image.planes[2].bytes;

      // Check if we should process this frame (C++ decides)
      final shouldProcess = _bridge.shouldProcessFrame();

      List<Map<String, dynamic>> mappedResults;

      if (shouldProcess) {
        // Process this frame with the model
        final results = _bridge.runInference(
          yPlane,
          uPlane,
          vPlane,
          image.planes[0].bytesPerRow,
          image.planes[1].bytesPerRow,
          image.planes[1].bytesPerPixel ?? 1,
          image.width,
          image.height,
        );

        if (!mounted) return;

        final bool isRotatedToPortrait = image.width > image.height;
        final int logicalImageWidth = isRotatedToPortrait
            ? image.height
            : image.width;
        final int logicalImageHeight = isRotatedToPortrait
            ? image.width
            : image.height;

        mappedResults = results.map((result) {
          final className = result.classId < _labels.length
              ? _labels[result.classId]
              : result.classId.toString();

          final centerX = result.x * logicalImageWidth;
          final centerY = result.y * logicalImageHeight;
          final boxWidth = result.w * logicalImageWidth;
          final boxHeight = result.h * logicalImageHeight;

          final x1 = (centerX - (boxWidth / 2)).clamp(
            0.0,
            logicalImageWidth.toDouble(),
          );
          final y1 = (centerY - (boxHeight / 2)).clamp(
            0.0,
            logicalImageHeight.toDouble(),
          );
          final x2 = (centerX + (boxWidth / 2)).clamp(
            0.0,
            logicalImageWidth.toDouble(),
          );
          final y2 = (centerY + (boxHeight / 2)).clamp(
            0.0,
            logicalImageHeight.toDouble(),
          );

          return <String, dynamic>{
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'class': className,
            'confidence': result.confidence,
          };
        }).toList();

        // Store the new results for reuse
        _lastValidResults = mappedResults;

        if (results.isNotEmpty) {
          final detectedNames = results
              .map((r) {
                String cName = r.classId.toString();
                if (r.classId < _labels.length) cName = _labels[r.classId];
                return '$cName(${(r.confidence * 100).toStringAsFixed(1)}%)';
              })
              .join(', ');
          debugPrint('Detected: $detectedNames');
        }
      } else {
        // Skip this frame - reuse last results
        mappedResults = _lastValidResults;
      }

      setState(() {
        _results = mappedResults;

        final bool isRotatedToPortrait = image.width > image.height;
        final int logicalImageWidth = isRotatedToPortrait
            ? image.height
            : image.width;
        final int logicalImageHeight = isRotatedToPortrait
            ? image.width
            : image.height;

        _imageSize = Size(
          logicalImageWidth.toDouble(),
          logicalImageHeight.toDouble(),
        );

        // Update FPS only when we actually process
        if (shouldProcess) {
          final now = DateTime.now();
          if (_lastInferenceTime != null) {
            final diff = now.difference(_lastInferenceTime!).inMilliseconds;
            if (diff > 0) {
              _fps = 1000 / diff;
            }
          }
          _lastInferenceTime = now;
        }
      });
    } catch (e) {
      debugPrint("Inference Error: $e");
    } finally {
      await Future.delayed(const Duration(milliseconds: 50));
      _isDetecting = false;
    }
  }

  Future<void> _showSkipCountDialog() async {
    final TextEditingController controller = TextEditingController(
      text: _skipCount.toString(),
    );

    final result = await showDialog<int>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Skip N Consecutive Frames'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Enter the number of frames to skip:'),
            const SizedBox(height: 16),
            TextField(
              controller: controller,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'N (1-10)',
                border: OutlineInputBorder(),
              ),
              inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final value = int.tryParse(controller.text) ?? 1;
              Navigator.pop(context, value.clamp(1, 10));
            },
            child: const Text('Apply'),
          ),
        ],
      ),
    );

    if (result != null) {
      setState(() {
        _skipCount = result;
      });
      _bridge.setFrameSkip(_selectedSkipMode.value, _skipCount);
    }
  }

  void _onSkipModeChanged(FrameSkipMode? mode) async {
    if (mode == null) return;

    setState(() {
      _selectedSkipMode = mode;
    });

    if (mode == FrameSkipMode.skipN) {
      // Show dialog to get N value
      await _showSkipCountDialog();
    } else {
      // Apply immediately for other modes
      _bridge.setFrameSkip(mode.value, _skipCount);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _bridge.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final size = MediaQuery.of(context).size;

    // Calculate the perfect aspect ratio to match the camera feed to the screen
    bool isPortrait =
        MediaQuery.of(context).orientation == Orientation.portrait;
    double cameraAspectRatio = _controller!.value.aspectRatio;
    double displayAspectRatio = isPortrait
        ? (1 / cameraAspectRatio)
        : cameraAspectRatio;

    return Scaffold(
      appBar: AppBar(
        title: const Text('ONNX Realtime Detection'),
        actions: [
          // Frame skip dropdown in AppBar
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
            child: DropdownButton<FrameSkipMode>(
              value: _selectedSkipMode,
              dropdownColor: Colors.grey[900],
              style: const TextStyle(color: Colors.white),
              underline: Container(height: 2, color: Colors.white),
              items: FrameSkipMode.values.map((mode) {
                return DropdownMenuItem(
                  value: mode,
                  child: Text(
                    mode == FrameSkipMode.skipN && _skipCount > 1
                        ? '${mode.label} ($_skipCount)'
                        : mode.label,
                  ),
                );
              }).toList(),
              onChanged: _onSkipModeChanged,
            ),
          ),
        ],
      ),
      backgroundColor: Colors.black, // Dark background for letterboxed areas
      body: Stack(
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: displayAspectRatio,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(_controller!),
                  if (_imageSize.width > 0)
                    CustomPaint(
                      painter: BoundingBoxPainter(
                        _results,
                        _imageSize.width.toInt(),
                        _imageSize.height.toInt(),
                        size.width,
                        size.height,
                      ),
                    ),
                ],
              ),
            ),
          ),
          if (!_isModelLoaded)
            const Center(
              child: Text(
                'Please put your ONNX model at assets/models/model.onnx',
                style: TextStyle(
                  color: Colors.red,
                  backgroundColor: Colors.white,
                  fontSize: 18,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          if (_isModelLoaded)
            Positioned(
              top: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 5,
                ),
                color: Colors.black54,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      '${_fps.toStringAsFixed(1)} FPS',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      _selectedSkipMode == FrameSkipMode.skipN
                          ? 'Skip: ${_skipCount}f'
                          : _selectedSkipMode.label,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
