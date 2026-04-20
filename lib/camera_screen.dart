import 'dart:io';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:object_detection_app/services/inference_bridge.dart';
import 'package:object_detection_app/main.dart';
import 'package:path_provider/path_provider.dart';

enum FrameSkipMode {
  noSkip('No Skip', 0, Icons.fast_forward_outlined),
  everySecond('Every 2nd Frame', 1, Icons.filter_2),
  skipN('Skip N Consecutive', 2, Icons.tune);

  const FrameSkipMode(this.label, this.value, this.icon);
  final String label;
  final int value;
  final IconData icon;
}

class ObjectDetectionScreen extends StatefulWidget {
  const ObjectDetectionScreen({super.key});

  @override
  State<ObjectDetectionScreen> createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen>
    with TickerProviderStateMixin {
  CameraController? _controller;
  bool _isModelLoaded = false;
  List<Map<String, dynamic>> _results = [];
  List<String> _labels = [];
  final InferenceBridge _bridge = InferenceBridge();
  Size _imageSize = Size.zero;
  double _fps = 0.0;
  double _inferenceFps = 0.0;
  int _frameCount = 0;
  DateTime? _fpsTimer;

  // Producer-Consumer stats
  int _framesPushed = 0;
  int _framesDropped = 0;
  int _queueSize = 2; // Recommended: 2-3 for smooth operation

  // Frame skip settings
  FrameSkipMode _selectedSkipMode = FrameSkipMode.noSkip;
  int _skipCount = 1;

  // Animation controllers
  late AnimationController _pulseController;
  late AnimationController _statsController;
  late Animation<double> _pulseAnimation;
  late Animation<double> _statsAnimation;

  bool _showStats = true;
  bool _showSettings = false;

  @override
  void initState() {
    super.initState();

    // Initialize animations
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);

    _statsController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _pulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _statsAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(parent: _statsController, curve: Curves.easeOut));

    _statsController.forward();

    _initModel();
    if (cameras.isNotEmpty) {
      _initCamera(cameras[0]);
    }
  }

  Future<void> _initModel() async {
    try {
      const modelName = 'model.onnx';
      final byteData = await rootBundle.load('assets/models/$modelName');
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/$modelName');
      await file.writeAsBytes(byteData.buffer.asUint8List());

      bool loaded = _bridge.loadModel(file.path);

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

        // Start the background inference worker with Producer-Consumer pattern
        _bridge.startInferenceWorker(maxQueueSize: _queueSize);
        debugPrint("ONNX Model loaded successfully!");
        debugPrint(
          "Background inference worker started with queue size: $_queueSize",
        );

        // Start periodic UI updates at 60 FPS
        _startResultsPolling();
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
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() {});

      // Start camera stream - this is the PRODUCER
      // Frames are pushed to the queue without blocking
      _controller!.startImageStream(_onCameraFrame);

      debugPrint("Camera initialized and streaming started");
    } catch (e) {
      debugPrint("Camera Error: $e");
    }
  }

  /// PRODUCER: Called by camera for each frame
  /// This runs on the camera thread and must be VERY fast
  void _onCameraFrame(CameraImage image) {
    if (!_isModelLoaded || !_bridge.isWorkerRunning) {
      return;
    }

    if (image.planes.length != 3) {
      return;
    }
    // Update image size for bbox rendering  
    if (_imageSize.width <= 0) {
      setState(() {
        _imageSize = Size(image.width.toDouble(), image.height.toDouble());
      });
    // Removed extra closing brace
    }

    _frameCount++;

    // Calculate FPS
    _fpsTimer ??= DateTime.now();
    final elapsed = DateTime.now().difference(_fpsTimer!).inMilliseconds;
    if (elapsed >= 1000) {
      setState(() {
        _fps = _frameCount * 1000 / elapsed;
      });
      _frameCount = 0;
      _fpsTimer = DateTime.now();
    }

    // Push frame to queue (non-blocking, very fast)
    final success = _bridge.pushFrame(
      image.planes[0].bytes, // Y plane
      image.planes[1].bytes, // U plane
      image.planes[2].bytes, // V plane
      image.planes[0].bytesPerRow,
      image.planes[1].bytesPerRow,
      image.planes[1].bytesPerPixel ?? 1,
      image.width,
      image.height,
    );

    if (success) {
      _framesPushed++;
    } else {
      _framesDropped++;
      // Frame dropped because queue is full
      // This is OK - inference is slower than camera
      // UI will still be smooth using latest results
    }
  }

  /// CONSUMER: Periodically fetch latest results and update UI
  /// Runs at 60 FPS for smooth UI, independent of inference speed
  void _startResultsPolling() {
    String? lastResultsHash;
    int inferenceCount = 0;
    DateTime? inferenceTimer;

    Stream.periodic(const Duration(milliseconds: 16)).listen((_) {
      if (!mounted || !_isModelLoaded || !_bridge.isWorkerRunning) return;

      // Get latest results (non-blocking, very fast)
      final results = _bridge.getLatestResults();

      if (results != null) {
        // Create a simple hash to detect if results changed (new inference)
        final currentHash =
            '${results.detections.length}_${results.detections.map((d) => '${d.classId}_${d.confidence.toStringAsFixed(3)}').join('_')}';

        // Check if this is a new inference result
        if (currentHash != lastResultsHash) {
          lastResultsHash = currentHash;
          inferenceCount++;

          // Calculate inference FPS
          inferenceTimer ??= DateTime.now();
          final elapsed = DateTime.now()
              .difference(inferenceTimer!)
              .inMilliseconds;
          if (elapsed >= 1000) {
            final fps = inferenceCount * 1000 / elapsed;
            setState(() {
              _inferenceFps = fps;
            });
            inferenceCount = 0;
            inferenceTimer = DateTime.now();
          }
        }

        if (results.detections.isNotEmpty) {
          // Map raw results to UI format
          final mappedResults = results.detections.map((result) {
            final className = result.classId < _labels.length
                ? _labels[result.classId]
                : result.classId.toString();

            // Use current image size or calculate from detection
            final imageWidth = _imageSize.width > 0 ? _imageSize.width : 640.0;
            final imageHeight = _imageSize.height > 0 ? _imageSize.height : 480.0;

            // Convert normalized coordinates to pixel coordinates
            final centerX = result.x * imageWidth;
            final centerY = result.y * imageHeight;
            final boxWidth = result.w * imageWidth;
            final boxHeight = result.h * imageHeight;

            final x1 = (centerX - (boxWidth / 2)).clamp(0.0, imageWidth);
            final y1 = (centerY - (boxHeight / 2)).clamp(0.0, imageHeight);
            final x2 = (centerX + (boxWidth / 2)).clamp(0.0, imageWidth);
            final y2 = (centerY + (boxHeight / 2)).clamp(0.0, imageHeight);

            return <String, dynamic>{
              'x1': x1,
              'y1': y1,
              'x2': x2,
              'y2': y2,
              'class': className,
              'confidence': result.confidence,
            };
          }).toList();

          setState(() {
            _results = mappedResults;
          });

          // Debug output for detections (only on new results)
          if (results.detections.isNotEmpty && currentHash != lastResultsHash) {
            final detectedNames = results.detections
                .map((r) {
                  String cName = r.classId.toString();
                  if (r.classId < _labels.length) cName = _labels[r.classId];
                  return '$cName(${(r.confidence * 100).toStringAsFixed(1)}%)';
                })
                .join(', ');
            debugPrint(
              'Detected: $detectedNames | Inference FPS: ${_inferenceFps.toStringAsFixed(1)}',
            );
          }
        } else {
          // Empty results - clear detections
          setState(() {
            _results = [];
          });
        }
      }
    });
  }

  Future<void> _showSkipCountDialog() async {
    final TextEditingController controller = TextEditingController(
      text: _skipCount.toString(),
    );

    final result = await showDialog<int>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey[900],
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.blue.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.tune, color: Colors.blue),
            ),
            const SizedBox(width: 12),
            const Text(
              'Frame Skip Settings',
              style: TextStyle(color: Colors.white),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Enter the number of frames to skip:',
              style: TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: controller,
              keyboardType: TextInputType.number,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                labelText: 'N (1-10)',
                labelStyle: const TextStyle(color: Colors.blue),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Colors.blue),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Colors.blue.withValues(alpha: 0.5)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Colors.blue, width: 2),
                ),
                filled: true,
                fillColor: Colors.grey[850],
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
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
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

  Future<void> _showQueueSizeDialog() async {
    final TextEditingController controller = TextEditingController(
      text: _queueSize.toString(),
    );

    final result = await showDialog<int>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey[900],
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.purple.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.queue, color: Colors.purple),
            ),
            const SizedBox(width: 12),
            const Text(
              'Queue Size Settings',
              style: TextStyle(color: Colors.white),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Set max queue size (2-3 recommended):',
              style: TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: controller,
              keyboardType: TextInputType.number,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                labelText: 'Queue Size (1-5)',
                labelStyle: const TextStyle(color: Colors.purple),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Colors.purple),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Colors.purple.withValues(alpha: 0.5)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Colors.purple, width: 2),
                ),
                filled: true,
                fillColor: Colors.grey[850],
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
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.purple,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
            onPressed: () {
              final value = int.tryParse(controller.text) ?? 2;
              Navigator.pop(context, value.clamp(1, 5));
            },
            child: const Text('Apply & Restart'),
          ),
        ],
      ),
    );

    if (result != null && result != _queueSize) {
      setState(() {
        _queueSize = result;
      });
      // Restart worker with new queue size
      _bridge.dispose();
      _bridge.startInferenceWorker(maxQueueSize: _queueSize);
      debugPrint("Worker restarted with queue size: $_queueSize");
    }
  }

  void _onSkipModeChanged(FrameSkipMode? mode) async {
    if (mode == null) return;

    setState(() {
      _selectedSkipMode = mode;
    });

    if (mode == FrameSkipMode.skipN) {
      await _showSkipCountDialog();
    } else {
      _bridge.setFrameSkip(mode.value, _skipCount);
    }
  }

  @override
  void dispose() {
    _controller?.stopImageStream();
    _controller?.dispose();
    _bridge.dispose(); // This stops the worker thread
    _pulseController.dispose();
    _statsController.dispose();
    super.dispose();
  }

  Widget _buildGlassmorphicContainer({
    required Widget child,
    double? width,
    double? height,
    EdgeInsets? padding,
  }) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          width: width,
          height: height,
          padding: padding ?? const EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withValues(alpha: 0.15),
                Colors.white.withValues(alpha: 0.05),
              ],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.2),
              width: 1.5,
            ),
          ),
          child: child,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Scaffold(
        backgroundColor: const Color(0xFF0A0E27),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ScaleTransition(
                scale: _pulseAnimation,
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: LinearGradient(
                      colors: [Colors.blue.shade400, Colors.purple.shade400],
                    ),
                  ),
                  child: const Icon(
                    Icons.camera_alt,
                    size: 60,
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(height: 24),
              const CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
              ),
              const SizedBox(height: 16),
              const Text(
                'Initializing Camera...',
                style: TextStyle(color: Colors.white70, fontSize: 16),
              ),
            ],
          ),
        ),
      );
    }

    bool isPortrait =
        MediaQuery.of(context).orientation == Orientation.portrait;
    double cameraAspectRatio = _controller!.value.aspectRatio;
    double displayAspectRatio = isPortrait
        ? (1 / cameraAspectRatio)
        : cameraAspectRatio;

    return Scaffold(
      backgroundColor: const Color(0xFF0A0E27),
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        flexibleSpace: ClipRect(
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Colors.blue.withValues(alpha: 0.1),
                    Colors.purple.withValues(alpha: 0.1),
                  ],
                ),
              ),
            ),
          ),
        ),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.blue.shade400, Colors.purple.shade400],
                ),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.sensors, size: 20),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'ONNX Detection',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                Text(
                  'Producer-Consumer Pattern',
                  style: TextStyle(fontSize: 10, color: Colors.white70),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: Icon(
              _showStats ? Icons.visibility : Icons.visibility_off,
              color: Colors.white,
            ),
            onPressed: () {
              setState(() {
                _showStats = !_showStats;
              });
            },
          ),
          IconButton(
            icon: const Icon(Icons.settings, color: Colors.white),
            onPressed: () {
              setState(() {
                _showSettings = !_showSettings;
              });
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          // Camera Preview
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
                        detections: _results
                            .map(
                              (result) => DetectionResult(
                                x1: result['x1'] ?? 0.0,
                                y1: result['y1'] ?? 0.0,
                                x2: result['x2'] ?? 0.0,
                                y2: result['y2'] ?? 0.0,
                                className: result['class'] ?? 'Unknown',
                                confidence: result['confidence'] ?? 0.0,
                              ),
                            )
                            .toList(),
                        imageSize: _imageSize,
                      ),
                    ),
                ],
              ),
            ),
          ),

          // Model Not Loaded Warning
          if (!_isModelLoaded)
            Center(
              child: _buildGlassmorphicContainer(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.warning_amber_rounded,
                      size: 60,
                      color: Colors.orange.shade400,
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Model Not Found',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Please place your ONNX model at\nassets/models/model.onnx',
                      style: TextStyle(color: Colors.white70, fontSize: 14),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),

          // Stats Display
          if (_isModelLoaded && _showStats)
            Positioned(
              top: 100,
              right: 16,
              child: FadeTransition(
                opacity: _statsAnimation,
                child: _buildGlassmorphicContainer(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      _buildStatRow(
                        icon: Icons.videocam,
                        label: 'Camera',
                        value: _fps.toStringAsFixed(1),
                        color: Colors.green.shade400,
                      ),
                      const SizedBox(height: 8),
                      _buildStatRow(
                        icon: Icons.memory,
                        label: 'Inference',
                        value: _inferenceFps.toStringAsFixed(1),
                        color: Colors.blue.shade400,
                      ),
                      const SizedBox(height: 8),
                      _buildStatRow(
                        icon: Icons.checklist,
                        label: 'Objects',
                        value: _results.length.toString(),
                        color: Colors.purple.shade400,
                      ),
                      const Divider(color: Colors.white24, height: 20),
                      _buildStatRow(
                        icon: Icons.upload,
                        label: 'Pushed',
                        value: _framesPushed.toString(),
                        color: Colors.cyan.shade400,
                      ),
                      const SizedBox(height: 8),
                      _buildStatRow(
                        icon: Icons.block,
                        label: 'Dropped',
                        value: _framesDropped.toString(),
                        color: Colors.orange.shade400,
                      ),
                    ],
                  ),
                ),
              ),
            ),

          // Settings Panel
          if (_showSettings)
            Positioned(
              bottom: 16,
              left: 16,
              right: 16,
              child: _buildGlassmorphicContainer(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.blue.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Icon(
                            Icons.tune,
                            color: Colors.blue,
                            size: 20,
                          ),
                        ),
                        const SizedBox(width: 12),
                        const Text(
                          'Settings',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const Spacer(),
                        IconButton(
                          icon: const Icon(Icons.queue, color: Colors.purple),
                          onPressed: _showQueueSizeDialog,
                          tooltip: 'Queue Size',
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Queue Size: $_queueSize',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.7),
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Frame Skip Mode',
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 12),
                    ...FrameSkipMode.values.map((mode) {
                      final isSelected = _selectedSkipMode == mode;
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: Material(
                          color: Colors.transparent,
                          child: InkWell(
                            borderRadius: BorderRadius.circular(12),
                            onTap: () => _onSkipModeChanged(mode),
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 12,
                              ),
                              decoration: BoxDecoration(
                                color: isSelected
                                    ? Colors.blue.withValues(alpha: 0.3)
                                    : Colors.white.withValues(alpha: 0.05),
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(
                                  color: isSelected
                                      ? Colors.blue
                                      : Colors.white.withValues(alpha: 0.1),
                                  width: 1.5,
                                ),
                              ),
                              child: Row(
                                children: [
                                  Icon(
                                    mode.icon,
                                    color: isSelected
                                        ? Colors.blue
                                        : Colors.white70,
                                    size: 20,
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Text(
                                      mode == FrameSkipMode.skipN &&
                                              _skipCount > 1
                                          ? '${mode.label} ($_skipCount)'
                                          : mode.label,
                                      style: TextStyle(
                                        color: isSelected
                                            ? Colors.white
                                            : Colors.white70,
                                        fontSize: 14,
                                        fontWeight: isSelected
                                            ? FontWeight.w600
                                            : FontWeight.normal,
                                      ),
                                    ),
                                  ),
                                  if (isSelected)
                                    const Icon(
                                      Icons.check_circle,
                                      color: Colors.blue,
                                      size: 20,
                                    ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ],
                ),
              ),
            ),

          // Detection Count Badge
          if (_isModelLoaded && _results.isNotEmpty)
            Positioned(
              top: 100,
              left: 16,
              child: _buildGlassmorphicContainer(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: Colors.green.shade400,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.green.shade400,
                            blurRadius: 8,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${_results.length} detected',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
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

  Widget _buildStatRow({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: color, size: 16),
        const SizedBox(width: 8),
        Text(
          '$label: ',
          style: const TextStyle(color: Colors.white70, fontSize: 12),
        ),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}

/// Simple data class to hold detection results
class DetectionResult {
  final double x1, y1, x2, y2;
  final String className;
  final double confidence;

  DetectionResult({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.className,
    required this.confidence,
  });
}

/// Custom painter to draw bounding boxes on the camera preview
class BoundingBoxPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size imageSize;

  BoundingBoxPainter({required this.detections, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    for (var detection in detections) {
      // Calculate scaling factors
      final scaleX = size.width / imageSize.width;
      final scaleY = size.height / imageSize.height;

      // Draw bounding box
      final rect = Rect.fromLTRB(
        detection.x1 * scaleX,
        detection.y1 * scaleY,
        detection.x2 * scaleX,
        detection.y2 * scaleY,
      );

      canvas.drawRect(
        rect,
        Paint()
          ..color = Colors.green
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0,
      );

      // Draw label background
      const textStyle = TextStyle(
        color: Colors.white,
        fontSize: 12,
        fontWeight: FontWeight.bold,
      );
      final label =
          '${detection.className} ${(detection.confidence * 100).toStringAsFixed(1)}%';
      final textPainter = TextPainter(
        text: TextSpan(text: label, style: textStyle),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final labelBgRect = Rect.fromLTWH(
        rect.left,
        rect.top - 20,
        textPainter.width + 4,
        18,
      );

      canvas.drawRect(
        labelBgRect,
        Paint()..color = Colors.green.withValues(alpha: 0.8),
      );

      // Draw label text
      textPainter.paint(canvas, Offset(rect.left + 2, rect.top - 18));
    }
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
