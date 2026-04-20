import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

// Native function signatures
typedef InitModelNative = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef InitModelDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef FreeModelNative = Void Function(Pointer<Void> context);
typedef FreeModelDart = void Function(Pointer<Void> context);

typedef SetFrameSkipNative =
    Void Function(Pointer<Void> context, Int32 skipMode, Int32 skipCount);
typedef SetFrameSkipDart =
    void Function(Pointer<Void> context, int skipMode, int skipCount);

typedef ShouldProcessFrameNative = Int32 Function(Pointer<Void> context);
typedef ShouldProcessFrameDart = int Function(Pointer<Void> context);

// Producer-Consumer API
typedef StartInferenceWorkerNative =
    Void Function(Pointer<Void> context, Int32 maxQueueSize);
typedef StartInferenceWorkerDart =
    void Function(Pointer<Void> context, int maxQueueSize);

typedef StopInferenceWorkerNative = Void Function(Pointer<Void> context);
typedef StopInferenceWorkerDart = void Function(Pointer<Void> context);

typedef PushFrameToQueueNative =
    Int32 Function(
      Pointer<Void> context,
      Pointer<Uint8> yPlane,
      Pointer<Uint8> uPlane,
      Pointer<Uint8> vPlane,
      Int32 yRowStride,
      Int32 uvRowStride,
      Int32 uvPixelStride,
      Int32 imgWidth,
      Int32 imgHeight,
    );

typedef PushFrameToQueueDart =
    int Function(
      Pointer<Void> context,
      Pointer<Uint8> yPlane,
      Pointer<Uint8> uPlane,
      Pointer<Uint8> vPlane,
      int yRowStride,
      int uvRowStride,
      int uvPixelStride,
      int imgWidth,
      int imgHeight,
    );

typedef GetLatestResultsNative =
    Pointer<Float> Function(
      Pointer<Void> context,
      Pointer<Int32> outCount,
      Pointer<Int64> outTimestamp,
    );

typedef GetLatestResultsDart =
    Pointer<Float> Function(
      Pointer<Void> context,
      Pointer<Int32> outCount,
      Pointer<Int64> outTimestamp,
    );

// Legacy synchronous API
typedef RunInferenceYUVNative =
    Pointer<Float> Function(
      Pointer<Void> context,
      Pointer<Uint8> yPlane,
      Pointer<Uint8> uPlane,
      Pointer<Uint8> vPlane,
      Int32 yRowStride,
      Int32 uvRowStride,
      Int32 uvPixelStride,
      Int32 imgWidth,
      Int32 imgHeight,
      Pointer<Int32> outCount,
    );

typedef RunInferenceYUVDart =
    Pointer<Float> Function(
      Pointer<Void> context,
      Pointer<Uint8> yPlane,
      Pointer<Uint8> uPlane,
      Pointer<Uint8> vPlane,
      int yRowStride,
      int uvRowStride,
      int uvPixelStride,
      int imgWidth,
      int imgHeight,
      Pointer<Int32> outCount,
    );

typedef FreeResultsNative = Void Function(Pointer<Float> results);
typedef FreeResultsDart = void Function(Pointer<Float> results);

class DetectionResult {
  final double x;
  final double y;
  final double w;
  final double h;
  final double confidence;
  final int classId;

  DetectionResult({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.confidence,
    required this.classId,
  });

  @override
  String toString() {
    return 'Detection(class: $classId, conf: ${confidence.toStringAsFixed(2)}, '
        'box: [${x.toStringAsFixed(3)}, ${y.toStringAsFixed(3)}, '
        '${w.toStringAsFixed(3)}, ${h.toStringAsFixed(3)}])';
  }
}

class DetectionResults {
  final List<DetectionResult> detections;
  final int timestamp;

  DetectionResults({required this.detections, required this.timestamp});

  bool get isEmpty => detections.isEmpty;
  bool get isNotEmpty => detections.isNotEmpty;
  int get length => detections.length;
}

class InferenceBridge {
  late final DynamicLibrary _lib;
  late final InitModelDart _initModel;
  late final FreeModelDart _freeModel;
  late final SetFrameSkipDart _setFrameSkip;
  late final ShouldProcessFrameDart _shouldProcessFrame;
  late final StartInferenceWorkerDart _startInferenceWorker;
  late final StopInferenceWorkerDart _stopInferenceWorker;
  late final PushFrameToQueueDart _pushFrameToQueue;
  late final GetLatestResultsDart _getLatestResults;
  late final RunInferenceYUVDart _runInferenceYUV;
  late final FreeResultsDart _freeResults;

  Pointer<Void>? _modelContext;
  bool _workerRunning = false;

  InferenceBridge() {
    // Load the native library
    if (Platform.isAndroid) {
      _lib = DynamicLibrary.open('libinference.so');
    } else if (Platform.isIOS) {
      _lib = DynamicLibrary.process();
    } else {
      throw UnsupportedError('Unsupported platform');
    }

    // Lookup functions
    _initModel = _lib
        .lookup<NativeFunction<InitModelNative>>('init_model')
        .asFunction();
    _freeModel = _lib
        .lookup<NativeFunction<FreeModelNative>>('free_model')
        .asFunction();
    _setFrameSkip = _lib
        .lookup<NativeFunction<SetFrameSkipNative>>('set_frame_skip')
        .asFunction();
    _shouldProcessFrame = _lib
        .lookup<NativeFunction<ShouldProcessFrameNative>>(
          'should_process_frame',
        )
        .asFunction();

    // Producer-Consumer API
    _startInferenceWorker = _lib
        .lookup<NativeFunction<StartInferenceWorkerNative>>(
          'start_inference_worker',
        )
        .asFunction();
    _stopInferenceWorker = _lib
        .lookup<NativeFunction<StopInferenceWorkerNative>>(
          'stop_inference_worker',
        )
        .asFunction();
    _pushFrameToQueue = _lib
        .lookup<NativeFunction<PushFrameToQueueNative>>('push_frame_to_queue')
        .asFunction();
    _getLatestResults = _lib
        .lookup<NativeFunction<GetLatestResultsNative>>('get_latest_results')
        .asFunction();

    // Legacy synchronous API
    _runInferenceYUV = _lib
        .lookup<NativeFunction<RunInferenceYUVNative>>('run_inference_yuv')
        .asFunction();
    _freeResults = _lib
        .lookup<NativeFunction<FreeResultsNative>>('free_results')
        .asFunction();
  }

  /// Load the ONNX model
  bool loadModel(String modelPath) {
    final pathPtr = modelPath.toNativeUtf8();
    try {
      _modelContext = _initModel(pathPtr);
      return _modelContext != nullptr;
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Set frame skip mode
  /// - skipMode: 0 = no skip, 1 = every 2nd frame, 2 = skip N consecutive
  /// - skipCount: number of frames to skip (only used when skipMode = 2)
  void setFrameSkip(int skipMode, int skipCount) {
    if (_modelContext == null || _modelContext == nullptr) return;
    _setFrameSkip(_modelContext!, skipMode, skipCount);
  }

  /// Check if current frame should be processed (for manual frame skipping)
  bool shouldProcessFrame() {
    if (_modelContext == null || _modelContext == nullptr) return true;
    return _shouldProcessFrame(_modelContext!) == 1;
  }

  // ============ Producer-Consumer API (Recommended) ============

  /// Start the background inference worker thread
  /// - maxQueueSize: Maximum number of frames to buffer (recommended: 2-3)
  ///
  /// This starts a dedicated thread that processes frames asynchronously.
  /// Call this once after loading the model.
  void startInferenceWorker({int maxQueueSize = 2}) {
    if (_modelContext == null || _modelContext == nullptr) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }
    if (_workerRunning) {
      print('Warning: Inference worker already running');
      return;
    }

    _startInferenceWorker(_modelContext!, maxQueueSize);
    _workerRunning = true;
    print('Inference worker started with queue size $maxQueueSize');
  }

  /// Stop the background inference worker thread
  ///
  /// Call this when you're done with inference (e.g., when disposing the detector).
  void stopInferenceWorker() {
    if (_modelContext == null || _modelContext == nullptr) return;
    if (!_workerRunning) return;

    _stopInferenceWorker(_modelContext!);
    _workerRunning = false;
    print('Inference worker stopped');
  }

  /// Push a camera frame to the inference queue (non-blocking)
  ///
  /// Returns true if the frame was queued, false if queue is full (frame dropped).
  /// This is very fast and won't block the camera thread.
  ///
  /// Usage in camera callback:
  /// ```dart
  /// cameraController.startImageStream((image) {
  ///   final success = bridge.pushFrame(
  ///     image.planes[0].bytes,
  ///     image.planes[1].bytes,
  ///     image.planes[2].bytes,
  ///     // ... strides and dimensions
  ///   );
  ///
  ///   if (!success) {
  ///     print('Frame dropped - inference too slow');
  ///   }
  /// });
  /// ```
  bool pushFrame(
    Uint8List yPlane,
    Uint8List uPlane,
    Uint8List vPlane,
    int yRowStride,
    int uvRowStride,
    int uvPixelStride,
    int imgWidth,
    int imgHeight,
  ) {
    if (_modelContext == null || _modelContext == nullptr) {
      return false;
    }
    if (!_workerRunning) {
      throw StateError(
        'Worker not running. Call startInferenceWorker() first.',
      );
    }

    final yPtr = malloc<Uint8>(yPlane.length);
    final uPtr = malloc<Uint8>(uPlane.length);
    final vPtr = malloc<Uint8>(vPlane.length);

    try {
      // Copy data to native memory
      yPtr.asTypedList(yPlane.length).setAll(0, yPlane);
      uPtr.asTypedList(uPlane.length).setAll(0, uPlane);
      vPtr.asTypedList(vPlane.length).setAll(0, vPlane);

      // Push to queue (non-blocking)
      final result = _pushFrameToQueue(
        _modelContext!,
        yPtr,
        uPtr,
        vPtr,
        yRowStride,
        uvRowStride,
        uvPixelStride,
        imgWidth,
        imgHeight,
      );

      return result == 1;
    } finally {
      malloc.free(yPtr);
      malloc.free(uPtr);
      malloc.free(vPtr);
    }
  }

  /// Get the latest detection results (non-blocking)
  ///
  /// Returns the most recent detection results processed by the worker thread.
  /// This is very fast and won't block - call it in your UI update loop.
  ///
  /// Returns null if no results are available yet.
  ///
  /// Usage:
  /// ```dart
  /// // In your periodic UI update (e.g., Timer.periodic or setState)
  /// final results = bridge.getLatestResults();
  /// if (results != null && results.isNotEmpty) {
  ///   // Update UI with detections
  ///   for (var detection in results.detections) {
  ///     drawBoundingBox(detection);
  ///   }
  /// }
  /// ```
  DetectionResults? getLatestResults() {
    if (_modelContext == null || _modelContext == nullptr) {
      return null;
    }

    final outCountPtr = malloc<Int32>();
    final outTimestampPtr = malloc<Int64>();

    try {
      final resultsPtr = _getLatestResults(
        _modelContext!,
        outCountPtr,
        outTimestampPtr,
      );

      final count = outCountPtr.value;
      final timestamp = outTimestampPtr.value;

      if (resultsPtr == nullptr || count == 0) {
        return DetectionResults(detections: [], timestamp: timestamp);
      }

      // Parse results (pointer is managed by C++, don't free it)
      final results = <DetectionResult>[];
      final floatList = resultsPtr.asTypedList(count * 6);

      for (int i = 0; i < count; i++) {
        results.add(
          DetectionResult(
            x: floatList[i * 6 + 0],
            y: floatList[i * 6 + 1],
            w: floatList[i * 6 + 2],
            h: floatList[i * 6 + 3],
            confidence: floatList[i * 6 + 4],
            classId: floatList[i * 6 + 5].toInt(),
          ),
        );
      }

      return DetectionResults(detections: results, timestamp: timestamp);
    } finally {
      malloc.free(outCountPtr);
      malloc.free(outTimestampPtr);
    }
  }

  // ============ Legacy Synchronous API ============

  /// Run inference synchronously (blocking - use for testing only)
  ///
  /// This blocks the calling thread until inference completes.
  /// For real-time camera inference, use the Producer-Consumer API instead:
  /// - startInferenceWorker()
  /// - pushFrame()
  /// - getLatestResults()
  @Deprecated(
    'Use pushFrame() and getLatestResults() instead for better performance',
  )
  List<DetectionResult> runInference(
    Uint8List yPlane,
    Uint8List uPlane,
    Uint8List vPlane,
    int yRowStride,
    int uvRowStride,
    int uvPixelStride,
    int imgWidth,
    int imgHeight,
  ) {
    if (_modelContext == null || _modelContext == nullptr) {
      return [];
    }

    final yPtr = malloc<Uint8>(yPlane.length);
    final uPtr = malloc<Uint8>(uPlane.length);
    final vPtr = malloc<Uint8>(vPlane.length);
    final outCountPtr = malloc<Int32>();

    try {
      // Copy data to native memory
      yPtr.asTypedList(yPlane.length).setAll(0, yPlane);
      uPtr.asTypedList(uPlane.length).setAll(0, uPlane);
      vPtr.asTypedList(vPlane.length).setAll(0, vPlane);

      // Run inference (blocking)
      final resultsPtr = _runInferenceYUV(
        _modelContext!,
        yPtr,
        uPtr,
        vPtr,
        yRowStride,
        uvRowStride,
        uvPixelStride,
        imgWidth,
        imgHeight,
        outCountPtr,
      );

      final count = outCountPtr.value;

      if (resultsPtr == nullptr || count == 0) {
        return [];
      }

      // Parse results
      final results = <DetectionResult>[];
      final floatList = resultsPtr.asTypedList(count * 6);

      for (int i = 0; i < count; i++) {
        results.add(
          DetectionResult(
            x: floatList[i * 6 + 0],
            y: floatList[i * 6 + 1],
            w: floatList[i * 6 + 2],
            h: floatList[i * 6 + 3],
            confidence: floatList[i * 6 + 4],
            classId: floatList[i * 6 + 5].toInt(),
          ),
        );
      }

      // Free native results
      _freeResults(resultsPtr);

      return results;
    } finally {
      malloc.free(yPtr);
      malloc.free(uPtr);
      malloc.free(vPtr);
      malloc.free(outCountPtr);
    }
  }

  /// Clean up and free resources
  void dispose() {
    // Stop worker if running
    if (_workerRunning) {
      stopInferenceWorker();
    }

    // Free model
    if (_modelContext != null && _modelContext != nullptr) {
      _freeModel(_modelContext!);
      _modelContext = null;
    }
  }

  /// Check if worker is running
  bool get isWorkerRunning => _workerRunning;

  /// Check if model is loaded
  bool get isModelLoaded => _modelContext != null && _modelContext != nullptr;
}
