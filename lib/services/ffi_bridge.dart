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
}

class InferenceBridge {
  late final DynamicLibrary _lib;
  late final InitModelDart _initModel;
  late final FreeModelDart _freeModel;
  late final SetFrameSkipDart _setFrameSkip;
  late final ShouldProcessFrameDart _shouldProcessFrame;
  late final RunInferenceYUVDart _runInferenceYUV;
  late final FreeResultsDart _freeResults;

  Pointer<Void>? _modelContext;

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
    _runInferenceYUV = _lib
        .lookup<NativeFunction<RunInferenceYUVNative>>('run_inference_yuv')
        .asFunction();
    _freeResults = _lib
        .lookup<NativeFunction<FreeResultsNative>>('free_results')
        .asFunction();
  }

  bool loadModel(String modelPath) {
    final pathPtr = modelPath.toNativeUtf8();
    try {
      _modelContext = _initModel(pathPtr);
      return _modelContext != nullptr;
    } finally {
      malloc.free(pathPtr);
    }
  }

  void setFrameSkip(int skipMode, int skipCount) {
    if (_modelContext == null || _modelContext == nullptr) return;
    _setFrameSkip(_modelContext!, skipMode, skipCount);
  }

  bool shouldProcessFrame() {
    if (_modelContext == null || _modelContext == nullptr) return true;
    return _shouldProcessFrame(_modelContext!) == 1;
  }

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

      // Run inference
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

  void dispose() {
    if (_modelContext != null && _modelContext != nullptr) {
      _freeModel(_modelContext!);
      _modelContext = null;
    }
  }
}
