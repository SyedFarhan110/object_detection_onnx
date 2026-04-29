import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'inference_bridge.dart' show DetectionResult;

typedef CreateCvNative = Pointer<Void> Function();
typedef CreateCvDart = Pointer<Void> Function();

typedef FreeCvNative = Void Function(Pointer<Void> ctx);
typedef FreeCvDart = void Function(Pointer<Void> ctx);

typedef RunCvNative =
    Pointer<Float> Function(
      Pointer<Void> ctx,
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

typedef RunCvDart =
    Pointer<Float> Function(
      Pointer<Void> ctx,
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

class CvBridge {
  late final DynamicLibrary _lib;
  late final CreateCvDart _create;
  late final FreeCvDart _free;
  late final RunCvDart _run;
  late final FreeResultsDart _freeResults;

  Pointer<Void>? _ctx;

  CvBridge() {
    if (Platform.isAndroid) {
      _lib = DynamicLibrary.open('libcv_detector.so');
    } else if (Platform.isIOS) {
      _lib = DynamicLibrary.process();
    } else {
      throw UnsupportedError('Unsupported platform for CV bridge');
    }

    _create = _lib
        .lookup<NativeFunction<CreateCvNative>>('create_cv_detector')
        .asFunction();
    _free = _lib
        .lookup<NativeFunction<FreeCvNative>>('free_cv_detector')
        .asFunction();
    _run = _lib
        .lookup<NativeFunction<RunCvNative>>('run_cv_detector_yuv')
        .asFunction();
    _freeResults = _lib
        .lookup<NativeFunction<FreeResultsNative>>('free_cv_results')
        .asFunction();

    _ctx = _create();
  }

  List<DetectionResult> runSync(
    Uint8List yPlane,
    Uint8List uPlane,
    Uint8List vPlane,
    int yRowStride,
    int uvRowStride,
    int uvPixelStride,
    int imgWidth,
    int imgHeight,
  ) {
    if (_ctx == null || _ctx == nullptr) return [];

    final yPtr = malloc<Uint8>(yPlane.length);
    final uPtr = malloc<Uint8>(uPlane.length);
    final vPtr = malloc<Uint8>(vPlane.length);
    final outCountPtr = malloc<Int32>();

    try {
      yPtr.asTypedList(yPlane.length).setAll(0, yPlane);
      uPtr.asTypedList(uPlane.length).setAll(0, uPlane);
      vPtr.asTypedList(vPlane.length).setAll(0, vPlane);

      final resultsPtr = _run(
        _ctx!,
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
      if (resultsPtr == nullptr || count == 0) return [];

      final floatList = resultsPtr.asTypedList(count * 6);
      final detections = <DetectionResult>[];
      for (int i = 0; i < count; i++) {
        detections.add(
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

      // free native buffer
      _freeResults(resultsPtr.cast<Float>());

      return detections;
    } finally {
      malloc.free(yPtr);
      malloc.free(uPtr);
      malloc.free(vPtr);
      malloc.free(outCountPtr);
    }
  }

  void dispose() {
    if (_ctx != null && _ctx != nullptr) {
      _free(_ctx!);
      _ctx = null;
    }
  }
}
