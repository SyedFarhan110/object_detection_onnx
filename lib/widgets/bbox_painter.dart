import 'package:flutter/material.dart';

class BoundingBoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> results;
  final int originalImageWidth;
  final int originalImageHeight;
  final double displayWidth;
  final double displayHeight;

  BoundingBoxPainter(
    this.results,
    this.originalImageWidth,
    this.originalImageHeight,
    this.displayWidth,
    this.displayHeight,
  );

  @override
  void paint(Canvas canvas, Size size) {
    if (results.isEmpty ||
        originalImageWidth == 0 ||
        originalImageHeight == 0) {
      return;
    }

    final imageAspectRatio = originalImageWidth / originalImageHeight;
    final containerAspectRatio = size.width / size.height;

    double scaleX;
    double scaleY;
    double offsetX = 0;
    double offsetY = 0;

    if (imageAspectRatio > containerAspectRatio) {
      scaleX = size.width / originalImageWidth;
      scaleY = scaleX;
      offsetY = (size.height - (originalImageHeight * scaleY)) / 2;
    } else {
      scaleY = size.height / originalImageHeight;
      scaleX = scaleY;
      offsetX = (size.width - (originalImageWidth * scaleX)) / 2;
    }

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (var result in results) {
      final x1 = (result['x1'] ?? 0.0).toDouble();
      final y1 = (result['y1'] ?? 0.0).toDouble();
      final x2 = (result['x2'] ?? 0.0).toDouble();
      final y2 = (result['y2'] ?? 0.0).toDouble();

      final scaledX1 = (x1 * scaleX) + offsetX;
      final scaledY1 = (y1 * scaleY) + offsetY;
      final scaledX2 = (x2 * scaleX) + offsetX;
      final scaledY2 = (y2 * scaleY) + offsetY;

      final className = result['class'] ?? 'Unknown';
      final confidence = (result['confidence'] ?? 0.0) * 100;
      final label = '$className ${confidence.toStringAsFixed(1)}%';

      if (confidence >= 80) {
        paint.color = Colors.green;
      } else if (confidence >= 50) {
        paint.color = Colors.orange;
      } else {
        paint.color = Colors.red;
      }

      final rect = Rect.fromLTRB(scaledX1, scaledY1, scaledX2, scaledY2);
      canvas.drawRect(rect, paint);

      final textSpan = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
          shadows: [
            Shadow(offset: Offset(1, 1), blurRadius: 3, color: Colors.black87),
          ],
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final labelY = scaledY1 - textPainter.height - 8;
      final finalLabelY = labelY < 0 ? scaledY1 + 4 : labelY;

      final labelRect = Rect.fromLTWH(
        scaledX1,
        finalLabelY,
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final labelPaint = Paint()..color = paint.color;
      canvas.drawRect(labelRect, labelPaint);

      textPainter.paint(canvas, Offset(scaledX1 + 4, finalLabelY + 2));
    }
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return oldDelegate.results != results ||
        oldDelegate.originalImageWidth != originalImageWidth ||
        oldDelegate.originalImageHeight != originalImageHeight ||
        oldDelegate.displayWidth != displayWidth ||
        oldDelegate.displayHeight != displayHeight;
  }
}
