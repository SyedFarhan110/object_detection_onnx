import 'package:flutter/material.dart';
import 'package:object_detection_app/services/polygon_validator_bridge.dart';

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
      canvas.drawPath(path, fillPaint);
    }

    canvas.drawPath(path, strokePaint);

    final pointPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;

    final pointBorderPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (final point in points) {
      final center = Offset(point.x, point.y);
      canvas.drawCircle(center, 6.0, pointPaint);
      canvas.drawCircle(center, 6.0, pointBorderPaint);
    }
  }

  @override
  bool shouldRepaint(PolygonPainter oldDelegate) {
    return oldDelegate.points.length != points.length ||
        oldDelegate.isComplete != isComplete;
  }
}
