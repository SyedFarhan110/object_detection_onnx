import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:object_detection_app/services/polygon_validator_bridge.dart';

class PolygonStorage {
  static const String _key = 'saved_polygons';

  static Future<void> savePolygon(String name, List<Point2D> points) async {
    final prefs = await SharedPreferences.getInstance();
    List<String> polygonsStr = prefs.getStringList(_key) ?? [];
    
    // Check if name exists
    if (polygonsStr.any((p) => json.decode(p)['name'] == name)) {
      throw Exception('Polygon with this name already exists.');
    }

    final polygonData = {
      'name': name,
      'points': points.map((p) => {'x': p.x, 'y': p.y}).toList(),
    };
    
    polygonsStr.add(json.encode(polygonData));
    await prefs.setStringList(_key, polygonsStr);
  }

  static Future<List<Map<String, dynamic>>> getPolygons() async {
    final prefs = await SharedPreferences.getInstance();
    List<String> polygonsStr = prefs.getStringList(_key) ?? [];
    return polygonsStr.map((p) => json.decode(p) as Map<String, dynamic>).toList();
  }
}
