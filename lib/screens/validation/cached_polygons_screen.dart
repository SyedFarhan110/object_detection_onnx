import 'dart:io';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:object_detection_app/services/polygon_storage.dart';
import 'package:object_detection_app/services/polygon_validator_bridge.dart';
import 'validation_capture_screen.dart';

class CachedPolygonsScreen extends StatefulWidget {
  final CameraDescription camera;

  const CachedPolygonsScreen({Key? key, required this.camera}) : super(key: key);

  @override
  State<CachedPolygonsScreen> createState() => _CachedPolygonsScreenState();
}

class _CachedPolygonsScreenState extends State<CachedPolygonsScreen> {
  List<Map<String, dynamic>> _polygons = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadPolygons();
  }

  Future<void> _loadPolygons() async {
    setState(() => _isLoading = true);
    final polygons = await PolygonStorage.getPolygons();
    setState(() {
      _polygons = polygons;
      _isLoading = false;
    });
  }

  Future<void> _deletePolygon(int index) async {
    final polygonName = _polygons[index]['name'];
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Row(
          children: [
            Icon(Icons.warning, color: Colors.orange),
            SizedBox(width: 12),
            Text('Delete Polygon?'),
          ],
        ),
        content: Text('Are you sure you want to delete "$polygonName"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      // Implement delete functionality
      setState(() {
        _polygons.removeAt(index);
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Row(
              children: [
                Icon(Icons.delete, color: Colors.white),
                SizedBox(width: 12),
                Text('Polygon deleted'),
              ],
            ),
            backgroundColor: Colors.red,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Validation Frames',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          if (_polygons.isNotEmpty)
            TextButton.icon(
              icon: const Icon(Icons.info_outline),
              label: Text('${_polygons.length} frame${_polygons.length > 1 ? 's' : ''}'),
              onPressed: null,
              style: TextButton.styleFrom(
                foregroundColor: Colors.white70,
              ),
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _polygons.isEmpty 
              ? _buildEmptyState()
              : RefreshIndicator(
                  onRefresh: _loadPolygons,
                  child: ListView.builder(
                      padding: const EdgeInsets.all(16),
                      itemCount: _polygons.length,
                      itemBuilder: (context, index) {
                        final polyData = _polygons[index];
                        final rawPoints = polyData['points'] as List;
                        final points = rawPoints.map((p) => Point2D(p['x'], p['y'])).toList();
                        
                        return _buildPolygonCard(polyData, points, index);
                      },
                    ),
                ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.crop_free,
            size: 80,
            color: Colors.grey[300],
          ),
          const SizedBox(height: 24),
          Text(
            'No validation frames',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.grey[600],
            ),
          ),
          const SizedBox(height: 12),
          Text(
            'Create a new polygon to get started',
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey[500],
            ),
          ),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            icon: const Icon(Icons.add),
            label: const Text('Draw New Polygon'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  }

  Widget _buildPolygonCard(Map<String, dynamic> polyData, List<Point2D> points, int index) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () {
            Navigator.push(
              context, 
              MaterialPageRoute(
                builder: (_) => ValidationCaptureScreen(
                  camera: widget.camera, 
                  polygonName: polyData['name'], 
                  points: points
                ),
              ),
            );
          },
          borderRadius: BorderRadius.circular(16),
          child: Container(
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.08),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
              border: Border.all(
                color: Colors.grey[200]!,
                width: 1,
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  // Thumbnail or placeholder
                  Builder(builder: (context) {
                    final imagePath = polyData['imagePath'] as String?;
                    if (imagePath != null && File(imagePath).existsSync()) {
                      return Container(
                        width: 72,
                        height: 72,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          image: DecorationImage(
                            image: FileImage(File(imagePath)),
                            fit: BoxFit.cover,
                          ),
                        ),
                      );
                    }

                    return Container(
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Theme.of(context).primaryColor,
                            Theme.of(context).primaryColor.withOpacity(0.7),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.crop_free,
                        color: Colors.white,
                        size: 28,
                      ),
                    );
                  }),
                  const SizedBox(width: 16),
                  
                  // Content
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          polyData['name'],
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.black87,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Row(
                          children: [
                            Icon(Icons.pin_drop, size: 16, color: Colors.grey[600]),
                            const SizedBox(width: 6),
                            Text(
                              '${points.length} points',
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey[600],
                              ),
                            ),
                            const SizedBox(width: 16),
                            Icon(Icons.access_time, size: 16, color: Colors.grey[600]),
                            const SizedBox(width: 6),
                            Text(
                              'Ready',
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  
                  // Actions
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: Icon(Icons.delete_outline, color: Colors.red[400]),
                        onPressed: () => _deletePolygon(index),
                        tooltip: 'Delete',
                      ),
                      Icon(
                        Icons.arrow_forward_ios,
                        color: Colors.grey[400],
                        size: 20,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}