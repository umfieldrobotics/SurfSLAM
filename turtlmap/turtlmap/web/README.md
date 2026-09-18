# TurtlMap Web Visualization

3D trajectory visualization with animated plotting and 3D model support.

## 🚀 Quick Start

### Offline Visualization (Static File)

```bash
cd /path/to/turtlmap/turtlmap/web
python3 offline_viz.py ../../build/trajectory_live.txt
```

Browser opens automatically at **http://localhost:8000**

### Live Visualization (Real-time Updates)

```bash
cd /path/to/turtlmap/build
./turtlmap/turtlmap_backend config.yaml data.hdf5 1 trajectory.txt \
    --live-file=trajectory_live.txt --web-port=8080
```

Open **http://localhost:8080** in your browser.

---

## 📁 Trajectory File Format

One pose per line: `x y z qw qx qy qz`

- `x y z`: Position (Z-axis UP)
- `qw qx qy qz`: Quaternion orientation

**Coordinate System:** X=Forward (Red), Y=Left (Green), Z=Up (Blue)

**Example:**
```
-0.137391 -0.029698 0.067484 -0.004926 0.962190 0.272325 0.002175
-0.119680 -0.042762 0.065105 -0.002363 0.866162 0.499757 0.000013
```

---

## 🎬 Features

### Animated Plotting
- Gradual point-by-point visualization
- Real-time progress display
- Status: 🟢 Plotting | ⬜ Complete | 🟡 Loading

### Visual Elements
- **Gradient Line**: Cyan (start) → Yellow (end)
- **RGB Axes**: At latest position (orientation)
- **Grid**: Spatial reference in XY plane
- **Auto Camera**: Fits entire trajectory
- **3D Models**: Animate FBX/GLTF/GLB models along trajectory

### Mouse Controls
- **Left Click + Drag**: Rotate camera
- **Right Click + Drag**: Pan
- **Scroll**: Zoom

---

## 🎨 3D Model Support

Animate your robot/vehicle model along the trajectory!

**Supported formats:** FBX, GLTF, GLB

### Setup

1. **Place model:**
```bash
mkdir -p models
cp /path/to/robot.glb models/
```

2. **Configure** in `js/config.js`:
```javascript
export const ViewerConfig = {
    modelPath: './models/robot.glb',      // null = disabled
    modelScale: 1.0,                      // Scale factor
    modelRotation: { x: -90, y: 0, z: 0 }, // Degrees
    showTrajectoryLine: true,             // Show path
    // ...
};
```

3. **Run:** Refresh browser to see animated model!

### Configuration Tips

```javascript
// Model orientation fixes
modelRotation: { x: -90, y: 0, z: 0 },    // Stand up
modelRotation: { x: 0, y: 0, z: 180 },    // Flip around
modelRotation: { x: -90, y: 0, z: 90 },   // Stand + turn

// Sizing
modelScale: 0.5,   // Smaller
modelScale: 2.0,   // Bigger
```

**Recommended:** Use GLB format for best performance (single file, fast loading)

---

## ⚙️ Configuration

Edit `js/config.js`:

```javascript
export const ViewerConfig = {
    animationSpeed: 50,       // ms between points (lower = faster)
    updateInterval: 500,      // ms between data fetches
    cameraInitialPosition: { x: 5, y: 5, z: 5 },
    
    // 3D Model settings
    modelPath: './models/robot.glb',
    modelScale: 1.0,
    modelRotation: { x: 0, y: 0, z: 0 },
    showTrajectoryLine: true,
};
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `animationSpeed` | 50ms | Time between plotting points |
| `updateInterval` | 500ms | Data fetch frequency |
| `modelPath` | null | Path to 3D model (FBX/GLTF/GLB) |
| `modelScale` | 1.0 | Model size multiplier |
| `modelRotation` | {x:0, y:0, z:0} | Initial rotation (degrees) |
| `showTrajectoryLine` | true | Show trajectory path |

**Speed Presets:**
- Fast: `animationSpeed: 25` (~7 sec for 290 points)
- Normal: `animationSpeed: 50` (~15 sec, default)
- Slow: `animationSpeed: 100` (~30 sec)
- Instant: `animationSpeed: 0` (no animation)

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
pkill -f "python3.*offline_viz.py"
# Or use different port:
python3 offline_viz.py trajectory.txt 8001
```

### No Visualization Appears
1. Check browser console (F12) for errors
2. Verify file: `wc -l trajectory.txt`
3. Test server: `curl http://localhost:8000/traj`
4. Try Chrome/Firefox

### Model Not Appearing
1. Check console (F12) for loading errors
2. Verify path in `config.js` is correct
3. Test model online: https://gltf-viewer.donmccurdy.com/

### Model Wrong Size
```javascript
modelScale: 0.01,  // Try: 0.01, 0.1, 1, 10
```

### Model Wrong Orientation
```javascript
modelRotation: { x: -90, y: 0, z: 0 },  // Adjust until correct
```

---

## 📂 Directory Structure

```
web/
├── offline_viz.py       # Offline visualizer script
├── index.html           # Main page
├── js/
│   ├── viewer.js        # Visualization logic
│   └── config.js        # Settings
├── models/              # 3D models (create manually)
│   └── robot.glb
├── node_modules/three/  # Three.js library
└── package.json
```

---

## 🔍 Technical Details

**Offline Mode:**
- Python HTTP server (blocking)
- Static file serving
- Duplicate detection (no looping)

**Live Mode:**
- C++ embedded server
- Incremental streaming (only new points)
- 500ms polling for smooth updates
- 98%+ bandwidth reduction

**Browser:** Chrome 90+, Firefox 88+, Edge 90+, Safari 14+

---

## 💡 Tips

1. **First time**: Watch full animation
2. **Restart**: Refresh browser (F5)
3. **Speed up**: Lower `animationSpeed` in config
4. **Multiple datasets**: Use different ports
5. **3D models**: Use GLB format for best performance

---

## 📚 Resources

- **Three.js**: https://threejs.org/docs/
- **GLTF Viewer**: https://gltf-viewer.donmccurdy.com/
- **Free Models**: https://sketchfab.com/, https://poly.pizza/

---

**Version**: 1.0.0  
**Three.js**: r170.0  
**Updated**: 2025-10-30
