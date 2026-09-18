// Configuration for trajectory viewer

export const ViewerConfig = {
    // Animation speed in milliseconds between points
    // Lower = faster, Higher = slower
    animationSpeed: 100,  // Default: 50ms per point
    
    // Auto-update interval for checking new data (in ms)
    // For live data: check frequently to avoid stuttering
    // For static files: duplicate detection prevents re-plotting
    updateInterval: 500,  // Default: 500ms (0.5 seconds)
    
    // Camera settings
    cameraFOV: 75,
    cameraNear: 0.1,
    cameraFar: 10000,
    cameraInitialPosition: { x: 8, y: 8, z: 8 },  // Initial camera position
    
    // Visual settings
    axesHelperSize: 2,
    gridSize: 100,
    gridDivisions: 50,
    
    // 3D Model settings (supports FBX, GLTF, GLB)
    modelPath: './models/turtle/turtle.glb',  // Path to 3D model (null = disabled)
    modelScale: 0.25,                 // Scale factor for the model
    modelRotation: { x: 0, y: 0, z: 180 },  // Initial rotation in degrees (x, y, z)
    showTrajectoryLine: true         // Show line connecting points
};
