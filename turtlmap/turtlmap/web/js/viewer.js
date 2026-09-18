import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { ViewerConfig } from './config.js';

export class TrajectoryViewer {
    constructor() {
        this.allTrajectoryPoints = [];  // Full trajectory data
        this.displayedPoints = [];       // Points currently displayed
        this.trajectoryLine = null;
        this.axesHelper = null;
        this.gridHelper = null;
        this.updateInterval = null;
        this.animationInterval = null;
        this.sensorUpdateInterval = null;
        this.currentIndex = 0;
        this.animationSpeed = ViewerConfig.animationSpeed;  // ms between adding points
        this.isAnimating = false;
        this.lastDataHash = null;        // Track last fetched data to avoid duplicates
        this.startTime = null;           // Track when first sensor data arrives
        
        // Sensor data for charts
        this.maxDataPoints = 100;  // Show last 100 data points
        this.sensorTimeData = [];
        this.imuAccelData = { ax: [], ay: [], az: [] };
        this.imuGyroData = { wx: [], wy: [], wz: [] };
        this.dvlVelData = { vx: [], vy: [], vz: [] };
        this.baroDepthData = [];
        
        // 3D model support (FBX/GLTF/GLB)
        this.model = null;               // The single 3D model that follows trajectory
        this.baseRotation = null;        // Base rotation from config
        this.fbxLoader = new FBXLoader();
        this.gltfLoader = new GLTFLoader();
        
        this.initCharts();
        this.init();
        this.loadModel();
        this.startAutoUpdate();
        this.startSensorUpdate();

        // Covariance visualization data
        this.covUpdateInterval = null;
        this.covLastHash = null;
        this.covEllipses = [];
        this.startCovarianceUpdate();
    }
    
    initCharts() {
        const chartConfig = (title, datasets) => ({
            type: 'line',
            data: {
                labels: [],
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#ddd',
                            font: { size: 10 },
                            boxWidth: 15,
                            padding: 5
                        }
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time (s)',
                            color: '#888',
                            font: { size: 10 }
                        },
                        ticks: {
                            color: '#888',
                            font: { size: 9 },
                            maxTicksLimit: 5
                        },
                        grid: {
                            color: '#333'
                        }
                    },
                    y: {
                        display: true,
                        ticks: {
                            color: '#888',
                            font: { size: 9 },
                            maxTicksLimit: 5
                        },
                        grid: {
                            color: '#333'
                        }
                    }
                }
            }
        });
        
        // IMU Acceleration Chart
        this.imuAccelChart = new Chart(
            document.getElementById('imu-accel-chart'),
            chartConfig('IMU Acceleration', [
                { label: 'ax', data: [], borderColor: '#ff4444', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 },
                { label: 'ay', data: [], borderColor: '#44ff44', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 },
                { label: 'az', data: [], borderColor: '#4444ff', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 }
            ])
        );
        
        // IMU Gyro Chart
        this.imuGyroChart = new Chart(
            document.getElementById('imu-gyro-chart'),
            chartConfig('IMU Gyro', [
                { label: 'wx', data: [], borderColor: '#ff4444', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 },
                { label: 'wy', data: [], borderColor: '#44ff44', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 },
                { label: 'wz', data: [], borderColor: '#4444ff', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 }
            ])
        );
        
        // DVL Velocity Chart
        this.dvlVelChart = new Chart(
            document.getElementById('dvl-vel-chart'),
            chartConfig('DVL Velocity', [
                { label: 'vx', data: [], borderColor: '#ff4444', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 },
                { label: 'vy', data: [], borderColor: '#44ff44', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 },
                { label: 'vz', data: [], borderColor: '#4444ff', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 }
            ])
        );
        
        // Barometer Chart
        this.baroChart = new Chart(
            document.getElementById('baro-chart'),
            chartConfig('Depth', [
                { label: 'depth', data: [], borderColor: '#00ffff', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 }
            ])
        );
    }
    
    init() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);
        
        // Camera setup - Z-axis points up
        this.camera = new THREE.PerspectiveCamera(
            ViewerConfig.cameraFOV,
            window.innerWidth / window.innerHeight,
            ViewerConfig.cameraNear,
            ViewerConfig.cameraFar
        );
        this.camera.up.set(0, 0, 1);  // Set Z as up direction
        const camPos = ViewerConfig.cameraInitialPosition;
        this.camera.position.set(camPos.x, camPos.y, camPos.z);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth * 0.75, window.innerHeight);  // 75% width
        const container = document.getElementById('canvas-container');
        container.appendChild(this.renderer.domElement);
        
        // OrbitControls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);
        
        // Grid helper - horizontal in XY plane (Z is up)
        this.gridHelper = new THREE.GridHelper(100, 50, 0x444444, 0x222222);
        this.gridHelper.rotation.x = Math.PI / 2;  // Rotate to XY plane
        this.scene.add(this.gridHelper);
        
        // Axes helper at origin (Red=X, Green=Y, Blue=Z pointing up)
        const originAxes = new THREE.AxesHelper(5);
        this.scene.add(originAxes);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start animation loop
        this.animate();
    }
    
    loadModel() {
        // Skip if no model path configured
        if (!ViewerConfig.modelPath) {
            console.log('No 3D model configured, using trajectory line visualization');
            return;
        }
        
        const modelPath = ViewerConfig.modelPath;
        const fileExtension = modelPath.split('.').pop().toLowerCase();
        
        console.log(`Loading ${fileExtension.toUpperCase()} model: ${modelPath}`);
        
        // Detect format and use appropriate loader
        if (fileExtension === 'fbx') {
            this.loadFBXModel(modelPath);
        } else if (fileExtension === 'gltf' || fileExtension === 'glb') {
            this.loadGLTFModel(modelPath);
        } else {
            console.error(`Unsupported model format: ${fileExtension}`);
            document.getElementById('info').textContent = `Error: Unsupported format .${fileExtension} (use .fbx, .gltf, or .glb)`;
        }
    }
    
    loadFBXModel(path) {
        this.fbxLoader.load(
            path,
            (fbx) => {
                this.onModelLoaded(fbx);
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(0);
                console.log(`Loading FBX model: ${percent}%`);
            },
            (error) => {
                this.onModelError(error, 'FBX');
            }
        );
    }
    
    loadGLTFModel(path) {
        this.gltfLoader.load(
            path,
            (gltf) => {
                // GLTF loader returns an object with a scene property
                this.onModelLoaded(gltf.scene);
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(0);
                console.log(`Loading GLTF model: ${percent}%`);
            },
            (error) => {
                this.onModelError(error, 'GLTF');
            }
        );
    }
    
    onModelLoaded(modelObject) {
        // Store the loaded model
        this.model = modelObject;
        
        // Apply scale from config
        this.model.scale.setScalar(ViewerConfig.modelScale);
        
        // Apply initial rotation from config (convert degrees to radians)
        const rot = ViewerConfig.modelRotation;
        this.model.rotation.set(
            THREE.MathUtils.degToRad(rot.x),
            THREE.MathUtils.degToRad(rot.y),
            THREE.MathUtils.degToRad(rot.z)
        );
        
        // Store the base rotation for later use
        this.baseRotation = this.model.rotation.clone();
        
        // Add model to scene
        this.scene.add(this.model);
        
        console.log('✓ 3D model loaded successfully');
        console.log(`  Scale: ${ViewerConfig.modelScale}`);
        console.log(`  Rotation: X=${rot.x}° Y=${rot.y}° Z=${rot.z}°`);
        document.getElementById('info').textContent = '3D model loaded! Waiting for trajectory data...';
        
        // Update position if we already have trajectory data
        if (this.displayedPoints.length > 0) {
            this.updateModelPosition();
        }
    }
    
    onModelError(error, format) {
        console.error(`Error loading ${format} model:`, error);
        document.getElementById('info').textContent = `Error loading ${format} model: ${error.message}`;
    }
    
    async fetchTrajectory() {
        try {
            // Fetch /traj endpoint
            // First call: returns all existing data (since last_file_size_ starts at 0)
            // Subsequent calls: returns only newly appended data
            const response = await fetch('/traj');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const text = await response.text();
            
            // Only process if we have data
            if (text.trim().length > 0) {
                // Check if this is duplicate data (static file server like Python http.server)
                // by comparing the text hash or content
                const dataHash = this.hashString(text);
                
                if (this.lastDataHash === dataHash) {
                    // Same data as before, don't append duplicates
                    // This happens with static file servers (Python http.server)
                    return;
                }
                
                this.lastDataHash = dataHash;
                
                // Always append mode - first call will have all data, subsequent calls have incremental
                const newPointCount = this.parseTrajectory(text, true);
                
                // Start animation if we have new data and not currently animating
                if (newPointCount > 0 && !this.isAnimating) {
                    this.startAnimation();
                }
            }
            
        } catch (error) {
            console.error('Error fetching trajectory:', error);
            document.getElementById('info').textContent = `Error: ${error.message}`;
        }
    }
    
    parseTrajectory(text, isIncremental = false) {
        const lines = text.trim().split('\n');
        const newPoints = [];
        
        for (const line of lines) {
            if (!line.trim()) continue;

            const values = line.trim().split(/\s+/).map(Number);

            // Support formats:
            // - timestamp x y z qw qx qy qz   (8 tokens)
            // - x y z qw qx qy qz             (7 tokens)
            // - x y z                          (3 tokens)
            let x, y, z, quat = null;
            if (values.length >= 8 && Number.isFinite(values[1]) && Number.isFinite(values[2]) && Number.isFinite(values[3])) {
                // timestamp + x y z + quaternion (qw qx qy qz)
                x = values[1];
                y = values[2];
                z = values[3];
                const qw = values[4], qx = values[5], qy = values[6], qz = values[7];
                if (Number.isFinite(qw) && Number.isFinite(qx) && Number.isFinite(qy) && Number.isFinite(qz)) {
                    quat = new THREE.Quaternion(qx, qy, qz, qw);
                }
            } else if (values.length >= 7 && Number.isFinite(values[0]) && Number.isFinite(values[1]) && Number.isFinite(values[2])) {
                // x y z qw qx qy qz
                x = values[0];
                y = values[1];
                z = values[2];
                const qw = values[3], qx = values[4], qy = values[5], qz = values[6];
                if (Number.isFinite(qw) && Number.isFinite(qx) && Number.isFinite(qy) && Number.isFinite(qz)) {
                    quat = new THREE.Quaternion(qx, qy, qz, qw);
                }
            } else if (values.length >= 3) {
                x = values[0];
                y = values[1];
                z = values[2];
            } else {
                continue;
            }

            newPoints.push({
                position: new THREE.Vector3(x, y, z),
                quaternion: quat
            });
        }
        
        const oldCount = this.allTrajectoryPoints.length;
        
        if (isIncremental) {
            // Append new points to existing trajectory
            this.allTrajectoryPoints.push(...newPoints);
        } else {
            // Replace entire trajectory (full fetch)
            this.allTrajectoryPoints = newPoints;
            
            // Reset animation state for new data
            if (newPoints.length < this.currentIndex) {
                this.currentIndex = 0;
                this.displayedPoints = [];
            }
        }
        
        return newPoints.length;  // Return count of new points added
    }
    
    updateVisualization() {
        // Remove old trajectory line if it exists
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
            this.trajectoryLine.geometry.dispose();
            this.trajectoryLine.material.dispose();
        }
        
        // Remove old axes helper if it exists
        if (this.axesHelper) {
            this.scene.remove(this.axesHelper);
        }
        
        if (this.displayedPoints.length === 0) return;
        
        // Create trajectory line (if enabled)
        if (ViewerConfig.showTrajectoryLine) {
            const positions = [];
            for (const point of this.displayedPoints) {
                positions.push(point.position.x, point.position.y, point.position.z);
            }
            
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            
            // Create gradient material (cyan to yellow) based on total trajectory
            const colors = [];
            const totalPoints = this.allTrajectoryPoints.length;
            for (let i = 0; i < this.displayedPoints.length; i++) {
                const actualIndex = i;  // Index in displayed points
                const t = actualIndex / Math.max(1, totalPoints - 1);
                // Cyan (0, 1, 1) to Yellow (1, 1, 0)
                colors.push(t, 1, 1 - t);
            }
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            const material = new THREE.LineBasicMaterial({
                vertexColors: true,
                linewidth: 2
            });
            
            this.trajectoryLine = new THREE.Line(geometry, material);
            this.scene.add(this.trajectoryLine);
        }
        
        // Update FBX model position
        this.updateModelPosition();
        
        // Add axes helper at the latest position
        if (this.displayedPoints.length > 0) {
            const latest = this.displayedPoints[this.displayedPoints.length - 1];
            this.axesHelper = new THREE.AxesHelper(2);
            this.axesHelper.position.copy(latest.position);
            
            // Apply orientation if quaternion is available
            if (latest.quaternion) {
                this.axesHelper.quaternion.copy(latest.quaternion);
            }
            
            this.scene.add(this.axesHelper);
        }
        
        // Auto-adjust camera to view trajectory (only once at start)
        if (this.displayedPoints.length < 10 || !this.cameraAdjusted) {
            this.fitCameraToTrajectory();
        }
    }
    
    updateModelPosition() {
        // Update 3D model position if loaded
        if (!this.model || this.displayedPoints.length === 0) return;
        
        // Get the latest displayed point
        const latest = this.displayedPoints[this.displayedPoints.length - 1];
        
        // Update position
        this.model.position.copy(latest.position);
        
        // Update orientation from quaternion
        if (latest.quaternion && this.baseRotation) {
            // Combine trajectory quaternion with base rotation
            // First apply trajectory rotation, then base rotation
            const baseQuat = new THREE.Quaternion().setFromEuler(this.baseRotation);
            this.model.quaternion.copy(latest.quaternion).multiply(baseQuat);
        } else if (latest.quaternion) {
            this.model.quaternion.copy(latest.quaternion);
        }
    }
    
    fitCameraToTrajectory() {
        // Use all trajectory points for camera fitting (not just displayed)
        const pointsToFit = this.allTrajectoryPoints.length > 0 ? 
            this.allTrajectoryPoints : this.displayedPoints;
            
        if (pointsToFit.length === 0) return;
        
        // Calculate bounding box
        const box = new THREE.Box3();
        for (const point of pointsToFit) {
            box.expandByPoint(point.position);
        }
        
        const center = new THREE.Vector3();
        box.getCenter(center);
        
        const size = new THREE.Vector3();
        box.getSize(size);
        
        const maxDim = Math.max(size.x, size.y, size.z);
        const distance = maxDim * 2;
        
        // Only adjust camera on first data load or significant change
        if (!this.cameraAdjusted || pointsToFit.length < 10) {
            // Position camera with Z-up orientation
            // Place camera at angle for better 3D view
            this.camera.position.set(
                center.x + distance * 0.7,
                center.y + distance * 0.7,
                center.z + distance * 0.7
            );
            this.controls.target.copy(center);
            this.cameraAdjusted = true;
        }
    }
    
    startAnimation() {
        if (this.isAnimating) return;
        
        this.isAnimating = true;
        this.animationInterval = setInterval(() => {
            if (this.currentIndex < this.allTrajectoryPoints.length) {
                // Add next point
                this.displayedPoints.push(this.allTrajectoryPoints[this.currentIndex]);
                this.currentIndex++;
                
                // Update visualization
                this.updateVisualization();
                this.updateInfo();
            } else {
                // Animation complete
                this.stopAnimation();
            }
        }, this.animationSpeed);
    }
    
    stopAnimation() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
            this.animationInterval = null;
        }
        this.isAnimating = false;
    }
    
    updateInfo() {
        const infoElement = document.getElementById('info');
        if (!infoElement) return;
        
        if (this.allTrajectoryPoints.length === 0) {
            infoElement.textContent = 'Waiting for trajectory data...';
            return;
        }
        
        const progress = this.displayedPoints.length;
        const total = this.allTrajectoryPoints.length;
        const percentage = ((progress / total) * 100).toFixed(0);
        
        if (this.displayedPoints.length > 0) {
            const latest = this.displayedPoints[this.displayedPoints.length - 1];
            const pos = latest.position;
            
            const status = this.isAnimating ? 
                `<span style="color: #0f0">● Plotting...</span>` : 
                `<span style="color: #888">■ Complete</span>`;
            
            infoElement.innerHTML = `
                ${status}<br/>
                Points: ${progress} / ${total} (${percentage}%)<br/>
                Latest: (${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})
            `;
        } else {
            infoElement.innerHTML = `
                <span style="color: #ff0">⟳ Loading...</span><br/>
                Total points: ${total}
            `;
        }
    }
    
    startAutoUpdate() {
        // Initial fetch
        this.fetchTrajectory();
        
        // Periodically fetch new data
        // - Static files (Python server): hash check prevents duplicates
        // - Live data (C++ server): gets incremental updates
        // Frequent polling ensures smooth live visualization
        this.updateInterval = setInterval(() => {
            this.fetchTrajectory();
        }, ViewerConfig.updateInterval);
    }

    // --- Covariance fetching & visualization ---
    async fetchCovariance() {
        try {
            const response = await fetch('/cov');
            if (!response.ok) return;
            const text = await response.text();
            if (!text.trim()) return;
            const hash = this.hashString(text);
            if (hash === this.covLastHash) return; // no changes
            this.covLastHash = hash;
            this.parseCovariance(text);
            this.updateCovarianceVisualization();
        } catch (e) {
            // silently ignore
        }
    }

    parseCovariance(text) {
        // Format per line: ts x y z cov_xx cov_xy cov_yy
        const lines = text.trim().split('\n');
        this.covData = [];
        for (const line of lines) {
            if (!line.trim()) continue;
            const parts = line.trim().split(/\s+/).map(Number);
            if (parts.length < 7) continue;
            const [ts, x, y, z, cxx, cxy, cyy] = parts;
            if ([ts,x,y,z,cxx,cxy,cyy].some(v => !Number.isFinite(v))) continue;
            this.covData.push({ ts, x, y, z, cxx, cxy, cyy });
        }
    }

    updateCovarianceVisualization() {
        // Remove existing ellipses and dispose resources
        for (const obj of this.covEllipses) {
            this.scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }
        this.covEllipses = [];
        if (!this.covData || this.covData.length === 0) return;

        // Draw each covariance as a FILLED ellipse in XY at height z
        for (const entry of this.covData) {
            const { x, y, z, cxx, cxy, cyy } = entry;
            // Build 2x2 covariance matrix
            const cov = new THREE.Matrix3();
            cov.set(
                cxx, cxy, 0,
                cxy, cyy, 0,
                0,   0,   0
            );
            // Eigen decomposition for 2x2 manually
            const trace = cxx + cyy;
            const det = cxx * cyy - cxy * cxy;
            const tmp = Math.max(trace*trace/4 - det, 0);
            const lambda1 = trace/2 + Math.sqrt(tmp);
            const lambda2 = trace/2 - Math.sqrt(tmp);
            // Principal axis for lambda1
            let vx = 1, vy = 0;
            if (Math.abs(cxy) > 1e-9) {
                vx = lambda1 - cyy;
                vy = cxy;
            }
            const norm = Math.hypot(vx, vy);
            if (norm > 1e-12) { vx /= norm; vy /= norm; }
            // Angle of principal axis
            const angle = Math.atan2(vy, vx);
            // 1-sigma ellipse radii
            const r1 = Math.sqrt(Math.max(lambda1, 0));
            const r2 = Math.sqrt(Math.max(lambda2, 0));
            // Create filled ellipse using a circle geometry scaled to eigen radii
            const segments = 64;
            const geom = new THREE.CircleGeometry(1, segments);
            const color = new THREE.Color(0x39ff14); // neon green
            const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.01, side: THREE.DoubleSide, depthWrite: false });
            const ellipseMesh = new THREE.Mesh(geom, material);
            ellipseMesh.scale.set(r1, r2, 1);
            ellipseMesh.position.set(x, y, z);
            ellipseMesh.rotation.z = angle; // rotate in XY plane
            this.scene.add(ellipseMesh);
            this.covEllipses.push(ellipseMesh);
        }
    }

    startCovarianceUpdate() {
        this.fetchCovariance();
        this.covUpdateInterval = setInterval(() => this.fetchCovariance(), 1000);
    }

    stopCovarianceUpdate() {
        if (this.covUpdateInterval) {
            clearInterval(this.covUpdateInterval);
            this.covUpdateInterval = null;
        }
    }
    
    stopAutoUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        const width = window.innerWidth * 0.75;  // 75% width for 3D view
        this.camera.aspect = width / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, window.innerHeight);
    }
    
    dispose() {
        this.stopAutoUpdate();
        this.stopAnimation();
        this.stopSensorUpdate();
        
        if (this.trajectoryLine) {
            this.trajectoryLine.geometry.dispose();
            this.trajectoryLine.material.dispose();
        }
        
        this.renderer.dispose();
        this.controls.dispose();
    }
    
    // Sensor data fetching and display
    async fetchSensorData() {
        try {
            const response = await fetch('/sensors');
            if (!response.ok) {
                return;  // Silently fail if endpoint doesn't exist yet
            }
            
            const text = await response.text();
            if (text.trim().length > 0) {
                this.parseSensorData(text);
            }
        } catch (error) {
            // Silently fail - sensor endpoint may not be implemented yet
        }
    }
    
    parseSensorData(text) {
        const lines = text.trim().split('\n');
        if (lines.length === 0) return;
        
        // Get the last line (most recent sensor data)
        const lastLine = lines[lines.length - 1];
        const values = lastLine.split(',').map(s => s.trim());
        
        // Format: timestamp,ax,ay,az,wx,wy,wz,vx,vy,vz,depth
        if (values.length >= 11) {
            const timestamp = parseFloat(values[0]);
            
            // Set start time on first data
            if (this.startTime === null) {
                this.startTime = timestamp;
            }
            
            const relativeTime = timestamp - this.startTime;
            
            // Add data to arrays
            this.sensorTimeData.push(relativeTime);
            this.imuAccelData.ax.push(parseFloat(values[1]));
            this.imuAccelData.ay.push(parseFloat(values[2]));
            this.imuAccelData.az.push(parseFloat(values[3]));
            this.imuGyroData.wx.push(parseFloat(values[4]));
            this.imuGyroData.wy.push(parseFloat(values[5]));
            this.imuGyroData.wz.push(parseFloat(values[6]));
            this.dvlVelData.vx.push(parseFloat(values[7]));
            this.dvlVelData.vy.push(parseFloat(values[8]));
            this.dvlVelData.vz.push(parseFloat(values[9]));
            this.baroDepthData.push(parseFloat(values[10]));  // Already negative from backend
            
            // Keep only last maxDataPoints
            if (this.sensorTimeData.length > this.maxDataPoints) {
                this.sensorTimeData.shift();
                this.imuAccelData.ax.shift();
                this.imuAccelData.ay.shift();
                this.imuAccelData.az.shift();
                this.imuGyroData.wx.shift();
                this.imuGyroData.wy.shift();
                this.imuGyroData.wz.shift();
                this.dvlVelData.vx.shift();
                this.dvlVelData.vy.shift();
                this.dvlVelData.vz.shift();
                this.baroDepthData.shift();
            }
            
            // Update charts
            this.updateCharts();
        }
    }
    
    updateCharts() {
        // Update IMU Acceleration Chart
        this.imuAccelChart.data.labels = this.sensorTimeData.map(t => t.toFixed(2));
        this.imuAccelChart.data.datasets[0].data = this.imuAccelData.ax;
        this.imuAccelChart.data.datasets[1].data = this.imuAccelData.ay;
        this.imuAccelChart.data.datasets[2].data = this.imuAccelData.az;
        this.imuAccelChart.update('none');
        
        // Update IMU Gyro Chart
        this.imuGyroChart.data.labels = this.sensorTimeData.map(t => t.toFixed(2));
        this.imuGyroChart.data.datasets[0].data = this.imuGyroData.wx;
        this.imuGyroChart.data.datasets[1].data = this.imuGyroData.wy;
        this.imuGyroChart.data.datasets[2].data = this.imuGyroData.wz;
        this.imuGyroChart.update('none');
        
        // Update DVL Velocity Chart
        this.dvlVelChart.data.labels = this.sensorTimeData.map(t => t.toFixed(2));
        this.dvlVelChart.data.datasets[0].data = this.dvlVelData.vx;
        this.dvlVelChart.data.datasets[1].data = this.dvlVelData.vy;
        this.dvlVelChart.data.datasets[2].data = this.dvlVelData.vz;
        this.dvlVelChart.update('none');
        
        // Update Barometer Chart
        this.baroChart.data.labels = this.sensorTimeData.map(t => t.toFixed(2));
        this.baroChart.data.datasets[0].data = this.baroDepthData;
        this.baroChart.update('none');
    }
    
    startSensorUpdate() {
        // Fetch sensor data every 100ms for smooth updates
        this.sensorUpdateInterval = setInterval(() => {
            this.fetchSensorData();
        }, 100);
    }
    
    stopSensorUpdate() {
        if (this.sensorUpdateInterval) {
            clearInterval(this.sensorUpdateInterval);
            this.sensorUpdateInterval = null;
        }
    }
    
    // Simple hash function to detect duplicate data
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return hash;
    }
}

// Initialize viewer when DOM is ready
try {
    document.getElementById('info').textContent = 'Initializing Three.js...';
    window.viewer = new TrajectoryViewer();
    console.log('Trajectory viewer initialized');
} catch (error) {
    console.error('Failed to initialize viewer:', error);
    document.getElementById('info').textContent = `Error: ${error.message}`;
}

