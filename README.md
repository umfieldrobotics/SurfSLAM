# GLB Viewer

A 3D visualization webpage for SLAM trajectories and reconstructions.

## Structure

```
website_resources/
├── index.html              # Main webpage
├── static/
│   └── models/            # GLB model files
│       ├── boiler_combined.glb
│       ├── long_combined.glb
│       └── start_combined.glb
├── .gitignore
└── README.md
```

## Deployment to GitHub Pages

This directory is ready to be deployed as a GitHub Pages site.

### Option 1: Deploy as subdirectory of main repo

1. **Commit and push** to your repository:
   ```bash
   git add .
   git commit -m "Add GLB viewer website"
   git push
   ```

2. **Enable GitHub Pages** in repository settings:
   - Go to Settings → Pages
   - Select branch: `main`
   - Select folder: `/(root)` or `/tools/website_resources`
   - Save

3. Access at: `https://yourusername.github.io/SurfSLAM/tools/website_resources/`

### Option 2: Deploy as separate GitHub Pages repo

1. **Initialize as separate git repo**:
   ```bash
   cd /home/frog/frog_repos/SurfSLAM/tools/website_resources
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create new GitHub repo** named `surfslam-viewer` (or similar)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/yourusername/surfslam-viewer.git
   git branch -M main
   git push -u origin main
   ```

4. **Enable GitHub Pages**: Settings → Pages → Source: `main` branch → Save

5. Access at: `https://yourusername.github.io/surfslam-viewer/`

## Local Testing

```bash
cd /home/frog/frog_repos/SurfSLAM/tools/website_resources
python3 -m http.server 8000
```

Open `http://localhost:8000/` in your browser.

## Features

- Interactive 3D viewers for GLB models
- Camera trajectory visualization with frustums
- Point cloud mesh overlays
- Responsive grid layout
- Mouse controls: rotate, pan, zoom
- No build process required - pure HTML/CSS/JS

## Technical Details

- Uses Three.js (loaded from CDN)
- GLB models via GLTFLoader
- Orbit controls for interaction
- Automatic camera fitting to models

