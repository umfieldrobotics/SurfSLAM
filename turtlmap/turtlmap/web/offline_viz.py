#!/usr/bin/env python3
"""
Offline Trajectory Visualizer
Usage: python3 offline_viz.py <trajectory_file> [port]

Starts a simple HTTP server to visualize a trajectory file using viewer.js
"""

import sys
import os
import shutil
import http.server
import socketserver
import webbrowser
import signal
import subprocess
import time
from pathlib import Path

def cleanup(web_dir):
    """Remove trajectory file copy on exit"""
    traj_copy = web_dir / "traj"
    if traj_copy.exists():
        traj_copy.unlink()
        print("\n✓ Cleaned up trajectory file copy")

def free_port(port):
    """Kill any process using the specified port"""
    killed = False
    
    try:
        # Try using fuser (most reliable)
        result = subprocess.run(
            ['fuser', '-k', f'{port}/tcp'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Freed port {port}")
            killed = True
    except FileNotFoundError:
        pass
    
    if not killed:
        try:
            # Fallback: try pkill for Python http.server
            result = subprocess.run(
                ['pkill', '-f', f'python3.*http.server.*{port}'],
                capture_output=True
            )
            if result.returncode == 0:
                print(f"✓ Killed existing Python server on port {port}")
                killed = True
        except FileNotFoundError:
            pass
    
    # Give OS time to release the port
    if killed:
        time.sleep(0.5)
    
    return killed

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 offline_viz.py <trajectory_file> [port]")
        print("\nExample:")
        print("  python3 offline_viz.py ../../build/trajectory_live.txt")
        print("  python3 offline_viz.py ../../build/trajectory_live.txt 8080")
        sys.exit(1)
    
    # Parse arguments
    traj_file = Path(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    # Validate trajectory file
    if not traj_file.exists():
        print(f"✗ Error: Trajectory file not found: {traj_file}")
        sys.exit(1)
    
    # Get web directory (where this script is located)
    web_dir = Path(__file__).parent.resolve()
    
    # Validate web directory
    index_html = web_dir / "index.html"
    if not index_html.exists():
        print(f"✗ Error: index.html not found in {web_dir}")
        sys.exit(1)
    
    # Copy trajectory file to web directory as 'traj'
    traj_copy = web_dir / "traj"
    try:
        shutil.copy(traj_file, traj_copy)
        print(f"✓ Copied {traj_file} → {traj_copy}")
    except Exception as e:
        print(f"✗ Error copying trajectory file: {e}")
        sys.exit(1)
    
    # Setup cleanup on exit
    def signal_handler(sig, frame):
        print("\n\n⏹  Shutting down server...")
        cleanup(web_dir)
        free_port(port)
        print("✓ Server stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Free port if already in use
    free_port(port)
    
    # Change to web directory
    os.chdir(web_dir)
    
    # Start HTTP server
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}"
            print(f"\n{'='*60}")
            print(f"🚀 Offline Trajectory Visualizer")
            print(f"{'='*60}")
            print(f"Server:     {url}")
            print(f"Trajectory: {traj_file.name} ({traj_file.stat().st_size} bytes)")
            print(f"{'='*60}")
            print(f"\n▶  Open in browser: {url}")
            print(f"▶  Press Ctrl+C to stop\n")
            
            # Try to open browser automatically
            try:
                webbrowser.open(url)
                print("✓ Browser opened automatically")
            except:
                print("⚠  Could not open browser automatically")
            
            print(f"\nServing on port {port}...")
            httpd.serve_forever()
    
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n✗ Error: Port {port} is already in use")
            print(f"Try a different port: python3 offline_viz.py {traj_file} {port+1}")
        else:
            print(f"\n✗ Error starting server: {e}")
        cleanup(web_dir)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹  Shutting down server...")
        cleanup(web_dir)
        free_port(port)
        print("✓ Server stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()

