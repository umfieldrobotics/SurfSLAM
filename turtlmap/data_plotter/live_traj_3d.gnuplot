# Live 3D trajectory viewer using gnuplot
# Usage: gnuplot -persist live_traj_3d.gnuplot
# It continuously reloads the trajectory_live.txt file and replots.

# Configuration
trajfile = "trajectory_live.txt"

set title "Live Trajectory"
set xlabel "X"
set ylabel "Y"
set zlabel "Z"
set ticslevel 0
set grid
set key off
set view 60, 30, 1, 1

# Adjust ranges automatically
unset xrange
unset yrange
unset zrange

splot trajfile using 1:2:3 with lines lt rgb "#1f77b4" lw 2
pause 0.25
reread
