#/bin/bash
#
ffmpeg -framerate 5 -pattern_type glob -i 'ani1/*.png' \
  -c:v libx264 -pix_fmt yuv420p out.mp4
