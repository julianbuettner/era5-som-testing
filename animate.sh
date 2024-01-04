#/bin/bash
#
ffmpeg -framerate 15 -pattern_type glob -i '/mnt/d/som/ani1/*.png' \
  -c:v libx264 -pix_fmt yuv420p /mnt/d/som/out.mp4
