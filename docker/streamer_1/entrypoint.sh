#!/bin/bash
echo "[INFO] Starting video stream broadcaster..."
# Loops the video continuously and pushes it over UDP to the inference container
exec ffmpeg -stream_loop -1 -re -i /streamer/test_video.mp4 -c:v libx264 -preset ultrafast -f mpegts udp://par_inference_node:1234?pkt_size=1316