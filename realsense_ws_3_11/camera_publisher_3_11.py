#!/usr/bin/env python3.11

import pyrealsense2 as rs
import numpy as np
import zmq
import time

def main():
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    
    # Create post-processing filters :cite[1]:cite[3]
    dec_filter = rs.decimation_filter(magnitude=1)  # Reduce resolution
    depth_to_disparity = rs.disparity_transform(True)
    spatial_filter = rs.spatial_filter(
        smooth_alpha=0.5,
        smooth_delta=20,
        magnitude=2,
        hole_fill=0
    )
    temporal_filter = rs.temporal_filter(
        smooth_alpha=0.4,
        smooth_delta=10,
        persistence_control=3  # Valid in 2/last 4 frames
    )
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter(mode=1)  # Farthest from around
    
    # ZeroMQ setup
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind("tcp://*:5555")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            # Apply recommended processing pipeline :cite[1]:cite[3]
            filtered = depth_frame
            filtered = dec_filter.process(filtered)
            filtered = depth_to_disparity.process(filtered)
            filtered = spatial_filter.process(filtered)
            filtered = temporal_filter.process(filtered)
            filtered = disparity_to_depth.process(filtered)
            filtered = hole_filling.process(filtered)
            
            # Get images
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            filtered_depth = np.asanyarray(filtered.get_data())
            
            # Send data
            socket.send_pyobj({
                'color': color_image,
                'depth': depth_image,
                'filtered': filtered_depth
            }, zmq.NOBLOCK)
            
    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()