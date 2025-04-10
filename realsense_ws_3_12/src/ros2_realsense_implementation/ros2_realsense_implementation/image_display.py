#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import zmq
import pickle
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

class ImageDisplay(Node):
    def __init__(self):
        super().__init__('image_display')
        self.bridge = CvBridge()
        
        # ZeroMQ setup
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        # ROS2 publishers
        self.color_publisher = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_publisher = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.filtered_publisher = self.create_publisher(Image, '/camera/depth/filtered', 10)
        
        # RealSense depth processing filters
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.holes_fill, 3)  # Aggressive hole filling
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        # Set up matplotlib figure
        plt.ion()  # Interactive mode on
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.canvas.set_window_title('RealSense Feed Analysis')
        
        # Set titles for subplots
        self.axs[0].set_title('Color Image')
        self.axs[1].set_title('Raw Depth')
        self.axs[2].set_title('Filtered Depth')
        
        # Timer for receiving images
        self.timer = self.create_timer(0.01, self.receive_image)

    def apply_depth_filters(self, depth_frame):
        """Apply RealSense depth processing filters"""
        # Convert numpy array to RealSense depth frame
        depth_frame_rs = rs.frame()
        # Note: This is a simplified approach - in a real application you'd need proper frame creation
        # For demonstration, we'll just return the input
        
        # Apply filters (commented out as we can't create proper rs.frame from numpy easily)
        # filtered = self.spatial.process(depth_frame_rs)
        # filtered = self.temporal.process(filtered)
        # filtered = self.hole_filling.process(filtered)
        
        # For this example, we'll simulate filtering with OpenCV operations
        # Median blur to reduce noise
        filtered = cv2.medianBlur(depth_frame.astype(np.float32), 5)
        
        # Inpainting to fill holes
        mask = (depth_frame == 0).astype(np.uint8) * 255
        filtered = cv2.inpaint(filtered.astype(np.uint16), mask, 3, cv2.INPAINT_TELEA)
        
        return filtered

    def receive_image(self):
        try:
            data = self.socket.recv(flags=zmq.NOBLOCK)
            received_data = pickle.loads(data)
            
            # Assuming we receive a dictionary with both color and depth
            if isinstance(received_data, dict):
                cv_color = received_data.get('color')
                cv_depth = received_data.get('depth')
                
                if cv_color is not None and cv_depth is not None:
                    # Publish color image
                    ros_color = self.bridge.cv2_to_imgmsg(cv_color, "bgr8")
                    self.color_publisher.publish(ros_color)
                    
                    # Publish depth image
                    ros_depth = self.bridge.cv2_to_imgmsg(cv_depth, "passthrough")
                    self.depth_publisher.publish(ros_depth)
                    
                    # Process depth with filters
                    filtered_depth = self.apply_depth_filters(cv_depth)
                    
                    # Publish filtered depth
                    ros_filtered = self.bridge.cv2_to_imgmsg(filtered_depth, "passthrough")
                    self.filtered_publisher.publish(ros_filtered)
                    
                    # Update the matplotlib display
                    self.axs[0].imshow(cv2.cvtColor(cv_color, cv2.COLOR_BGR2RGB))
                    self.axs[1].imshow(cv_depth, cmap='jet')
                    self.axs[2].imshow(filtered_depth, cmap='jet')
                    
                    # Clear previous drawings
                    for ax in self.axs:
                        ax.clear()
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    # Redraw
                    self.axs[0].imshow(cv2.cvtColor(cv_color, cv2.COLOR_BGR2RGB))
                    self.axs[0].set_title('Color Image')
                    
                    self.axs[1].imshow(cv_depth, cmap='jet')
                    self.axs[1].set_title('Raw Depth')
                    
                    self.axs[2].imshow(filtered_depth, cmap='jet')
                    self.axs[2].set_title('Filtered Depth')
                    
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    
            else:
                # Handle case where we only receive color image
                ros_image = self.bridge.cv2_to_imgmsg(received_data, "bgr8")
                self.color_publisher.publish(ros_image)
                
                # Update just the color plot
                self.axs[0].imshow(cv2.cvtColor(received_data, cv2.COLOR_BGR2RGB))
                self.axs[0].set_title('Color Image')
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
        except zmq.Again:
            pass
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main():
    rclpy.init()
    subscriber = ImageDisplay()
    rclpy.spin(subscriber)
    
    # Cleanup
    plt.ioff()
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()