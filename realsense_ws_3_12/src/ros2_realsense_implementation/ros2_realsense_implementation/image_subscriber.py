#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import zmq
import cv2
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        
        # ZeroMQ setup
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        # ROS2 publishers
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.filtered_pub = self.create_publisher(Image, '/camera/depth/filtered', 10)
        
        # Display setup
        self.window_name = "RealSense Feed - Color | Raw Depth | Filtered Depth"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 540)

    def receive_image(self):
        try:
            data = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            
            # Publish to ROS
            self.color_pub.publish(self.bridge.cv2_to_imgmsg(data['color'], "bgr8"))
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(data['depth'], "mono16"))
            self.filtered_pub.publish(self.bridge.cv2_to_imgmsg(data['filtered'], "mono16"))
            
            # Create visualizations
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(data['depth'], alpha=0.03),
                cv2.COLORMAP_JET
            )
            filtered_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(data['filtered'], alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Combine images
            combined = np.hstack((data['color'], depth_colormap, filtered_colormap))
            cv2.imshow(self.window_name, combined)
            cv2.waitKey(1)
            
        except zmq.Again:
            pass

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    subscriber = ImageSubscriber()
    subscriber.create_timer(0.001, subscriber.receive_image)
    
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()