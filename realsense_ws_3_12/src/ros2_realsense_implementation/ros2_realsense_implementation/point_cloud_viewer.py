#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np
from threading import Lock
import sys

class PointCloudVisualizer(Node):
    def __init__(self):
        super().__init__('pointcloud_visualizer')
        
        # Use reliable QoS for point cloud
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscriber for the assembled cloud map
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rtabmap/cloud_map',
            self.pointcloud_callback,
            qos_profile)
        
        # Visualization setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='RTAB-Map Point Cloud', width=1280, height=720)
        
        # Point cloud object
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        # Add coordinate frame for reference
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)
        
        # View control setup
        self.ctr = self.vis.get_view_control()
        self.ctr.set_zoom(0.5)
        
        # Thread safety
        self.lock = Lock()
        self.new_data_available = False
        self.current_points = None
        self.current_colors = None
        
        # Initial view parameters
        self.view_params = None
        
        # Timer for visualization update
        self.timer = self.create_timer(0.05, self.update_visualization)
        
        self.get_logger().info("Point cloud visualizer initialized, listening to /rtabmap/cloud_map")

    def pointcloud_callback(self, msg):
        try:
            # Read points with their colors
            gen = point_cloud2.read_points(msg, 
                                         field_names=("x", "y", "z", "rgb"), 
                                         skip_nans=True)
            
            points = []
            colors = []
            
            for p in gen:
                points.append([p[0], p[1], p[2]])
                
                # Convert RGB from packed float to separate components
                rgb = p[3]
                if not np.isnan(rgb):
                    # reinterpret float bits as uint32
                    rgb_int = int.from_bytes(np.float32(rgb).tobytes(), byteorder='little')
                    # extract RGB components
                    r = ((rgb_int >> 16) & 0xff) / 255.0
                    g = ((rgb_int >> 8) & 0xff) / 255.0
                    b = (rgb_int & 0xff) / 255.0
                    colors.append([r, g, b])
                else:
                    colors.append([1.0, 1.0, 1.0])  # Default to white if invalid
            
            with self.lock:
                if points:
                    self.current_points = np.array(points)
                    self.current_colors = np.array(colors)
                    self.new_data_available = True
                    self.get_logger().info(f"Received point cloud with {len(points)} points")
                    self.get_logger().info(f"Sample point: {points[0]}, color: {colors[0]}")
                else:
                    self.get_logger().warn("Received empty point cloud")
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def update_visualization(self):
        try:
            if not self.new_data_available:
                self.vis.poll_events()
                self.vis.update_renderer()
                return
            
            with self.lock:
                if self.current_points is not None and self.current_colors is not None:
                    # Update point cloud data
                    self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
                    self.pcd.colors = o3d.utility.Vector3dVector(self.current_colors)
                    
                    # Print debug info
                    self.get_logger().info(f"Updating visualization with {len(self.current_points)} points")
                    
                    # Reset the view to look at the new point cloud
                    if len(self.current_points) > 0:
                        self.vis.update_geometry(self.pcd)
                        
                        # Set reasonable view if we don't have saved parameters
                        if self.view_params is None:
                            self.ctr.set_front([0, 0, -1])
                            self.ctr.set_lookat(self.pcd.get_center())
                            self.ctr.set_up([0, -1, 0])
                            self.ctr.set_zoom(0.5)
                        else:
                            # Restore previous view
                            self.ctr.convert_from_pinhole_camera_parameters(self.view_params)
                    
                    # Save current view parameters
                    self.view_params = self.ctr.convert_to_pinhole_camera_parameters()
                
                self.new_data_available = False
            
            self.vis.poll_events()
            self.vis.update_renderer()
        except Exception as e:
            self.get_logger().error(f"Error in visualization update: {str(e)}")

    def destroy_node(self):
        self.vis.destroy_window()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    visualizer = PointCloudVisualizer()
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()