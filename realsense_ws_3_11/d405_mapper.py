import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from datetime import datetime
from collections import deque

# ===== Configuration =====
VOXEL_SIZE = 0.01           # Downsample voxel size (meters)
OUTLIER_NEIGHBORS = 20      # Statistical outlier removal
OUTLIER_STD_RATIO = 1.0     # Aggressiveness of outlier removal
MAX_FRAMES = 20             # Only keep this many recent frames
DEPTH_COLOR_SCALE = 0.1     # Scale for depth visualization

# Mouse callback variables
mouse_x, mouse_y = 0, 0
show_depth = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, show_depth
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        show_depth = not show_depth

# ===== Initialize RealSense =====
pipeline = rs.pipeline()
config = rs.config()

# Enable streams - ensure matching resolutions
WIDTH, HEIGHT = 640, 480
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

# Start pipeline
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy preset

# Get depth scale
depth_scale = depth_sensor.get_depth_scale()

# Align depth to color
align = rs.align(rs.stream.color)

# ===== Define Filters =====
decimation = rs.decimation_filter()
spatial = rs.spatial_filter(0.5, 20.0, 2.0, 0.0)
temporal = rs.temporal_filter(0.4, 20.0, 3)

# ===== 3D Reconstruction Setup =====
pointcloud_deque = deque(maxlen=MAX_FRAMES)  # Stores only recent frames
vis_3d = o3d.visualization.Visualizer()
vis_3d.create_window("3D Reconstruction", width=800, height=600)

# Create OpenCV window
cv2.namedWindow("Depth Overlay")
cv2.setMouseCallback("Depth Overlay", mouse_callback)

# Store the viewpoint parameters
view_params = None

try:
    frame_count = 0
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        
        # Align depth to color
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Warning: Missing frames")
            continue

        # Apply filters
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create depth colormap
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=DEPTH_COLOR_SCALE),
            cv2.COLORMAP_JET
        )

        # Ensure dimensions match
        if depth_colormap.shape != color_image.shape:
            depth_colormap = cv2.resize(depth_colormap, 
                                      (color_image.shape[1], color_image.shape[0]))

        # Create overlay
        overlay = cv2.addWeighted(color_image, 0.5, depth_colormap, 0.5, 0)

        # Show depth under mouse
        if show_depth and 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
            depth_mm = depth_image[mouse_y, mouse_x]
            cv2.circle(overlay, (mouse_x, mouse_y), 5, (0, 255, 0), -1)
            cv2.putText(overlay, f"{depth_mm/10:.1f} cm", 
                       (mouse_x+10, mouse_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display overlay
        cv2.imshow("Depth Overlay", overlay)

        # Process point cloud with color
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        
        vtx = np.asanyarray(points.get_vertices())
        if vtx.size == 0:
            print("Warning: Empty point cloud")
            continue
            
        vtx = vtx.view(np.float32).reshape(-1, 3)
        clr = np.asanyarray(color_frame.get_data()).reshape(-1, 3)/255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        pcd.colors = o3d.utility.Vector3dVector(clr)

        # Downsample and clean
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        if len(pcd.points) < OUTLIER_NEIGHBORS:
            continue
            
        pcd, _ = pcd.remove_statistical_outlier(OUTLIER_NEIGHBORS, OUTLIER_STD_RATIO)

        # Add to recent frames deque
        pointcloud_deque.append(pcd)

        # Rebuild global map from recent frames only
        global_map = o3d.geometry.PointCloud()
        for recent_pcd in pointcloud_deque:
            global_map += recent_pcd

        # Voxelize the combined map
        global_map = global_map.voxel_down_sample(VOXEL_SIZE)

        # Update visualization only if we have points
        if len(global_map.points) > 0:
            # First frame initialization
            if frame_count == 0:
                vis_3d.add_geometry(global_map)
            else:
                # Get current viewpoint before update
                if view_params is None:
                    ctr = vis_3d.get_view_control()
                    view_params = ctr.convert_to_pinhole_camera_parameters()
                
                vis_3d.clear_geometries()
                vis_3d.add_geometry(global_map)
                
                # Restore viewpoint if we had one
                if view_params is not None:
                    ctr = vis_3d.get_view_control()
                    ctr.convert_from_pinhole_camera_parameters(view_params)

        # Update 3D view
        vis_3d.poll_events()
        vis_3d.update_renderer()

        # Exit on ESC
        key = cv2.waitKey(1)
        if key == 27:
            break

        frame_count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    vis_3d.destroy_window()

    # Save reconstruction
    if 'global_map' in locals() and len(global_map.points) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        o3d.io.write_point_cloud(f"reconstruction_{timestamp}.ply", global_map)
        print(f"Saved 3D reconstruction to reconstruction_{timestamp}.ply")
    else:
        print("Error: Empty point cloud - nothing to save")