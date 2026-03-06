import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import time
import asyncio
import math
import numpy as np
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage

# --- Config ---
# User to provide the specific depth camera topic name here
DEPTH_TOPIC = "/rgbd_image/depth_image"  # Example topic name
SAFE_DISTANCE = 2.0  # Meters
FORWARD_SPEED = 0.5  # m/s

# Global variables
latest_depth_img = None
current_pos = None

async def position_observer(drone):
    global current_pos
    async for pos in drone.telemetry.position():
        current_pos = pos

def calculate_distance(pos1, pos2):
    if pos1 is None or pos2 is None:
        return float('inf')
    # Simple Euclidean distance approx for small local changes (ignoring curvature for small steps)
    # 1 deg lat ~ 111km. 
    # But easier to just use NED if available OR just use GPS diff approach
    lat_diff_m = (pos1.latitude_deg - pos2.latitude_deg) * 111139
    lon_diff_m = (pos1.longitude_deg - pos2.longitude_deg) * 111139 * math.cos(math.radians(pos1.latitude_deg))
    return math.sqrt(lat_diff_m**2 + lon_diff_m**2)

def depth_callback(msg):
    global latest_depth_img
    # print("Callback triggered") # Debug
    try:
        # Depth images are typically Float32
        # Note: Depending on the plugin, this might be uint16 (mm) or float32 (m).
        # Standard Gz depth is float32.
        # We assume the data stream corresponds to the full image size.
        # If float32, bytes per pixel is 4.
        # Determine dtype based on bytes per pixel
        # msg.step is the number of bytes per row
        bpp = msg.step // msg.width
        
        if bpp == 4:
            dtype = np.float32
        elif bpp == 2:
            dtype = np.uint16
        else:
            # Fallback/Guess (most Gz depth is float32)
            dtype = np.float32

        img_array = np.frombuffer(msg.data, dtype=dtype)
        latest_depth_img = img_array.reshape((msg.height, msg.width))
        
    except Exception as e:
        print(f"Error processing depth image: {e}")
        # pass

async def run():
    # 1. Setup Drone
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("-- Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    # 2. Setup Gazebo Node for Depth Camera
    node = Node()
    if node.subscribe(GzImage, DEPTH_TOPIC, depth_callback):
        print(f"-- Subscribed to {DEPTH_TOPIC}")
    else:
        print(f"-- Failed to subscribe to {DEPTH_TOPIC}")

    # 3. Arm and Takeoff
    # Start position observer
    asyncio.create_task(position_observer(drone))
    
    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(8)  # Wait for takeoff to complete (approx 2.5m alt)

    # 4. Start Offboard Mode
    print("-- Starting Offboard")
    initial_setpoint = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    await drone.offboard.set_velocity_body(initial_setpoint)

    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed: {error}")
        return

    # 5. Ascend to 3m
    # Assuming standard takeoff is ~2.5m, we go up 0.5m.
    # To be safe and ensure we hit 3m, we'll command a climb.
    # If we want to be precise, we should use position control, but velocity is requested pattern here.
    # Let's climb at 0.5 m/s for 2 seconds -> +1m. Total ~3.5m.
    print("-- Ascending to ~3m height")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -0.5, 0.0))
    await asyncio.sleep(2)
    # Stop climb
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(1)

    # 6. Go Forward
    print("-- Going Forward")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(1.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(5) # Move forward for 5 seconds
    
    # Stop
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(1)

    # 7. Descend to 1.5m
    # We are at ~3.5m. Want 1.5m. Diff = 2.0m down.
    # Descend at 0.5 m/s for 4 seconds.
    print("-- Descending to ~1.5m")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.5, 0.0))
    await asyncio.sleep(4)
    
    # Stop
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(1)

    # 8. Yaw 90 deg Clockwise
    print("-- Yawing 90 deg Clockwise")
    # 30 deg/s for 3 seconds
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
    await asyncio.sleep(3)
    
    # Stop Yaw
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(1)

    # 9. Main Task: Move Forward with Obstacle Avoidance
    print("-- Moving Forward with Obstacle Avoidance")
    
    obstacle_first_detection_time = None
    
    # Stuck detection variables
    stuck_check_time = time.time()
    stuck_pos_ref = current_pos

    while True:
        # Check if stuck (movement < 20cm in 6s)
        if current_pos and stuck_pos_ref:
            dist_moved = calculate_distance(current_pos, stuck_pos_ref)
            time_elapsed = time.time() - stuck_check_time
            
            if time_elapsed > 6.0:
                if dist_moved < 0.2: # Less than 20cm movement in 6s
                    print(f"!! STUCK DETECTED (Moved {dist_moved:.3f}m in {time_elapsed:.1f}s). Performing yaw!")
                    
                    # Stop
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    
                    # Yaw 90 deg Clockwise
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
                    await asyncio.sleep(3)
                    
                    # Stop Yaw
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await asyncio.sleep(0.5)
                    
                    # Reset stuck reference
                    stuck_check_time = time.time()
                    stuck_pos_ref = current_pos
                    obstacle_first_detection_time = None
                    print("-- Resuming from stuck")
                    continue
                else:
                    # Moved enough, reset reference
                    stuck_check_time = time.time()
                    stuck_pos_ref = current_pos

        if latest_depth_img is not None:
            # Check center region for obstacles
            h, w = latest_depth_img.shape
            
            # center crop (10% of image center)
            cy, cx = h // 2, w // 2
            roi_h, roi_w = int(h * 0.1), int(w * 0.1)
            
            roi = latest_depth_img[cy - roi_h:cy + roi_h, cx - roi_w:cx + roi_w]
            
            # Filter out infinite/nan values (common in depth maps for "too far" or "too close")
            # In Gazebo, inf often means "beyond sensor range".
            valid_pixels = roi[np.isfinite(roi)]
            
            if len(valid_pixels) > 0:
                min_dist = np.min(valid_pixels)
                
                if min_dist < SAFE_DISTANCE:
                    print(f"!! Obstacle detected at {min_dist:.2f}m. Avoiding!")
                    
                    if obstacle_first_detection_time is None:
                        obstacle_first_detection_time = time.time()
                    
                    # Check if stuck for > 6 seconds due to obstacle prevention
                    elif time.time() - obstacle_first_detection_time > 6.0:
                        print("!! Obstacle persistent for > 6s. Performing Evasive Maneuver: Yaw 90 deg!")
                        
                        # Stop backward motion first
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                        
                        # Yaw 90 deg Clockwise (30 deg/s for 3s)
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
                        await asyncio.sleep(3)
                        
                        # Stop Yaw
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                        await asyncio.sleep(0.5)
                        
                        # Reset timer
                        obstacle_first_detection_time = None
                        # Reset stuck timer too as we just moved
                        stuck_check_time = time.time()
                        stuck_pos_ref = current_pos
                        print("-- Resuming forward motion")
                        continue

                    # Stop or move backward slightly
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-0.2, 0.0, 0.0, 0.0))
                else:
                    obstacle_first_detection_time = None
                    # Clear path, move forward slowly (Main task)
                    # print(f"Path clear ({min_dist:.2f}m). Proceeding.")
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_SPEED, 0.0, 0.0, 0.0))
            else:
                obstacle_first_detection_time = None
                # If all are inf, it implies open space (usually).
                # Move forward
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_SPEED, 0.0, 0.0, 0.0))
        else:
            print("No depth data received yet...")
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(run())
