import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import time
import asyncio
import math
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage
from ultralytics import YOLO

# --- MISSION CONFIGURATION ---
CAMERA_TOPIC = "/camera/image_raw" 
DEPTH_TOPIC = "/rgbd_image/depth_image"
DOWN_TOPIC = "/rgbd_image_down/image"

TARGET_ALTITUDE = 1.5
SEARCH_ALTITUDE = 3.5
IMG_CENTER_X = 320
IMG_CENTER_Y = 240
IMG_WIDTH = 640
IMG_HEIGHT = 480

# Detection parameters
MIN_CONTOUR_AREA = 500
CENTER_TOLERANCE_PX = 10

# Control parameters
SAFE_DISTANCE = 2.0
FORWARD_SPEED = 0.5

# HSV Blue Range
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

# Global variables - Mission 1 (Window)
window_detected = False
window_center_x = 0
rel_altitude = 0.0

# Global variables - Mission 2 (Depth/Obstacle)
latest_depth_img = None
current_pos = None

# Global variables - Mission 4 (H-Pad)
pad_center = None
pad_conf = 0.0

# YOLO Model for H-Pad
PAD_MODEL_PATH = "best.pt"
model = YOLO(PAD_MODEL_PATH)


# ==================== CALLBACKS ====================

def front_camera_callback(msg):
    """Front camera callback for green window detection"""
    global window_detected, window_center_x
    
    try:
        frame_bytes = np.frombuffer(msg.data, dtype=np.uint8)
        img = frame_bytes.reshape((msg.height, msg.width, 3))
        bgr_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        window_detected = False
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > MIN_CONTOUR_AREA:
                window_detected = True
                x, y, w, h = cv2.boundingRect(largest_contour)
                window_center_x = x + w // 2
                
                cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.circle(bgr_frame, (window_center_x, y + h // 2), 8, (0, 0, 255), -1)
        
        status = "DETECTED" if window_detected else "SEARCHING"
        cv2.putText(bgr_frame, status, (IMG_WIDTH - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if window_detected else (0, 255, 255), 2)
        
        cv2.imshow("IMAV - Blue Window Tracker", bgr_frame)
        cv2.waitKey(1)
        
    except Exception as e:
        print(f"Front camera error: {e}")


def depth_callback(msg):
    """Depth camera callback for obstacle avoidance"""
    global latest_depth_img
    try:
        bpp = msg.step // msg.width
        dtype = np.float32 if bpp == 4 else np.uint16
        img_array = np.frombuffer(msg.data, dtype=dtype)
        latest_depth_img = img_array.reshape((msg.height, msg.width))
    except Exception as e:
        print(f"Depth camera error: {e}")


def down_camera_callback(msg):
    """Downward camera callback for H-pad detection"""
    global pad_center, pad_conf
    try:
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        results = model(bgr, verbose=False, conf=0.5)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            coords = box.xyxy[0].tolist()
            pad_center = (int((coords[0]+coords[2])/2), int((coords[1]+coords[3])/2))
            pad_conf = float(box.conf)
            cv2.rectangle(bgr, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
        else:
            pad_conf = 0.0

        cv2.imshow("IMAV - Down Camera (H-Pad)", bgr)
        cv2.waitKey(1)
    except:
        pass


# ==================== HELPER FUNCTIONS ====================

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def calculate_distance(pos1, pos2):
    if pos1 is None or pos2 is None:
        return float('inf')
    lat_diff_m = (pos1.latitude_deg - pos2.latitude_deg) * 111139
    lon_diff_m = (pos1.longitude_deg - pos2.longitude_deg) * 111139 * math.cos(math.radians(pos1.latitude_deg))
    return math.sqrt(lat_diff_m**2 + lon_diff_m**2)

async def run_mission():
    global rel_altitude, window_detected, current_pos
    global window_center_x, latest_depth_img, pad_center, pad_conf
    
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("\n⏳ Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected: 
            print("✓ Drone connected!\n")
            break

    # Telemetry background tasks
    async def get_height():
        global rel_altitude
        async for pos in drone.telemetry.position():
            rel_altitude = pos.relative_altitude_m
    asyncio.create_task(get_height())
    
    async def get_position():
        global current_pos
        async for pos in drone.telemetry.position():
            current_pos = pos
    asyncio.create_task(get_position())

    # ==============================================================
    # MISSION 1: BLUE WINDOW TRAVERSAL
    # ==============================================================
    print("\n" + "="*60)
    print("  MISSION 1: BLUE WINDOW TRAVERSAL")
    print("="*60)

    # ========== PHASE 1: TAKEOFF ==========
    print(f"[PHASE 1] 🚁 Taking off to {TARGET_ALTITUDE}m")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(TARGET_ALTITUDE)
    await drone.action.takeoff()
    
    while rel_altitude < (TARGET_ALTITUDE - 0.2):
        print(f"  ↑ Altitude: {rel_altitude:.2f}m", end='\r')
        await asyncio.sleep(0.3)
    print(f"\n✓ Reached {rel_altitude:.2f}m\n")
    await asyncio.sleep(1.0)

    # ========== PHASE 2: OFFBOARD MODE ==========
    print("[PHASE 2] 🎮 Enabling offboard control")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try: 
        await drone.offboard.start()
        print("✓ Offboard mode active\n")
    except OffboardError as e:
        print(f"✗ Offboard failed: {e}")
        return

    # ========== PHASE 3: DRIFT RIGHT TO FIND WINDOW ==========
    print("[PHASE 3] ➡ Drifting Right to find and center window")
    
    drift_speed = 0.2
    
    while True:
        if window_detected and abs(window_center_x - IMG_CENTER_X) < CENTER_TOLERANCE_PX:
             print(f"\n✓ Window Centered! Error: {window_center_x - IMG_CENTER_X}")
             break
             
        vy = 0.0
        
        if not window_detected:
            vy = drift_speed
            print(f"  Drifting right... (Searching)", end='\r')
        else:
            error_x = window_center_x - IMG_CENTER_X
            vy = clamp(error_x * 0.002, -0.4, 0.4)
            print(f"  Centering... Error: {error_x}px, Vy: {vy:.2f}", end='\r')

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, vy, 0, 0))
        await asyncio.sleep(0.1)

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(1.0)

    # ========== PHASE 4: TRAVERSE WINDOW ==========
    print("[PHASE 4] ➜ Traversing window slowly")
    
    TRAVERSE_SPEED = 0.8
    DURATION_SECONDS = 12.0
    
    print(f"  Moving forward at {TRAVERSE_SPEED} m/s")
    
    steps = int(DURATION_SECONDS / 0.1)
    for i in range(steps):
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(TRAVERSE_SPEED, 0, 0, 0))
        await asyncio.sleep(0.1)
        if i % 10 == 0:
             print(f"  Traversing... {i*0.1:.1f}s")
    
    print("\n✓ Window traversal complete!\n")
    
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(1.0)

    # ==============================================================
    # MISSION 2: OBSTACLE AVOIDANCE NAVIGATION
    # ==============================================================
    print("\n" + "="*60)
    print("  MISSION 2: OBSTACLE AVOIDANCE NAVIGATION")
    print("="*60)

    # ========== PHASE 5: FORWARD WITH DEPTH ==========
    print("[PHASE 5] 🚶 Moving forward with obstacle avoidance")
    
    obstacle_first_detection_time = None
    stuck_check_time = time.time()
    stuck_pos_ref = current_pos
    
    mission2_duration = 30.0  # Run for 30 seconds
    mission2_start = time.time()
    
    while time.time() - mission2_start < mission2_duration:
        # Check if stuck
        if current_pos and stuck_pos_ref:
            dist_moved = calculate_distance(current_pos, stuck_pos_ref)
            time_elapsed = time.time() - stuck_check_time
            
            if time_elapsed > 6.0:
                if dist_moved < 0.2:
                    print(f"\n!! STUCK DETECTED. Performing yaw!")
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
                    await asyncio.sleep(3)
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                    await asyncio.sleep(0.5)
                    stuck_check_time = time.time()
                    stuck_pos_ref = current_pos
                    obstacle_first_detection_time = None
                    continue
                else:
                    stuck_check_time = time.time()
                    stuck_pos_ref = current_pos

        if latest_depth_img is not None:
            h, w = latest_depth_img.shape
            cy, cx = h // 2, w // 2
            roi_h, roi_w = int(h * 0.1), int(w * 0.1)
            roi = latest_depth_img[cy - roi_h:cy + roi_h, cx - roi_w:cx + roi_w]
            valid_pixels = roi[np.isfinite(roi)]
            
            if len(valid_pixels) > 0:
                min_dist = np.min(valid_pixels)
                
                if min_dist < SAFE_DISTANCE:
                    print(f"  !! Obstacle at {min_dist:.2f}m", end='\r')
                    
                    if obstacle_first_detection_time is None:
                        obstacle_first_detection_time = time.time()
                    elif time.time() - obstacle_first_detection_time > 6.0:
                        print("\n!! Obstacle persistent. Yaw 90 deg!")
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
                        await asyncio.sleep(3)
                        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
                        await asyncio.sleep(0.5)
                        obstacle_first_detection_time = None
                        stuck_check_time = time.time()
                        stuck_pos_ref = current_pos
                        continue

                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-0.2, 0.0, 0.0, 0.0))
                else:
                    obstacle_first_detection_time = None
                    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_SPEED, 0.0, 0.0, 0.0))
            else:
                obstacle_first_detection_time = None
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_SPEED, 0.0, 0.0, 0.0))
        else:
            print("  No depth data...", end='\r')
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_SPEED, 0.0, 0.0, 0.0))
            
        await asyncio.sleep(0.1)
    
    print("\n✓ Obstacle avoidance phase complete!\n")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(1.0)

    # ==============================================================
    # MISSION 4: H-PAD SEARCH AND LANDING
    # ==============================================================
    print("\n" + "="*60)
    print("  MISSION 4: H-PAD SEARCH AND LANDING")
    print("="*60)

    # ========== PHASE 6: SEARCH AND LAND ON H-PAD ==========
    print("[PHASE 6] 🔍 Searching for H-Pad (Lawnmower pattern)")
    
    step_timer = 0
    
    while True:
        # TARGET FOUND: Tracking Logic
        if pad_conf > 0.6:
            print(f"\n🎯 H-Pad Detected! Conf: {pad_conf:.2f}")
            
            vx = (240 - pad_center[1]) * 0.03
            vy = (pad_center[0] - 320) * 0.03
            vz = 0.4 if abs(vx) < 0.15 and abs(vy) < 0.15 else 0.0
            
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, 0))
            
            if rel_altitude < 0.4:
                print("\n✓ Landing on H-Pad!")
                try:
                    await drone.offboard.stop()
                except:
                    pass
                await drone.action.land()
                break
        
        # SEARCHING: Lawnmower Pattern
        else:
            step_timer += 1
            if step_timer < 80:
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.8, 0, 0, 0))
            elif step_timer < 100:
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 45.0))
            else:
                step_timer = 0
                
        await asyncio.sleep(0.1)

    print("\n" + "="*60)
    print("  ALL MISSIONS COMPLETE!")
    print("="*60 + "\n")

async def main():
    gz_node = Node()
    
    # Subscribe to all cameras
    gz_node.subscribe(GzImage, CAMERA_TOPIC, front_camera_callback)
    gz_node.subscribe(GzImage, DEPTH_TOPIC, depth_callback)
    gz_node.subscribe(GzImage, DOWN_TOPIC, down_camera_callback)
    
    print("\n" + "="*70)
    print("  IMAV 2025 - COMBINED MISSION")
    print("  Mission 1: Blue Window Traversal")
    print("  Mission 2: Obstacle Avoidance Navigation")
    print("  Mission 4: H-Pad Search and Landing")
    print("="*70 + "\n")
    
    await run_mission()
    
    print("="*70)
    print("  ALL MISSIONS ENDED")
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
