import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import time
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage
from ultralytics import YOLO

# --- CONFIGURATION ---
DOWN_TOPIC = "/rgbd_image_down/image"
SEARCH_ALTITUDE = 3.5
PAD_MODEL_PATH = "best.pt"
DESCENT_STEP = 0.3
CENTRAL_ZONE_PX = 80
STREAK_REQUIRED = 3
REACQUIRE_TIMEOUT = 3.0
MAX_VXY = 0.6
MAX_YAW = 20.0
PIXEL_TO_VEL_GAIN = 0.002
PREDICTIVE_GAIN = 0.5

# Arena boundaries (10m x 10m, assuming camera at center when drone is centered)
ARENA_SIZE_M = 10.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CENTER = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)

# Launch pad exclusion (stationary pad near image center at start)
LAUNCH_PAD_CENTER = IMAGE_CENTER
LAUNCH_PAD_IGNORE_RADIUS = 120  # Increased radius for better exclusion

# Global State
pad_center = None
pad_conf = 0.0
rel_alt = 0.0
detection_streak = 0
last_detection_time = 0.0
last_pad_center = None
pad_velocity_px_s = (0.0, 0.0)
drone_has_moved = False  # Track if drone has moved away from launch pad
initial_position = None
current_position = None

model = YOLO(PAD_MODEL_PATH)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def is_launch_pad(cx, cy, velocity_px_s):
    """
    Determine if detected pad is the launch pad based on:
    1. Position near image center
    2. Low velocity (stationary)
    3. Drone hasn't moved significantly yet
    """
    global drone_has_moved, initial_position, current_position
    
    # Calculate distance from image center
    dist_to_center = np.linalg.norm([cx - IMAGE_CENTER[0], cy - IMAGE_CENTER[1]])
    
    # Calculate pad speed
    vx, vy = velocity_px_s
    speed_px = np.hypot(vx, vy)
    
    # Launch pad characteristics:
    # - Near center of image when drone is at launch position
    # - Stationary (low velocity)
    # - Only relevant before drone has moved significantly
    is_stationary = speed_px < 8  # pixels/second
    is_centered = dist_to_center < LAUNCH_PAD_IGNORE_RADIUS
    
    # Check if drone has moved away from initial position
    if initial_position is not None and current_position is not None:
        horizontal_movement = np.linalg.norm([
            current_position[0] - initial_position[0],
            current_position[1] - initial_position[1]
        ])
        # Consider drone has moved if >2m from start
        if horizontal_movement > 2.0:
            drone_has_moved = True
    
    # Launch pad only matters before drone has moved significantly
    if not drone_has_moved and is_stationary and is_centered:
        return True
    
    return False

def is_within_safe_bounds(cx, cy, margin_px=50):
    """
    Check if the detected pad is within safe image bounds.
    Prevents tracking pads too close to image edges which might be outside arena.
    """
    return (margin_px < cx < IMAGE_WIDTH - margin_px and 
            margin_px < cy < IMAGE_HEIGHT - margin_px)

def down_callback(msg):
    global pad_center, pad_conf, detection_streak, rel_alt
    global last_detection_time, last_pad_center, pad_velocity_px_s

    try:
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        dynamic_conf = 0.65 if rel_alt > 2.0 else 0.50
        results = model(bgr, verbose=False, conf=dynamic_conf)

        now = time.time()

        if len(results[0].boxes) > 0:
            # Find best valid detection (exclude launch pad and out-of-bounds)
            valid_detection = None
            best_conf = 0.0
            
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()
                cx = int((coords[0] + coords[2]) / 2)
                cy = int((coords[1] + coords[3]) / 2)
                conf = float(box.conf)
                
                # Calculate temporary velocity for this detection
                temp_velocity = (0.0, 0.0)
                if last_pad_center is not None and last_detection_time > 0:
                    dt = now - last_detection_time
                    if dt > 0:
                        temp_velocity = ((cx - last_pad_center[0]) / dt,
                                       (cy - last_pad_center[1]) / dt)
                
                # Skip if it's the launch pad
                if is_launch_pad(cx, cy, temp_velocity):
                    continue
                
                # Skip if too close to image edges (likely outside arena)
                if not is_within_safe_bounds(cx, cy):
                    continue
                
                # Choose detection with highest confidence
                if conf > best_conf:
                    best_conf = conf
                    valid_detection = (cx, cy, conf)
            
            if valid_detection is not None:
                cx, cy, conf = valid_detection
                
                # Update velocity tracking
                if last_pad_center is not None and last_detection_time > 0:
                    dt = now - last_detection_time
                    if dt > 0:
                        vx = (cx - last_pad_center[0]) / dt
                        vy = (cy - last_pad_center[1]) / dt
                        # Smooth velocity with exponential moving average
                        alpha = 0.3
                        pad_velocity_px_s = (
                            alpha * vx + (1 - alpha) * pad_velocity_px_s[0],
                            alpha * vy + (1 - alpha) * pad_velocity_px_s[1]
                        )
                else:
                    pad_velocity_px_s = (0.0, 0.0)

                last_pad_center = (cx, cy)
                last_detection_time = now

                pad_center = (cx, cy)
                pad_conf = conf
                detection_streak = min(detection_streak + 1, STREAK_REQUIRED + 5)
            else:
                # No valid detection found
                if detection_streak > 0:
                    detection_streak -= 1
                pad_conf = 0.0
                if time.time() - last_detection_time > 1.0:
                    pad_center = None
                    last_pad_center = None
                    pad_velocity_px_s = (0.0, 0.0)
        else:
            # No detections at all
            if detection_streak > 0:
                detection_streak -= 1
            pad_conf = 0.0
            if time.time() - last_detection_time > 1.0:
                pad_center = None
                last_pad_center = None
                pad_velocity_px_s = (0.0, 0.0)

        # Visualization
        cv2.circle(bgr, IMAGE_CENTER, CENTRAL_ZONE_PX, (255, 0, 0), 2)
        cv2.circle(bgr, LAUNCH_PAD_CENTER, LAUNCH_PAD_IGNORE_RADIUS, (128, 128, 128), 1)
        
        # Draw safe bounds
        margin = 50
        cv2.rectangle(bgr, (margin, margin), 
                     (IMAGE_WIDTH - margin, IMAGE_HEIGHT - margin), 
                     (255, 255, 0), 1)
        
        if pad_center:
            cv2.circle(bgr, pad_center, 6, (0, 0, 255), -1)
            cv2.putText(bgr, f"Vel: {pad_velocity_px_s[0]:.1f}, {pad_velocity_px_s[1]:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        status = "MOVED" if drone_has_moved else "AT LAUNCH"
        cv2.putText(bgr, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Arena-Locked Tracker", bgr)
        cv2.waitKey(1)
    except Exception as e:
        print(f"Callback error: {e}")

async def perform_iterative_descent(drone):
    global pad_center, rel_alt, detection_streak, pad_velocity_px_s

    while True:
        if rel_alt <= 0.15:
            await drone.action.land()
            return

        # Align loop
        while True:
            if pad_center is None or detection_streak < STREAK_REQUIRED:
                t0 = time.time()
                while time.time() - t0 < REACQUIRE_TIMEOUT:
                    if pad_center is not None and detection_streak >= STREAK_REQUIRED:
                        break
                    await asyncio.sleep(0.05)
                if pad_center is None or detection_streak < STREAK_REQUIRED:
                    return

            img_cx, img_cy = IMAGE_CENTER
            err_x = pad_center[0] - img_cx
            err_y = img_cy - pad_center[1]

            vx_px, vy_px = pad_velocity_px_s
            err_x_pred = err_x + PREDICTIVE_GAIN * vx_px
            err_y_pred = err_y + PREDICTIVE_GAIN * vy_px

            vx_cmd = clamp(err_y_pred * PIXEL_TO_VEL_GAIN, -MAX_VXY, MAX_VXY)
            vy_cmd = clamp(err_x_pred * PIXEL_TO_VEL_GAIN, -MAX_VXY, MAX_VXY)

            if abs(err_x) <= CENTRAL_ZONE_PX and abs(err_y) <= CENTRAL_ZONE_PX:
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
                break
            else:
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx_cmd, vy_cmd, 0, 0))

            await asyncio.sleep(0.08)

        await asyncio.sleep(0.8)

        if pad_center is None or detection_streak < STREAK_REQUIRED:
            return

        target_alt = max(rel_alt - DESCENT_STEP, 0.0)
        v_down = 0.3
        while rel_alt > target_alt + 0.02 and rel_alt > 0.12:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, v_down, 0))
            await asyncio.sleep(0.1)

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
        await asyncio.sleep(0.8)

async def run_mission():
    global rel_alt, detection_streak, initial_position, current_position
    drone = System()
    await drone.connect(system_address="udp://:14540")

    async def get_telem():
        global rel_alt, initial_position, current_position
        async for p in drone.telemetry.position():
            rel_alt = p.relative_altitude_m
            current_position = (p.latitude_deg, p.longitude_deg, p.absolute_altitude_m)
            if initial_position is None:
                initial_position = current_position
    asyncio.create_task(get_telem())

    await drone.action.arm()
    await drone.action.set_takeoff_altitude(SEARCH_ALTITUDE)
    await drone.action.takeoff()
    while rel_alt < (SEARCH_ALTITUDE - 0.3):
        await asyncio.sleep(0.3)

    # Move away from launch pad by ~2m to clearly separate from landing pad
    print("Moving away from launch pad...")
    for _ in range(40):  # 0.5 m/s * 4s ≈ 2m
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.5, 0, 0, 0))
        await asyncio.sleep(0.1)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(1.5)

    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
        await drone.offboard.start()
    except Exception:
        return

    search_timer = 0
    print("Starting search pattern...")
    while True:
        if detection_streak >= STREAK_REQUIRED and pad_center is not None:
            print("Valid landing pad detected, starting descent...")
            await perform_iterative_descent(drone)
            if rel_alt <= 0.15:
                break

        search_timer += 1
        if search_timer < 40:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.4, 0, 0, 0))
        elif search_timer < 55:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, MAX_YAW))
        else:
            search_timer = 0

        await asyncio.sleep(0.1)

async def main():
    node = Node()
    node.subscribe(GzImage, DOWN_TOPIC, down_callback)
    await run_mission()

if __name__ == "__main__":
    asyncio.run(main())
