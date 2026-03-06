import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage
# from ultralytics import YOLO

# --- MISSION CONFIGURATION ---
CAMERA_TOPIC = "/camera/image_raw" 
# WINDOW_MODEL_PATH = "best_1.pt"  # Not used
TARGET_ALTITUDE = 1.5
IMG_CENTER_X = 320
IMG_CENTER_Y = 240
IMG_WIDTH = 640
IMG_HEIGHT = 480

# Detection parameters
MIN_CONTOUR_AREA = 500
APPROACH_WIDTH_PX = 200  # Stop approaching when window is this wide
ALIGNED_WIDTH_MIN = 180  # Minimum width for aligned view
ALIGNED_HEIGHT_MIN = 180  # Minimum height for aligned view
ASPECT_RATIO_ALIGNED = (0.8, 1.2)  # Square-ish when head-on
CENTER_TOLERANCE_PX = 10

# Control parameters
SEARCH_YAW_RATE = 25.0  # Clockwise search
APPROACH_SPEED = 0.5
ALIGNMENT_YAW_RATE = 20.0
YAW_GAIN = 0.15          # Proportional Gain (Kp)

# Global variables
window_detected = False
window_aligned = False
window_bbox = None  # (x1, y1, x2, y2)
window_confidence = 0.0
window_center_x = 0
window_center_y = 0
window_width = 0
window_height = 0
window_aspect_ratio = 0.0
rel_altitude = 0.0

# HSV Green Range
LOWER_GREEN = np.array([40, 100, 50])
UPPER_GREEN = np.array([80, 255, 255])

# No YOLO load
# model = YOLO(WINDOW_MODEL_PATH)


def is_window_aligned(width, height):
    """
    Check if window appears square (head-on view) based on dimensions
    """
    if width < ALIGNED_WIDTH_MIN or height < ALIGNED_HEIGHT_MIN:
        return False
    
    aspect = width / height if height > 0 else 0
    return ASPECT_RATIO_ALIGNED[0] <= aspect <= ASPECT_RATIO_ALIGNED[1]

def image_callback(msg):
    global window_detected, window_aligned, window_bbox, window_confidence
    global window_center_x, window_center_y, window_width, window_height, window_aspect_ratio
    
    try:
        # Unpack image
        frame_bytes = np.frombuffer(msg.data, dtype=np.uint8)
        img = frame_bytes.reshape((msg.height, msg.width, 3))
        bgr_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # --- COLOR DETECTION (GREEN) ---
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset detection
        window_detected = False
        window_aligned = False
        
        # Process detections
        if contours:
            # Assume largest green contour is the window
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > MIN_CONTOUR_AREA:
                window_detected = True
                window_confidence = 1.0 # Detection confident if area is big enough
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                window_bbox = (x, y, x+w, y+h)
                
                window_width = w
                window_height = h
                window_center_x = x + w // 2
                window_center_y = y + h // 2
                window_aspect_ratio = w / h if h > 0 else 0
                
                # Check alignment
                window_aligned = is_window_aligned(window_width, window_height)
                
                # === VISUALIZATION ===
                box_color = (0, 255, 0) if window_aligned else (0, 165, 255)
                cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), box_color, 3)
                cv2.circle(bgr_frame, (window_center_x, window_center_y), 8, (0, 0, 255), -1)
                
                cv2.putText(bgr_frame, "GREEN WINDOW", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
                           
                cv2.putText(bgr_frame, f"Size: {window_width}x{window_height}", (10, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(bgr_frame, f"Aspect: {window_aspect_ratio:.2f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(bgr_frame, f"Error: {window_center_x - IMG_CENTER_X}", 
                           (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Overall status
        if window_detected:
            status = "DETECTED"
            color = (0, 255, 0)
        else:
            status = "SEARCHING"
            color = (0, 255, 255)
        
        cv2.putText(bgr_frame, status, (IMG_WIDTH - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # cv2.imshow("Mask", mask) # Optional debug
        cv2.imshow("IMAV - Green Window Tracker", bgr_frame)
        cv2.waitKey(1)
        
    except Exception as e:
        print(f"Vision callback error: {e}")

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

async def run_mission():
    global rel_altitude, window_detected, window_aligned
    global window_center_x, window_width, window_height
    
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("\n⏳ Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected: 
            print("✓ Drone connected!\n")
            break

    # Telemetry background task
    async def get_height():
        global rel_altitude
        async for pos in drone.telemetry.position():
            rel_altitude = pos.relative_altitude_m
    asyncio.create_task(get_height())

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
    
    # Drift right until we see the window and center it
    drift_speed = 0.2
    
    # Loop until centered (detected and error small)
    while True:
        # Check success condition
        if window_detected and abs(window_center_x - IMG_CENTER_X) < CENTER_TOLERANCE_PX:
             print(f"\n✓ Window Centered! Error: {window_center_x - IMG_CENTER_X}")
             break
             
        vy = 0.0
        
        if not window_detected:
            # Blind drift to the right
            vy = drift_speed
            print(f"  Drifting right... (Searching)", end='\r')
        else:
            # Window detected, active centering with strafe (no yaw)
            error_x = window_center_x - IMG_CENTER_X
            
            # P-Control for strafe
            # If window is to the Right (error > 0), we need to Move Right (vy > 0) to bring it left.
            # So Positive Gain.
            
            vy = clamp(error_x * 0.002, -0.4, 0.4)
            
            print(f"  Centering... Error: {error_x}px, Vy: {vy:.2f}", end='\r')

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0, vy, 0, 0)
        )
        await asyncio.sleep(0.1)

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(1.0)

    # ========== PHASE 6: APPROACH AND TRAVERSE ==========
    print("[PHASE 6] ➜ Approaching and traversing window")
    
    # Just move forward at high pace
    TRAVERSE_SPEED = 2.5
    DURATION_SECONDS = 7.0 # Increased to ensure full traversal
    
    print(f"  Moving forward at {TRAVERSE_SPEED} m/s (High Pace)")
    
    steps = int(DURATION_SECONDS / 0.1)
    
    for i in range(steps):
        # constant forward velocity, no corrections
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(TRAVERSE_SPEED, 0, 0, 0)
        )
        await asyncio.sleep(0.1)
        
        if i % 10 == 0:
             print(f"  Traversing... {i*0.1:.1f}s")
    
    print("\n✓ Window traversal complete!\n")

    # ========== PHASE 7: STOP ==========
    print("[PHASE 7] 🛑 Stopping")
    
    # Send 0 velocity to stop motion
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(2.0)
    
    # Stop offboard control (switches to Hold mode)
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
        
    print("✓ Mission complete! (Hovering)\n")

async def main():
    gz_node = Node()
    gz_node.subscribe(GzImage, CAMERA_TOPIC, image_callback)
    
    print("\n" + "="*70)
    print("  IMAV 2025 - GREEN WINDOW MISSION")
    print("  Strategy: Detect (HSV) → Align (Drift) → Traverse")
    print("="*70 + "\n")
    
    await run_mission()
    
    print("="*70)
    print("  MISSION ENDED")
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
