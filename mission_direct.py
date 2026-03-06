import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage

# --- MISSION CONFIGURATION ---
TARGET_ALTITUDE = 1.5  # Meters
IMG_CENTER_X = 320     # For 640x480 resolution 
ALIGN_THRESHOLD = 20   # Pixels of error allowed before pushing through
TRAVERSE_TIME = 4.0    # Seconds to fly forward through the window

# Global Tracking Variables
window_error_x = 0
window_detected = False
is_aligned = False

def image_callback(msg):
    global window_error_x, window_detected, is_aligned
    try:
        # Unpack Gazebo Image 
        frame_bytes = np.frombuffer(msg.data, dtype=np.uint8)
        img = frame_bytes.reshape((msg.height, msg.width, 3))
        bgr_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # COLOR DETECTION LOGIC (Targeting Blue Window)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest blue object (the window)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                window_detected = True
                M = cv2.moments(largest_contour)
                cx = int(M['m10']/M['m00'])
                
                window_error_x = cx - IMG_CENTER_X
                is_aligned = abs(window_error_x) < ALIGN_THRESHOLD
                
                # Visual Feedback
                cv2.drawContours(bgr_frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(bgr_frame, (cx, 240), 5, (0, 0, 255), -1)
        else:
            window_detected = False

        cv2.imshow("IMAV Window Tracker", bgr_frame)
        cv2.waitKey(1)
    except Exception as e:
        print(f"Vision Error: {e}")

async def run_mission():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Connecting to drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    # 1. ARM & TAKEOFF TO 1.5M
    print("-- Arming --")
    await drone.action.arm()
    print(f"-- Taking off to {TARGET_ALTITUDE}m --")
    await drone.action.set_takeoff_altitude(TARGET_ALTITUDE)
    await drone.action.takeoff()

    # Wait for altitude [cite: 99]
    async for pos in drone.telemetry.position():
        if pos.relative_altitude_m >= (TARGET_ALTITUDE - 0.1):
            break

    # 2. ENGAGE OFFBOARD FOR ALIGNMENT
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard fail: {e}")
        return

    print("-- Searching for Blue Window --")
    while not is_aligned:
        if window_detected:
            # Proportional control for Yaw (Steering)
            yaw_speed = 0.1 * (window_error_x / 10) 
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.2, 0.0, 0.0, yaw_speed) # Slow creep forward
            )
        else:
            # Scan/Search rotation
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 10.0))
        await asyncio.sleep(0.1)

    # 3. TRAVERSE THROUGH
    print("-- ALIGNED! Traversing Window --")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(1.5, 0.0, 0.0, 0.0))
    await asyncio.sleep(TRAVERSE_TIME)

    # 4. LAND
    print("-- Mission Complete: Landing --")
    await drone.offboard.stop()
    await drone.action.land()

async def main():
    gz_node = Node()
    gz_node.subscribe(GzImage, "/camera/image_raw", image_callback)
    await run_mission()

if __name__ == "__main__":
    asyncio.run(main())
