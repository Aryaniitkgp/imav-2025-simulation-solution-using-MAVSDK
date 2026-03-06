import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import asyncio
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage
from ultralytics import YOLO

# --- CONFIGURATION ---
DOWN_TOPIC = "/rgbd_image_down/image" # Verified downward camera topic [cite: 101]
SEARCH_ALTITUDE = 3.5           # Safely above 3m cylinders [cite: 46, 53]
PAD_MODEL_PATH = "best.pt" # Your custom YOLO model

# Global State
pad_center = None
pad_conf = 0.0
rel_alt = 0.0

model = YOLO(PAD_MODEL_PATH)

def down_callback(msg):
    global pad_center, pad_conf
    try:
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run Custom YOLO
        results = model(bgr, verbose=False, conf=0.5)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            coords = box.xyxy[0].tolist()
            pad_center = (int((coords[0]+coords[2])/2), int((coords[1]+coords[3])/2))
            pad_conf = float(box.conf)
            cv2.rectangle(bgr, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
        else:
            pad_conf = 0.0

        cv2.imshow("IMAV High-Alt View", bgr)
        cv2.waitKey(1)
    except: pass

async def run_mission():
    global pad_center, pad_conf, rel_alt
    drone = System()
    await drone.connect(system_address="udp://:14540")

    async def get_telem():
        global rel_alt
        async for p in drone.telemetry.position(): rel_alt = p.relative_altitude_m
    asyncio.create_task(get_telem())

    # 1. Takeoff to Safe Height
    print(f"-- Ascending to {SEARCH_ALTITUDE}m --")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(SEARCH_ALTITUDE)
    await drone.action.takeoff()
    while rel_alt < (SEARCH_ALTITUDE - 0.3): await asyncio.sleep(0.5)

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
    try: await drone.offboard.start()
    except: return

    # 2. Scanning State Machine
    print("Beginning Lawnmower Search Pattern...")
    step_timer = 0
    
    while True:
        # A. TARGET FOUND: Tracking Logic
        if pad_conf > 0.6:
            print(f"H-Pad Detected! Conf: {pad_conf:.2f}")
            # Center the pad in view
            vx = (240 - pad_center[1]) * 0.03
            vy = (pad_center[0] - 320) * 0.03
            # Descend only when aligned to avoid hitting things
            vz = 0.4 if abs(vx) < 0.15 and abs(vy) < 0.15 else 0.0
            
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, 0))
            if rel_alt < 0.4:
                await drone.action.land()
                break
        
        # B. SEARCHING: Lawnmower Pattern (Fixed straight lines)
        else:
            step_timer += 1
            # Fly straight for 80 steps (approx 8 seconds)
            if step_timer < 80:
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.8, 0, 0, 0))
            # Turn 90 degrees to stay inside arena
            elif step_timer < 100:
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 45.0))
            else:
                step_timer = 0 # Reset for next leg
                
        await asyncio.sleep(0.1)

async def main():
    node = Node()
    node.subscribe(GzImage, DOWN_TOPIC, down_callback)
    await run_mission()

if __name__ == "__main__":
    asyncio.run(main())
