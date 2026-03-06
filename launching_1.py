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

LAUNCH_PAD_CENTER = (320, 240)
LAUNCH_PAD_IGNORE_RADIUS = 100

# Global State
pad_center = None
pad_conf = 0.0
rel_alt = 0.0
detection_streak = 0
last_detection_time = 0.0
last_pad_center = None
pad_velocity_px_s = (0.0, 0.0)

model = YOLO(PAD_MODEL_PATH)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
            box = results[0].boxes[0]
            coords = box.xyxy[0].tolist()
            cx = int((coords[0] + coords[2]) / 2)
            cy = int((coords[1] + coords[3]) / 2)

            # Ignore launch pad (stationary, near center)
            dist_to_launch = np.linalg.norm([cx - LAUNCH_PAD_CENTER[0], cy - LAUNCH_PAD_CENTER[1]])
            vx, vy = pad_velocity_px_s
            speed_px = np.hypot(vx, vy)
            if dist_to_launch < LAUNCH_PAD_IGNORE_RADIUS and speed_px < 5:
                return

            if last_pad_center is not None and last_detection_time > 0:
                dt = now - last_detection_time
                if dt > 0:
                    vx = (cx - last_pad_center[0]) / dt
                    vy = (cy - last_pad_center[1]) / dt
                    pad_velocity_px_s = (vx, vy)
            else:
                pad_velocity_px_s = (0.0, 0.0)

            last_pad_center = (cx, cy)
            last_detection_time = now

            pad_center = (cx, cy)
            pad_conf = float(box.conf)
            detection_streak = min(detection_streak + 1, STREAK_REQUIRED + 5)
        else:
            if detection_streak > 0:
                detection_streak -= 1
            pad_conf = 0.0
            if time.time() - last_detection_time > 1.0:
                pad_center = None
                last_pad_center = None
                pad_velocity_px_s = (0.0, 0.0)

        cv2.circle(bgr, (320, 240), CENTRAL_ZONE_PX, (255, 0, 0), 2)
        if pad_center:
            cv2.circle(bgr, pad_center, 6, (0, 0, 255), -1)
        cv2.imshow("Arena-Locked Tracker", bgr)
        cv2.waitKey(1)
    except Exception:
        pass

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

            img_cx, img_cy = 320, 240
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
    global rel_alt, detection_streak
    drone = System()
    await drone.connect(system_address="udp://:14540")

    async def get_telem():
        global rel_alt
        async for p in drone.telemetry.position():
            rel_alt = p.relative_altitude_m
    asyncio.create_task(get_telem())

    await drone.action.arm()
    await drone.action.set_takeoff_altitude(SEARCH_ALTITUDE)
    await drone.action.takeoff()
    while rel_alt < (SEARCH_ALTITUDE - 0.3):
        await asyncio.sleep(0.3)

    # --- NEW STEP: move away from launch pad by 1m ---
    for _ in range(20):  # 0.5 m/s * 2s ≈ 1m
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.5, 0, 0, 0))
        await asyncio.sleep(0.1)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    await asyncio.sleep(1.0)

    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
        await drone.offboard.start()
    except Exception:
        return

    search_timer = 0
    while True:
        if detection_streak >= STREAK_REQUIRED and pad_center is not None:
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

