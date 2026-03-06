import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus, VehicleLocalPosition
from cv_bridge import CvBridge
import cv2
import math
import time
import numpy as np
from ultralytics import YOLO


class GreenWindowMission(Node):
    
    def __init__(self):
        super().__init__('green_window_mission')
        
        # QoS profile for PX4 compatibility
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # --- Publishers ---
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        # --- Subscribers ---
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', 
            self.vehicle_status_callback, qos_profile=qos_profile)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', 
            self.local_position_callback, qos_profile=qos_profile)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', 
            self.image_callback, 10)

        # --- State Variables ---
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MANUAL
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.local_position = VehicleLocalPosition()
        
        # --- Control Timer ---
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        
        # --- CV Bridge ---
        self.cv_bridge = CvBridge()
        
        # --- YOLO Model ---
        self.model = YOLO("best.pt")
        self.GREEN_CLASS_ID = 1
        self.get_logger().info(f"✓ Loaded YOLO model. Classes: {self.model.names}")
        
        # --- Mission States ---
        self.MISSION_ACTIVE = False
        self.STATE = "IDLE"  # IDLE, TAKEOFF, SEARCH, APPROACH, ALIGN, TRAVERSE, COMPLETE
        
        # --- Image Parameters ---
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        self.IMAGE_CENTER_X = 320
        self.IMAGE_CENTER_Y = 240
        
        # --- Detection Variables ---
        self.window_detected = False
        self.window_aligned = False
        self.window_center_x = 0
        self.window_center_y = 0
        self.window_width = 0
        self.window_height = 0
        self.window_aspect_ratio = 0.0
        self.window_confidence = 0.0
        
        # --- Control Parameters ---
        self.TAKEOFF_ALTITUDE = -1.5  # NED frame (negative = up)
        self.SEARCH_YAW_RATE = 0.5  # rad/s (clockwise)
        self.APPROACH_SPEED = 0.5  # m/s
        self.APPROACH_WIDTH_TARGET = 200  # px (stop when window is this wide)
        self.ALIGN_YAW_RATE = -0.4  # rad/s (counter-clockwise)
        self.TRAVERSE_SPEED = 1.5  # m/s
        self.CENTER_TOLERANCE = 60  # px
        self.YAW_GAIN = 0.002
        self.ALIGNED_ASPECT_MIN = 0.8
        self.ALIGNED_ASPECT_MAX = 1.2
        
        # --- Timing Variables ---
        self.state_start_time = time.time()
        self.takeoff_complete = False
        self.search_timeout = 30.0  # seconds
        self.maneuver_timeout = 25.0  # seconds
        self.align_timeout = 15.0  # seconds
        self.traverse_duration = 5.0  # seconds
        
        # --- Counters ---
        self.offboard_setpoint_counter = 0
        self.alignment_counter = 0
        self.ALIGNMENT_FRAMES_NEEDED = 10
        
        self.get_logger().info("="*60)
        self.get_logger().info("  GREEN WINDOW MISSION - ROS2 + PX4")
        self.get_logger().info("  Commands:")
        self.get_logger().info("    - Mission will start automatically after takeoff")
        self.get_logger().info("="*60)

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state
    
    def local_position_callback(self, msg):
        self.local_position = msg
    
    def is_window_aligned(self):
        """Check if window appears square (head-on view)"""
        if self.window_width < 150 or self.window_height < 150:
            return False
        return self.ALIGNED_ASPECT_MIN <= self.window_aspect_ratio <= self.ALIGNED_ASPECT_MAX
    
    def is_centered(self):
        """Check if window is centered in image"""
        error_x = abs(self.window_center_x - self.IMAGE_CENTER_X)
        error_y = abs(self.window_center_y - self.IMAGE_CENTER_Y)
        return error_x < self.CENTER_TOLERANCE and error_y < self.CENTER_TOLERANCE
    
    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv_image = cv2.resize(cv_image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            
            # Run YOLO detection
            results = self.model(cv_image, verbose=False, conf=0.5)
            
            # Reset detection
            self.window_detected = False
            self.window_aligned = False
            
            # Find GREEN window only
            best_box = None
            best_conf = 0
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    if class_id != self.GREEN_CLASS_ID:
                        continue
                    
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
            
            if best_box is not None:
                self.window_detected = True
                self.window_confidence = best_conf
                
                # Get bounding box
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate parameters
                self.window_width = x2 - x1
                self.window_height = y2 - y1
                self.window_center_x = (x1 + x2) // 2
                self.window_center_y = (y1 + y2) // 2
                self.window_aspect_ratio = self.window_width / self.window_height if self.window_height > 0 else 0
                self.window_aligned = self.is_window_aligned()
                
                # Visualization
                box_color = (0, 255, 0) if self.window_aligned else (0, 165, 255)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), box_color, 3)
                cv2.circle(cv_image, (self.window_center_x, self.window_center_y), 8, (0, 0, 255), -1)
                
                # Status text
                status_text = "✓ ALIGNED" if self.window_aligned else "✗ SIDE VIEW"
                status_color = (0, 255, 0) if self.window_aligned else (0, 0, 255)
                cv2.putText(cv_image, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.putText(cv_image, f"GREEN WINDOW | Conf: {self.window_confidence*100:.1f}%", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(cv_image, f"Size: {self.window_width}x{self.window_height} | Aspect: {self.window_aspect_ratio:.2f}", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Crosshair at center
            cv2.line(cv_image, (self.IMAGE_CENTER_X - 40, self.IMAGE_CENTER_Y), 
                    (self.IMAGE_CENTER_X + 40, self.IMAGE_CENTER_Y), (255, 255, 255), 2)
            cv2.line(cv_image, (self.IMAGE_CENTER_X, self.IMAGE_CENTER_Y - 40), 
                    (self.IMAGE_CENTER_X, self.IMAGE_CENTER_Y + 40), (255, 255, 255), 2)
            
            # Mission state
            cv2.putText(cv_image, f"STATE: {self.STATE}", 
                       (10, self.IMAGE_HEIGHT - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Green Window Mission", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def timer_callback(self):
        # Publish offboard control mode heartbeat
        self.publish_offboard_control_mode()
        
        # Send setpoints before offboard mode
        if self.offboard_setpoint_counter < 10:
            self.offboard_setpoint_counter += 1
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)
        
        # Check if armed
        if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if time.time() % 2 < 0.1:
                self.get_logger().info("Waiting for arming... Arm drone manually or via command")
            return
        
        # Switch to offboard mode
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.get_logger().info("Switching to OFFBOARD mode")
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 
                param1=1.0, param2=6.0)
            return
        
        # Start mission automatically after offboard is active
        if not self.MISSION_ACTIVE:
            self.MISSION_ACTIVE = True
            self.STATE = "TAKEOFF"
            self.state_start_time = time.time()
            self.get_logger().info("🚁 MISSION STARTED - TAKEOFF")
        
        # State machine
        self.run_state_machine()
    
    def run_state_machine(self):
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        if self.STATE == "TAKEOFF":
            # Ascend to target altitude
            current_alt = self.local_position.z
            
            if current_alt < self.TAKEOFF_ALTITUDE + 0.2:  # Close enough
                self.publish_trajectory_setpoint(0.0, 0.0, -0.3, 0.0)  # Ascend
            else:
                self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)  # Hover
                if not self.takeoff_complete:
                    self.takeoff_complete = True
                    self.change_state("SEARCH")
                    self.get_logger().info(f"✓ Takeoff complete at {-current_alt:.2f}m")
        
        elif self.STATE == "SEARCH":
            # Rotate clockwise to search for green window
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0, self.SEARCH_YAW_RATE)
            
            if self.window_detected:
                self.change_state("APPROACH")
                self.get_logger().info(f"✓ Green window detected! Confidence: {self.window_confidence*100:.1f}%")
            elif elapsed > self.search_timeout:
                self.get_logger().warn("Search timeout! No window found.")
                self.change_state("COMPLETE")
        
        elif self.STATE == "APPROACH":
            # Move forward while keeping window centered
            if not self.window_detected:
                self.get_logger().warn("Lost window during approach, re-searching...")
                self.change_state("SEARCH")
                return
            
            # Stop when window is close (large enough)
            if self.window_width >= self.APPROACH_WIDTH_TARGET:
                self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)  # Stop
                self.change_state("ALIGN")
                self.get_logger().info(f"✓ Reached approach distance (width: {self.window_width}px)")
                return
            
            # Calculate yaw correction to keep centered
            error_x = self.window_center_x - self.IMAGE_CENTER_X
            yaw_correction = np.clip(error_x * self.YAW_GAIN, -0.3, 0.3)
            
            # Convert local forward velocity to global
            heading = self.local_position.heading
            vx = self.APPROACH_SPEED * math.cos(heading)
            vy = self.APPROACH_SPEED * math.sin(heading)
            
            self.publish_trajectory_setpoint(vx, vy, 0.0, yaw_correction)
            
            if elapsed > self.maneuver_timeout:
                self.get_logger().warn("Approach timeout!")
                self.change_state("SEARCH")
        
        elif self.STATE == "ALIGN":
            # Rotate counter-clockwise + strafe to get head-on view
            if not self.window_detected:
                self.get_logger().warn("Lost window during alignment, re-searching...")
                self.change_state("SEARCH")
                return
            
            if self.window_aligned and self.is_centered():
                self.alignment_counter += 1
                if self.alignment_counter >= self.ALIGNMENT_FRAMES_NEEDED:
                    self.change_state("CENTER")
                    self.get_logger().info(f"✓ Aligned! Aspect: {self.window_aspect_ratio:.2f}")
            else:
                self.alignment_counter = 0
                
                # Maneuver to get aligned
                error_x = self.window_center_x - self.IMAGE_CENTER_X
                
                # Counter-clockwise rotation + strafing
                yaw_rate = self.ALIGN_YAW_RATE
                
                # Strafe perpendicular to viewing direction
                heading = self.local_position.heading
                strafe_speed = 0.3 if error_x > 100 else -0.3 if error_x < -100 else 0
                vx = -strafe_speed * math.sin(heading)
                vy = strafe_speed * math.cos(heading)
                
                # Small forward movement
                fwd_speed = 0.2
                vx += fwd_speed * math.cos(heading)
                vy += fwd_speed * math.sin(heading)
                
                self.publish_trajectory_setpoint(vx, vy, 0.0, yaw_rate)
            
            if elapsed > self.align_timeout:
                self.get_logger().warn("Alignment timeout, proceeding anyway...")
                self.change_state("CENTER")
        
        elif self.STATE == "CENTER":
            # Fine-tune centering with yaw
            if not self.window_detected:
                self.change_state("TRAVERSE")
                return
            
            error_x = self.window_center_x - self.IMAGE_CENTER_X
            
            if abs(error_x) < self.CENTER_TOLERANCE:
                self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)
                self.change_state("TRAVERSE")
                self.get_logger().info(f"✓ Centered! Error: {error_x}px")
            else:
                yaw_correction = np.clip(error_x * self.YAW_GAIN * 1.5, -0.3, 0.3)
                
                # Slow forward creep
                heading = self.local_position.heading
                vx = 0.15 * math.cos(heading)
                vy = 0.15 * math.sin(heading)
                
                self.publish_trajectory_setpoint(vx, vy, 0.0, yaw_correction)
            
            if elapsed > 10.0:
                self.change_state("TRAVERSE")
        
        elif self.STATE == "TRAVERSE":
            # Move forward through window
            heading = self.local_position.heading
            vx = self.TRAVERSE_SPEED * math.cos(heading)
            vy = self.TRAVERSE_SPEED * math.sin(heading)
            
            # Minor yaw correction if window still visible
            yaw_correction = 0.0
            if self.window_detected:
                error_x = self.window_center_x - self.IMAGE_CENTER_X
                yaw_correction = np.clip(error_x * self.YAW_GAIN * 0.5, -0.2, 0.2)
            
            self.publish_trajectory_setpoint(vx, vy, 0.0, yaw_correction)
            
            if elapsed > self.traverse_duration:
                self.change_state("COMPLETE")
                self.get_logger().info("✓ Traverse complete!")
        
        elif self.STATE == "COMPLETE":
            # Hover in place
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)
            if elapsed > 2.0:
                self.get_logger().info("🎉 MISSION COMPLETE! Hovering...")
                self.STATE = "HOVER"
        
        elif self.STATE == "HOVER":
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)
    
    def change_state(self, new_state):
        """Change state and reset timer"""
        self.STATE = new_state
        self.state_start_time = time.time()
        self.alignment_counter = 0

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, vx, vy, vz, yawspeed):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        nan = float('nan')
        
        msg.position = [nan, nan, nan]
        msg.velocity = [vx, vy, vz]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = nan
        msg.yawspeed = yawspeed
        
        self.trajectory_setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    mission = GreenWindowMission()
    try:
        rclpy.spin(mission)
    except KeyboardInterrupt:
        pass
    finally:
        mission.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
