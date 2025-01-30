import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
from threading import Thread, Lock
from data_augmentation import DataAugmentation
from model_comparison import ModelComparison
import signal
import sys

class EyeTracker:
    def __init__(self):
        # Initialize Mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Smoothing parameters (adjusted for smoother movement)
        self.prev_x = self.screen_w // 2
        self.prev_y = self.screen_h // 2
        self.smooth_factor_x = 0.1  # Adjust for smoother response
        self.smooth_factor_y = 0.1  # Adjust for smoother response
        
        # Sensitivity controls (increased for more responsive movement)
        self.vertical_boost = 4.0  # Increase for more sensitivity
        self.horizontal_boost = 3.0  # Increase for more sensitivity
        
        # Blink detection
        self.blink_threshold = 0.2
        self.blink_cooldown = 0.3
        self.last_blink = 0
        
        # Threading
        self.frame = None
        self.lock = Lock()
        self.running = False
        
        # Add data augmentation and model comparison
        self.augmenter = DataAugmentation()
        self.model_comparator = ModelComparison()
        
        # Add visualization window
        self.show_preview = True
        self.preview_scale = 0.3  # 30% of original size
        
        # Add exit flag
        self.exit_requested = False
        
        # Setup signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle exit signal gracefully"""
        print("\n shutdown initiated...")
        self.exit_requested = True
        
    def start_camera(self):
        """Initialize camera with horizontal flip"""
        self.running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        def capture_thread():
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    # Mirror correction
                    frame = cv2.flip(frame, 1)
                    
                    # Show preview window if enabled
                    if self.show_preview:
                        preview = frame.copy()
                        h, w = preview.shape[:2]
                        preview = cv2.resize(preview, 
                                          (int(w * self.preview_scale), 
                                           int(h * self.preview_scale)))
                        cv2.imshow('Eye Tracking Preview', preview)
                        cv2.waitKey(1)
                    
                    with self.lock:
                        self.frame = frame

        self.thread = Thread(target=capture_thread)
        self.thread.start()
        time.sleep(1)  # Warmup

    def stop_camera(self):
        """Release resources"""
        print("Cleaning up resources...")
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        print("Cleanup complete. Exiting...")

    def get_eyes(self, frame):
        """Detect eye landmarks with Mediapipe"""
        # Apply data augmentation for better detection
        augmented_frame = self.augmenter.augment_image(frame)
        
        rgb = cv2.cvtColor(augmented_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        # Left eye indices (Mediapipe)
        left_eye = np.array([(landmarks[33].x, landmarks[33].y),
                            (landmarks[133].x, landmarks[133].y),
                            (landmarks[159].x, landmarks[159].y),
                            (landmarks[145].x, landmarks[145].y),
                            (landmarks[158].x, landmarks[158].y),
                            (landmarks[153].x, landmarks[153].y)])
        
        # Right eye indices (Mediapipe)
        right_eye = np.array([(landmarks[362].x, landmarks[362].y),
                             (landmarks[385].x, landmarks[385].y),
                             (landmarks[386].x, landmarks[386].y),
                             (landmarks[263].x, landmarks[263].y),
                             (landmarks[374].x, landmarks[374].y),
                             (landmarks[380].x, landmarks[380].y)])
        
        return left_eye, right_eye

    def eye_aspect_ratio(self, eye):
        """Calculate EAR for blink detection"""
        vert = np.linalg.norm(eye[1]-eye[5]) + np.linalg.norm(eye[2]-eye[4])
        horiz = np.linalg.norm(eye[0]-eye[3])
        return vert / (2.0 * horiz)

    def map_gaze(self, left_eye, right_eye):
        """Convert eye position to screen coordinates"""
        # Calculate combined eye center
        eye_x = (np.mean(left_eye[:,0]) + np.mean(right_eye[:,0])) / 2
        eye_y = (np.mean(left_eye[:,1]) + np.mean(right_eye[:,1])) / 2
        
        # Normalize eye positions to be centered
        eye_x = (eye_x - 0.5) * 2  # Normalize to [-1, 1]
        eye_y = (eye_y - 0.5) * 2  # Normalize to [-1, 1]
        
        # Apply sensitivity boosts
        screen_x = eye_x * self.screen_w * self.horizontal_boost
        screen_y = eye_y * self.screen_h * self.vertical_boost
        
        # Keep within screen bounds
        screen_x = np.clip(screen_x, 0, self.screen_w)
        screen_y = np.clip(screen_y, 0, self.screen_h)
        
        # Apply exponential smoothing for smoother movements
        smooth_x = (1 - self.smooth_factor_x) * self.prev_x + self.smooth_factor_x * screen_x
        smooth_y = (1 - self.smooth_factor_y) * self.prev_y + self.smooth_factor_y * screen_y
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def _process_frame(self, frame):
        """Process a single frame with performance monitoring."""
        try:
            # Apply real-time augmentation
            augmented_frame = self.augmenter.augment_image(frame, real_time=True)
            
            # Measure performance
            performance = self.model_comparator.measure_performance(augmented_frame)
            
            # Get eye landmarks
            eyes = self.get_eyes(augmented_frame)
            if not eyes:
                return None
            
            # Dynamic parameter adjustment based on performance
            if len(self.model_comparator.latency_buffer) >= 50:
                self._adjust_parameters(performance)
            
            return eyes
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None

    def run(self):
        """Main tracking loop"""
        try:
            print("Eye tracking started. Press Ctrl+C to exit.")
            while self.running and not self.exit_requested:
                with self.lock:
                    if self.frame is None: continue
                    frame = self.frame.copy()
                
                # Process frame
                result = self._process_frame(frame)
                if result is None: continue
                
                left_eye, right_eye = result
                
                # Enhanced blink detection with smoother tracking
                ear_left = self.eye_aspect_ratio(left_eye)
                ear_right = self.eye_aspect_ratio(right_eye)
                current_time = time.time()
                
                if (ear_left < self.blink_threshold or 
                    ear_right < self.blink_threshold):
                    if current_time - self.last_blink > self.blink_cooldown:
                        pyautogui.click()
                        self.last_blink = current_time
                
                # Get and apply cursor position
                x, y = self.map_gaze(left_eye, right_eye)
                pyautogui.moveTo(x, y, duration=0.1)
                
                # Update performance metrics
                performance = self.model_comparator.get_current_metrics()
                print(f"Avg Latency (YOLO): {performance['avg_latency']:.2f} ms, "
                      f"Accuracy (YOLO): {performance['accuracy_rate']:.2%}, "
                      f"Movement Variance: {performance['movement_variance']:.2f}")
                
                # Save metrics to a text file
                with open("model_performance.txt", "a") as f:
                    f.write(f"Avg Latency (YOLO): {performance['avg_latency']:.2f} ms\n")
                    f.write(f"Accuracy (YOLO): {performance['accuracy_rate']:.2%}\n")
                    f.write(f"Movement Variance: {performance['movement_variance']:.2f}\n")
                    f.write("\n")  # Add a newline for separation
                
                # Add keyboard check for 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExit requested via 'q' key...")
                    break
                
        except Exception as e:
            print(f"Error in tracking loop: {e}")
        finally:
            self.stop_camera()
            sys.exit(0)

    def _adjust_parameters(self, performance):
        """Dynamically adjust tracking parameters based on performance metrics"""
        yolo_metrics = performance['YOLO']
        
        # Adjust smoothing factors based on latency
        if yolo_metrics['avg_latency'] > 50:  # If latency is high
            self.smooth_factor_x = max(0.05, self.smooth_factor_x - 0.02)
            self.smooth_factor_y = max(0.05, self.smooth_factor_y - 0.02)
        else:
            self.smooth_factor_x = min(0.2, self.smooth_factor_x + 0.02)
            self.smooth_factor_y = min(0.2, self.smooth_factor_y + 0.02)
        
        # Adjust sensitivity based on accuracy
        if yolo_metrics['accuracy_rate'] < 0.8:
            self.vertical_boost = max(2.0, self.vertical_boost - 0.1)
            self.horizontal_boost = max(1.5, self.horizontal_boost - 0.1)
        else:
            self.vertical_boost = min(3.0, self.vertical_boost + 0.1)
            self.horizontal_boost = min(2.5, self.horizontal_boost + 0.1)
       
def main():
    """Main entry point with graceful exit handling"""
    tracker = None
    try:
        tracker = EyeTracker()
        tracker.start_camera()
        tracker.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if tracker:
            tracker.stop_camera()
        sys.exit(0)

if __name__ == '__main__':
    main()
       