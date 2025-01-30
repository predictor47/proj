import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from torchvision.models import resnet18
import statistics
from collections import deque

class ModelComparison:
    def __init__(self):
        # Initialize models
        self.yolo_model = YOLO('yolov8n.pt')  # Pretrained YOLO model
        self.resnet_model = resnet18(weights='DEFAULT')  # Use the latest weights
        self.resnet_model.eval()
        
        # Performance tracking
        self.latency_buffer = deque(maxlen=100)
        self.accuracy_buffer = deque(maxlen=100)
        self.mouse_movement_buffer = deque(maxlen=100)  # For smoothness tracking
    
    def update_metrics(self, latency, accuracy, mouse_movement):
        """Update rolling performance metrics."""
        self.latency_buffer.append(latency)
        self.accuracy_buffer.append(accuracy)
        self.mouse_movement_buffer.append(mouse_movement)
    
    def get_current_metrics(self):
        """Get current performance metrics."""
        if not self.latency_buffer:
            return {
                'avg_latency': 0,
                'accuracy_rate': 0,
                'latency_std': 0,
                'movement_variance': 0
            }
            
        return {
            'avg_latency': statistics.mean(self.latency_buffer),
            'accuracy_rate': statistics.mean(self.accuracy_buffer),
            'latency_std': statistics.stdev(self.latency_buffer) if len(self.latency_buffer) > 1 else 0,
            'movement_variance': statistics.variance(self.mouse_movement_buffer) if len(self.mouse_movement_buffer) > 1 else 0
        }
    
    def measure_performance(self, frame):
        """
        Measure real-time performance on a single frame.
        
        Args:
            frame: Current video frame
        
        Returns:
            dict: Performance metrics for both models
        """
        start_time = time.time()
        
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        end_time = time.time()
        latency_yolo = (end_time - start_time) * 1000  # ms
        
        # Basic accuracy estimation for YOLO
        accuracy_yolo = float(len(results[0].boxes) > 0)
        
        # Run ResNet for feature extraction (if needed)
        # Assuming you have a method to preprocess and run ResNet
        # resnet_latency, resnet_accuracy = self.run_resnet(frame)
        
        # Update rolling metrics
        self.update_metrics(latency_yolo, accuracy_yolo, latency_yolo)  # Using latency as a proxy for mouse movement
        
        return {
            'YOLO': {
                'avg_latency': latency_yolo,
                'accuracy_rate': accuracy_yolo
            },
            # Uncomment if you implement ResNet performance
            # 'ResNet': {
            #     'avg_latency': resnet_latency,
            #     'accuracy_rate': resnet_accuracy
            # }
        }
    
    def _preprocess_image(self, img):
        """Preprocess image for ResNet."""
        try:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return None

def main():
    # Load test images
    test_images = [
        cv2.imread('test_image1.jpg'),
        cv2.imread('test_image2.jpg'),
        cv2.imread('test_image3.jpg')
    ]
    
    # Perform model comparison
    comparison = ModelComparison()
    results = comparison.measure_performance(test_images)
    
    # Print results
    for model, metrics in results.items():
        print(f"{model} Model Performance:")
        print(f"  Average Latency: {metrics['avg_latency']:.2f} ms")
        print(f"  Accuracy Rate: {metrics['accuracy_rate']:.2%}")
        print(f"  Latency Std Dev: {metrics['latency_std']:.2f} ms")
        print(f"  Movement Variance: {metrics['movement_variance']:.2f}\n")

if __name__ == '__main__':
    main()