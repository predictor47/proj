import cv2
import numpy as np
import albumentations as A
from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class DataAugmentation:
    def __init__(self):
        # Define lighter augmentation pipeline for real-time processing
        self.real_time_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.GaussNoise(p=0.1),
        ])
        
        # Define full augmentation pipeline for training
        self.training_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),  # Scale range
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Translation range
                rotate=(-15, 15),  # Rotation range
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                p=0.2
            )
        ])
    
    def augment_image(self, image, real_time=True):
        """
        Apply random augmentations to input image.
        
        Args:
            image: Input image
            real_time: If True, use lighter augmentations for real-time processing
        """
        transform = self.real_time_transform if real_time else self.training_transform
        try:
            augmented = transform(image=image)['image']
            return augmented
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image
    
    def generate_augmented_dataset(self, original_images, num_augmentations=5):
        """Generate augmented versions of input images for training."""
        augmented_dataset = []
        
        for img in original_images:
            for _ in range(num_augmentations):
                augmented_img = self.augment_image(img, real_time=False)
                augmented_dataset.append(augmented_img)
        
        return augmented_dataset

def load_dataset():
    """Load dataset from Roboflow."""
    try:
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if api_key is None:
            print("Error: No API key found in environment variables")
            return None
            
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("virtual-mouse-z2ehv")
        dataset = project.version(1).download("yolov8")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

def main():
    # Example usage
    augmenter = DataAugmentation()
    
    # Load dataset with your API key
    dataset = load_dataset()
    
    if dataset is None:
        print("Failed to load dataset. Please check your API key.")
        return
        
    # Load sample images
    sample_images = [cv2.imread(img_path) for img_path in dataset.images]
    
    # Generate augmented dataset
    augmented_images = augmenter.generate_augmented_dataset(sample_images)
    
    # Optionally, save or display augmented images
    for i, img in enumerate(augmented_images):
        cv2.imwrite(f'augmented_image_{i}.jpg', img)

if __name__ == '__main__':
    main()