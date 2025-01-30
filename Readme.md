# Eye Tracking Project

## Overview
This project implements a real-time eye tracking system that translates eye movements into mouse cursor movements on a screen. The system utilizes computer vision techniques to detect eye positions and provides an alternative input method for users, particularly beneficial for individuals with mobility impairments.

## Features
- Real-time eye tracking using webcam input.
- Mouse cursor control based on eye movements.
- Data augmentation techniques to improve model robustness.
- Performance evaluation metrics for model accuracy and latency.
- Blink detection for mouse click functionality.

## Dataset
The dataset used for this project is sourced from Roboflow, specifically designed for eye tracking applications. It contains images annotated with facial landmarks, including eye positions.

## Models Used
- **YOLO (You Only Look Once)**: A real-time object detection model for detecting facial landmarks.
- **ResNet18**: A convolutional neural network (CNN) for feature extraction.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eye-tracking-project.git
   cd eye-tracking-project
   ```

2. Run the installation script to set up the virtual environment and install dependencies:
   ```bash
   requirements_windows.bat
   ```

3. Create a `.env` file in the root directory and add your Roboflow API key:
   ```plaintext
   ROBOFLOW_API_KEY=your_api_key_here
   ```

## Usage
1. Activate the virtual environment:
   ```bash
   gaze_mouse_env\Scripts\activate
   ```

2. Create a config.py file:
    ```Add Roboflow Api Key``

3. Run the eye tracking module:
   ```bash
   python eye_tracking_module.py
   ```

4. Follow the on-screen instructions to use the eye tracking system. The mouse cursor will move based on your eye movements, and blinking will trigger mouse clicks.

## Evaluation Metrics
The system evaluates performance using the following metrics:
- Average Latency: Time taken to process each frame.
- Accuracy Rate: Percentage of correctly detected eye positions.
- Movement Variance: Variance in mouse movement for smoothness assessment.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Roboflow](https://roboflow.com/) for providing the dataset.
- [Ultralytics](https://github.com/ultralytics/yolov5) for the YOLO model implementation.
- [Mediapipe](https://google.github.io/mediapipe/) for facial landmark detection.

## Contact
For any inquiries, please contact [your_email@example.com](mailto:your_email@example.com).