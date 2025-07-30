AI Traffic Monitoring System 🚗💨
This project is a web application built with Streamlit that uses artificial intelligence to analyze traffic videos. The system detects and tracks vehicles, estimates their speed, and generates detailed visualizations and reports from the analysis.

✨ Key Features
Video Upload: Upload video files in common formats (.mp4, .avi, .mov).

Vehicle Detection: Uses a YOLOv8 model to identify vehicles with high accuracy.

Object Tracking: Implements the DeepSORT algorithm to assign a unique ID to each vehicle and follow its trajectory across frames.

Speed Estimation: Calculates the speed of each vehicle in pixels per second.

Results Visualization:

Generates a processed video showing bounding boxes, trajectory, and the speed of each vehicle.

Creates analytical graphs, including a line chart of vehicle speeds over time and a speed heatmap overlaid on a video frame.

Data Export: Allows downloading position and speed data in CSV files for further analysis.

Customizable Settings: Adjust key parameters like the confidence threshold for detection and the max track age to optimize results.

🛠️ Tech Stack
Python 3

Streamlit: For the web application interface.

OpenCV: For video processing.

Ultralytics (YOLO): For object detection.

deep-sort-realtime: For object tracking.

NumPy & Pandas: For data manipulation.

Matplotlib & Seaborn: For generating plots.

⚙️ Installation
Follow these steps to get the project running in your local environment.

Clone the repository:

git clone https://github.com/your-username/AI-Traffic-Monitoring-System.git
cd AI-Traffic-Monitoring-System

Create a virtual environment (recommended):

# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

Install the dependencies:
Create a requirements.txt file with the following content:

streamlit
opencv-python
numpy
pandas
matplotlib
seaborn
ultralytics
deep-sort-realtime

And then run:

pip install -r requirements.txt

Download the YOLO model:
Make sure the pre-trained model (modelo_vista_superior.pt) is located in a folder named models within the main project directory.

🚀 How to Use
Run the Streamlit application:
Open your terminal, navigate to the project directory, and run the following command (assuming your main script is named app.py):

streamlit run app.py

Interact with the interface:

A new tab will open in your web browser.

Use the sidebar to upload your traffic video.

Adjust the "Confidence Threshold" and "Max Track Age" sliders if desired.

Click the "Process Video" button.

Review the results:

The process may take a few minutes. Once complete, the application will display:

The processed video with the tracking overlays.

The speed analysis graphs.

Buttons to download the processed video and the CSV data files.

🔩 How It Works
The system's workflow can be summarized in the following steps:

Input and Preprocessing: The user uploads a video. The system resizes it to optimize performance and calculates a frame_skip value for high-FPS videos.

Detection (VideoProcessor class): Each frame of the video is passed to the YOLO model, which returns the coordinates of the detected vehicles.

Tracking (VehicleTracker class): The detections from YOLO are passed to the DeepSort tracker. It assigns a unique ID to each vehicle and maintains a history of its positions.

Data Calculation: The distance in pixels that each vehicle travels between frames is calculated. Using the video's FPS, this distance is converted into a speed in pixels per second.

Output Generation:

A new video is created where bounding boxes, track IDs, and calculated speeds are drawn over each vehicle.

Position and speed data are saved to CSV files.

Graphs are generated from the CSV data for quick visual analysis.