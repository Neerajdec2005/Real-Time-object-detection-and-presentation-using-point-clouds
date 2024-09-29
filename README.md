# Real-Time-object-detection-and-presentation-using-point-clouds
This project focuses on real-time 3D object detection using YOLOv3 in Google Colab. It captures a live video feed, detects objects, and visualizes them in 3D space using libraries like Open3D. The system is ideal for applications like augmented reality and robotics, providing both object identification and spatial context in real-time.

The methodology of this project involves a step-by-step integration of object detection with point cloud generation, ensuring real-time analysis and accurate 3D representation of detected objects. The following stages describe the process in detail:

Object Detection Using YOLO: The project begins with real-time object detection using the YOLO (You Only Look Once) model, specifically leveraging the higher version of YOLO for enhanced accuracy and speed. YOLO is selected due to its ability to perform real-time detection with high confidence scores. The model detects multiple objects within the frame, assigns bounding boxes, and labels them with their respective class names and confidence levels. This stage is crucial as it identifies the regions of interest in the image that will later be transformed into point clouds. The model runs on live camera input to continuously detect objects in each frame.

Exclusion of Specific Classes (Person Filter): A custom filtering process is incorporated to exclude a specific object class, in this case, "person." The filtering ensures that detected persons are ignored in both the object detection output and subsequent point cloud generation. By filtering the class of interest, the system focuses on detecting and generating point clouds for the remaining objects within the scene. This step demonstrates the system’s flexibility to handle varying scenarios and tailor the detection process to specific needs.

Depth Estimation and Point Cloud Generation: Once the target objects are detected, the methodology shifts to point cloud generation. Using image-processing techniques like contour extraction, the boundaries of the detected object are identified. Depth estimation techniques are employed to assign a z-coordinate (depth) to each point in the object's 2D contour, transforming it into a 3D representation. MiDaS, a state-of-the-art monocular depth estimation model, is used in this project to generate depth information from the 2D images, which is essential for accurate point cloud creation.  ![image](https://github.com/user-attachments/assets/7188c9fe-32d1-4779-b189-5006f4cda8a2)
![image](https://github.com/user-attachments/assets/5aa54a7f-03e5-4633-be4f-75809ed68b9f)

![image](https://github.com/user-attachments/assets/68f491db-8ece-489c-84a3-bc2e44d33019)

![image](https://github.com/user-attachments/assets/4650ea28-4358-447a-9810-e73fcf93076f)


Refinement of Point Cloud Accuracy: To achieve high fidelity in point cloud generation, several techniques are applied to improve the accuracy of the 3D representation. This involves adjusting the point density, reducing noise from edge detection, and refining the placement of points in 3D space. The number of points can reach up to 100,000 for precise representation, ensuring that the point cloud closely mirrors the actual shape and structure of the detected object. This stage ensures that even complex objects are represented accurately, providing a clear visual interpretation of the 3D shape.

Visualization of Point Cloud: The point cloud is visualized using 3D plotting tools such as Plotly. The plot includes x, y, and z coordinates of the object’s surface points, allowing users to interact with and observe the object from different perspectives. The 3D scatter plot presents the object in space, and the high-density point cloud ensures that the object is visually recognizable. This visualization step is critical in showcasing the real-time capability of the system and offering an intuitive understanding of the detected objects.

Real-Time Operation: The system is designed to operate in real-time, processing frames from a live camera feed. Object detection, point cloud generation, and visualization are performed continuously to ensure a seamless experience. The real-time nature of the system allows it to be applied in dynamic environments, where objects are continuously moving and the scene is constantly changing.

By integrating these stages, the project achieves a real-time system that not only detects objects but also generates accurate 3D point clouds, providing users with a deeper understanding of their surroundings. This methodology can be applied across various domains, including robotics, augmented reality, and autonomous systems.
