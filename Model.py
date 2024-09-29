

# Load YOLOv5 model from Ultralytics (pre-trained on COCO dataset)
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Use a larger model for better accuracy

# JavaScript to capture image from webcam
def get_webcam_image():
    js = Javascript('''
    async function takePhoto() {
      const div = document.createElement('div');
      const video = document.createElement('video');
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      const button = document.createElement('button');
      div.appendChild(button);
      button.textContent = 'Take Photo';
      await new Promise((resolve) => button.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    ''')
    display(js)
    return js

# Function to extract object contours and generate point cloud
def contour_to_pointcloud(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = img[y1:y2, x1:x2]  # Crop the object region
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)  # Detect edges

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []

    for contour in contours:
        for point in contour:
            x, y = point[0]
            z = np.random.uniform(0, 1)  # Assign random depth (improve if you have actual depth data)
            points.append([x + x1, y + y1, z])  # Add offset to original image coordinates

    return np.array(points)

# Function to visualize the point cloud using Plotly
def visualize_pointcloud(pointcloud, label):
    if pointcloud.size == 0:
        print("No point cloud to visualize.")
        return

    # Create a 3D scatter plot
    trace = go.Scatter3d(
        x=pointcloud[:, 0],
        y=pointcloud[:, 1],
        z=pointcloud[:, 2],
        mode='markers',
        marker=dict(
            size=4,  # Smaller point size for better accuracy
            color=np.random.randn(len(pointcloud)),  # Color by random values
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        title=f"Point Cloud for {label}",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the figure
    fig.show()

# Function to capture the image, perform detection, and display one object as a point cloud
def webcam_object_detection():
    js = get_webcam_image()  # Call get_webcam_image()
    data = eval_js('takePhoto()')

    # Decode the base64 image data
    binary = b64decode(data.split(',')[1])
    jpg = np.frombuffer(binary, dtype=np.uint8)

    # Convert the image to OpenCV format and resize it
    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 480))  # Resize for faster inference
    cv2.imwrite('webcam.jpg', img_resized)  # Save the image for detection

    # Perform object detection
    results = model_yolo(img_resized)

    # Extracting detected objects
    labels = results.names

    # Move detected objects tensor to CPU before converting to NumPy
    detected_objects = results.xyxy[0].cpu().numpy()

    # Display results on the image
    results.show()

    if len(detected_objects) == 0:
        print("No objects detected.")
        return

    # Choose the object with the highest confidence
    best_object = detected_objects[np.argmax(detected_objects[:, 4])]
    xmin, ymin, xmax, ymax, confidence, class_id = best_object
    class_name = labels[int(class_id)]
    print(f"Detected: {class_name} with confidence {confidence:.2f}")

    # Generate point cloud using object contours
    points = contour_to_pointcloud(img_resized, [xmin, ymin, xmax, ymax])
    print(f"Generating point cloud for: {class_name} with {len(points)} points")

    # Visualize the point cloud for the most confident object
    visualize_pointcloud(points, class_name)

# Run the function to detect objects and visualize point cloud for one object
webcam_object_detection()
