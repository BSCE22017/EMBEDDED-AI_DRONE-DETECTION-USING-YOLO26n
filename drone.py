import cv2
import numpy as np
import onnxruntime as ort
import sys
import time

model_path = "best.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]


if len(sys.argv) < 2:
    print(" Error: Please provide an image path.")
    print("Usage: python laptop_inference.py your_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]


frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: Could not read or find the image '{image_path}'")
    sys.exit(1)

h_orig, w_orig, _ = frame.shape
start_time = time.time()

# --- Preprocessing Pipeline ---
# Resize input image to match model dimensions (640x640)
input_img = cv2.resize(frame, (640, 640))
# Convert color spacing from BGR to RGB
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
# Reorder layout from HWC to CHW and scale down pixel values to FP16 (0.0 - 1.0)
input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
# Add batch axis dimension (1, 3, 640, 640)
input_img = np.expand_dims(input_img, axis=0)

# 4. Run the model inference
outputs = session.run(output_names, {input_name: input_img})
inference_time = (time.time() - start_time) * 1000

print(f"⚡ Laptop Inference complete in {inference_time:.2f} ms")


output = outputs[0]
# Dynamically reshape the matrix from (1, 5, 8400) to (8400, 5)
if output.shape[1] < output.shape[2]:
    predictions = np.squeeze(output).T  
else:
    predictions = np.squeeze(output)    

boxes = []
confidences = []

for pred in predictions:
    confidence = pred[4]
    if confidence > 0.30:  # Confidence threshold filter
        # Step A: Normalize against the 640 model space, then scale up to original image pixels
        x_center = (pred[0] / 640.0) * w_orig
        y_center = (pred[1] / 640.0) * h_orig
        width = (pred[2] / 640.0) * w_orig
        height = (pred[3] / 640.0) * h_orig
        
        # Step B: Convert center coordinates to top-left (x1, y1) format for OpenCV
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        
        boxes.append([x1, y1, int(width), int(height)])
        confidences.append(float(confidence))

# 5. Apply Non-Maximum Suppression (NMS) to eliminate overlapping duplicate boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.30, nms_threshold=0.45)

drone_count = 0
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        conf = confidences[i]
        drone_count += 1
        
        # Clip coordinates to prevent drawing lines outside the image array frame
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_orig, x + w)
        y2 = min(h_orig, y + h)
        
        # Render the tracking visual assets
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Drone: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 6. Save the marked-up output image
output_path = "laptop_result.jpg"
cv2.imwrite(output_path, frame)

print(f"Successfully tracked {drone_count} drone(s). Output saved as: {output_path}")