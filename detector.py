from super_gradients.training import models
import torch
import cv2
import numpy as np
import os
import requests

# Get the YOLO NAS small model with pretrained weights on COCO dataset
yolo_nas = models.get("yolo_nas_l", pretrained_weights="coco")

# Set the device to use GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else "cpu"
print("using device: ", device)

# Create a Detector class to encapsulate object detection functionality
class Detector:
    def __init__(self, model=yolo_nas):
        self.model = model

    def onImage(self, image_path: str, output_file: str = "im_detections.jpg", filter: str = "", conf_threshold: float = 0.25):
        """
        Perform object detection on a single image file.
        - image_path: The path to the image file.
        - conf_threshold: Confidence threshold for object detection.
        - output_file: The output file name for the annotated image.
        """

        model = self.model.to(device)
        detections = model.predict(image_path, conf=conf_threshold)
        
        if filter != "" and filter != "all":
            # Obtener predicciones de detecciones y clases
            bboxes = detections.prediction.bboxes_xyxy
            confidences = detections.prediction.confidence
            int_labels = detections.prediction.labels.astype(int)
            class_names = detections.class_names

            # Leer la imagen
            resp = requests.get(image_path, stream=True, headers={'User-Agent': 'Mozilla/5.0'} ).raw
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Dibujar cuadros delimitadores y etiquetas en la imagen
            for bbox, confidence, label in zip(bboxes, confidences, int_labels):
                if class_names[label] == filter and confidence >= conf_threshold:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"{class_names[label]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Guardar la imagen con las detecciones anotadas
            saved = cv2.imwrite(output_file, image)
            if saved:
                print(f"Detections saved to {output_file}")
            else:
                print(f"Error saving detections to {output_file}")
        else:
            detections.save(output_file)
        #self.model.to(device).predict(image_path, conf=conf_threshold).save(output_file)

    def onVideo(self, video_path: str, output_file: str = "detections.mp4"):
        """
        Perform object detection on a video file.
        - video_path: The path to the video file.
        - output_file: The output file name for the annotated video.
        """
        model = self.model.to(device)
        detections = model.predict(video_path)
        detections.save(output_file)

    def realTime(self):
        """
        Perform object detection in real-time using the webcam.
        """
        output = self.model.predict_webcam()