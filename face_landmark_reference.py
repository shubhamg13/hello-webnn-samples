import math
import numpy as np
np.math = math

from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite.render import Colors, landmarks_to_render_data, render_to_image
from PIL import Image

# load detection models
detect_faces = FaceDetection()
detect_face_landmarks = FaceLandmark()

# open image; by default, the "front camera"-model is used, which is smaller
# and ideal for selfies, and close-up portraits
img = Image.open('portrait.jpg')
# detect face
face_detections = detect_faces(img)
if len(face_detections):
    # get ROI for the first face found
    face_roi = face_detection_to_roi(face_detections[0], img.size)
    # detect face landmarks
    face_landmarks = detect_face_landmarks(img, face_roi)
    # convert detections to render data
    render_data = landmarks_to_render_data(
        face_landmarks, [], landmark_color=Colors.PINK, thickness=3)
    # render and display landmarks (points only)
    render_to_image(render_data, img).show()
else:
    print('no face detected :(') 