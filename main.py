import cv2
import numpy as np

image_used = "test"
prototxt_neural_path = 'models/MobileNetSSD_deploy.prototxt'
model_neural_path = 'waiting for access' #Currently waiting for access to this neural network
min_confidence = 0.2
cap = cv2.VideoCapture(0)







img_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]




np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(img_classes), 3))


neural_net = cv2.dnn.readNetFromCaffe(prototxt_neural_path, model_neural_path)

while True:
    
    _, image = cap.read()
    img_height, img_weight = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300, 300)), 0.007, (300, 300), 130)

    neural_net.setInput(blob)
    detected_objects = neural_net.foward()

    for i in range(detected_objects.shape[2]):

        new_confidence = int(detected_objects[0][0][i][2])

        if new_confidence > min_confidence:

            img_classes_index = int(detected_objects[0, 0, i, 1])

    

  

