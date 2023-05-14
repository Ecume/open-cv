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
    img_height, img_width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300, 300)), 0.007, (300, 300), 130)

    neural_net.setInput(blob)
    detected_objects = neural_net.foward()

    for i in range(detected_objects.shape[2]):

        new_confidence = (detected_objects[0][0][i][2])

        if new_confidence > min_confidence:

            img_classes_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3] * img_width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * img_height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * img_width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * img_height)

            prediction_text = f"{img_classes[img_classes_index]}: {new_confidence:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[img_classes_index], 3)
            cv2.putText(image, prediction_text, (upper_left_x, upper_left_y -15 if upper_left_y > 30 else upper_left_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[img_classes_index], 2)
        
    cv2.imshow("Objected detected", image)
    cv2.waitKey(5)

    cv2.destroyAllWindows()
    cap.release()

    

  

