import cv2
import numpy as np
import math

classes = ["coke", "sheep"]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def calculate_distance(name, h, x, Y):

    f = 640
    Z = (f*Y)/h

    p = (x-320)
    X = (p * Z)/f
    print("Depth", Z)
    # print("x cordinate", X)

    # if((Z > 0.5 and Z < 1.5) and (X > -0.2 and X < 0.2)):
    #     # return(Z, X)
    #     return([name, Z])
    return  ([name, Z])

    # else:
    #     return 0
    # return Z, X

def get_cordinate(image):
    # image = cv2.imread(img)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.8
    nms_threshold = 0.8


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if(len(indices) > 0):
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            height = round(h)
            if(class_ids[i] == 0): #if coke
                # print(classes[class_ids[i]], "cordinates, x, y, width, height", round(x), round(y), round(w), round(h))
                return calculate_distance(class_ids[i], round(h), round(x), 0.14)
            else:
                # print(classes[class_ids[i]], "cordinates, x, y, width, height", round(x), round(y), round(w), round(h))
                return calculate_distance(class_ids[i], round(h), round(x), 0.204)

                # print(calculate_distance(round(h), round(x)))
            # print(classes[class_ids[i]], round(x), round(y), round(x+w), round(y+h))

        # cv2.imshow("object detection", image)
        # cv2.waitKey()
            
        # cv2.imwrite("object-detection.jpg", image)
        # cv2.destroyAllWindows()
    else:
        # print("There is no object")
        return 0
