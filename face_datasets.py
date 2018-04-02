import cv2
import os
import numpy as np
import urllib.request

def assure_path_exists(path):
    dir = os.path.dirname(path)
    print("Setting file path...")
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Creating folder...")

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id
face_id = 5

# Initialize sample face image
count = 0

assure_path_exists("./dataset/")

# Start looping
while(True):

    # Capture video frame
    with urllib.request.urlopen("http://172.16.10.241:8080/shot.jpg") as url:
        imgReard = url.read()
    imgNp = np.array(bytearray(imgReard), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = cv2.resize(img, (900, 540))
    image_frame = img

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("./dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)
    print("Count: " + str(count))
    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video

    elif count>100:
        break

# Stop video


# Close all started windows
cv2.destroyAllWindows()
