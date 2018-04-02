import urllib.request
import cv2

# Import numpy for matrices calculations
import numpy as np

import os 

def assure_path_exists(path):
    dir = os.path.dirname(path)
    print("Setting file path...")
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Creating folder...")

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX
hc = 0


def clear_screen():
    """
    clear the screen in the command shell
    works on windows (nt, xp, Vista) or Linux
    """
    import os
    os.system(['clear', 'cls'][os.name == 'nt'])

vidpath = "172.16.10.241"

# Loop
while True:

    # Read the video frame
    with urllib.request.urlopen("http://" + vidpath + ":8080/shot.jpg") as url:
        imgReard = url.read()
    imgNp = np.array(bytearray(imgReard), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = cv2.resize(img, (900, 540))
    im = img

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    os.system("clear")
    print("-- Facial Recognition --")
    print("Camera IP:")
    print("  " + vidpath)
    print("Currently Visible: ")

    # For each face in faces
    for(x,y,w,h) in faces:

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist
        if(Id == 1):
            print("  Evan ")
            hc = 1

        if (Id == 2):
            print("  Warren ")
            hc = 1

        if (Id == 3):
            print("  Gianna  ")
            hc = 1
    if hc == 0:
        print("  NONE")
    hc = 0
