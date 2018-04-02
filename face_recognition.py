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


def clear_screen():
    """
    clear the screen in the command shell
    works on windows (nt, xp, Vista) or Linux
    """
    import os
    os.system(['clear', 'cls'][os.name == 'nt'])


# Loop
while True:

    # Read the video frame
    with urllib.request.urlopen("http://172.16.10.241:8080/shot.jpg") as url:
        imgReard = url.read()
    imgNp = np.array(bytearray(imgReard), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = cv2.resize(img, (900, 540))
    im = img

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    print("\n", end="")
    print("Currently Visible: ", end="")
    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist

        if(Id == 1 or Id == 5):
            Id = "Evan {0:.2f}%".format(round(100 - confidence, 2))
            print("Evan ", end="")

        if (Id == 2):
            Id = "Warren {0:.2f}%".format(round(100 - confidence, 2))
            print("Warren ", end="")

        if (Id == 3 or Id == 4):
            Id = "Gianna {0:.2f}%".format(round(100 - confidence, 2))
            print("Gianna  ", end="")
        # if (Id == 4):
        #     Id = "Gianna {0:.2f}%".format(round(100 - confidence, 2))
        #     print("Gianna  ", end="")

        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
