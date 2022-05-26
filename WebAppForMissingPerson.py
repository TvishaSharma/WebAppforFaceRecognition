import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import streamlit as st

html_temp = """

   <body style="background-color:red;">
   <div style="background-color:teal ;padding:10px">
   <h1 style="background-color:black;text-align:centre;">FIND YOUR MISSING ONES</h1>
   <h2 style="color:white;text-align:center;">A WebApp for Facial Recognition </h2>
   <p>Each year , police stations across the country receive thousands of reports of missing persons. Many of those who are reported  to police as missing are located within a short span of time.There are however,others who are never found, eventually, identified as victims of crime or misfortune.")
Numerous studies have shown that a significant number of people go missing each year.While there are many intrinsic risks associated with any missing incident,specific sections of population is more vulnerable to harm while missing.The reasons for going missing are many and varied and can include menatl illness, miscommunication, misadventure, domestic viiolence, and being a victim of crime.</p>
   </div>
   </body>
   """
st.markdown(html_temp, unsafe_allow_html=True)
st.write("Click on run to see the result")
run = st.checkbox('RUN')
st.text("Uncheck the Run button once done with matching of face")
FRAME_WINDOW = st.image([])
path = 'images'
images = []
classNames = []
myList = os.listdir(path)

#Separating name and extension from images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#we will write now face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Completed')
#We will now access our camera
cam = cv2.VideoCapture(0)

while run:
    ret, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # We will resize the images since the images might be of different sizes.
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # We need to find face in video too
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            markAttendance(name)
        else:
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, 'Not the missing person', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)



    FRAME_WINDOW.image(img)
st.image("missing people imgs//india stats of missing cases.jpeg")
st.image("missing people imgs//news article.jpeg")

st.text("Thank You for choosing us, Hope you have found your missing family member")

