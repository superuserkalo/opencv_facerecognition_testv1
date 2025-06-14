import cv2
import dlib
import face_recognition_models
import face_recognition
import numpy as np
import recog_version as rv

print(rv.recog_version())

kaloimg = face_recognition.load_image_file("kaloimg.jpg")
kaloimg_encoding = face_recognition.face_encodings(kaloimg)[0]

jordaimg = face_recognition.load_image_file("jordaimg.jpg")
jordaimg_encoding = face_recognition.face_encodings(jordaimg)[0]

davutimg = face_recognition.load_image_file("davut.jpg")
davutimg_encoding = face_recognition.face_encodings(davutimg)[0]

jetimg = face_recognition.load_image_file("jetlum.jpg")
jetimg_encoding = face_recognition.face_encodings(jetimg)[0]

known_faces = [
    kaloimg_encoding,
    jordaimg_encoding,
    davutimg_encoding,
    jetimg_encoding,
]

known_faces_names = [
    "Kalo",
    "Jorda",
    "Davut",
    "Jetlum",
]

face_location = []
face_encodings = []
face_names = []
process_frame = True


cam = cv2.VideoCapture(0)

if not cam.isOpened():
        raise IOError("Cannot open webcam")

while True:
    ret, frame = cam.read()

    if not ret:
        raise IOError("Cannot read data")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if process_frame:
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)


        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        #print(face_locations)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []

    for faces in face_encodings:
        match = face_recognition.compare_faces(known_faces,faces)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, faces)
        best_match_index = np.argmin(face_distances)
        if match[best_match_index]:
            name = known_faces_names[best_match_index]

            face_names.append(name)

    process_frame = not process_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, str(face_locations), (10, 30), font, 1.0, (0, 0, 255), 1)
        cv2.putText(frame, str(face_distances), (10, 60), font, 1.0, (0, 0, 255), 1)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == 27:
        break






cam.release()
cv2.destroyAllWindows()
