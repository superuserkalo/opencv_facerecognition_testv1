import cv2
from mtcnn import MTCNN

detector = MTCNN()

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cam.read()

    if not ret:
        raise IOError("Cannot read data")

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize the frame to 1/4th the size for faster processing
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

    # Detect faces in the resized frame
    faces = detector.detect_faces(small_frame)

    # Initialize a variable to store the biggest face's bounding box
    biggest_face = (0, 0, 0, 0)
    new_frame = small_frame*4

    for face in faces:
        (x, y, w, h) = face['box']
        # Scale the bounding box back to the original frame size
        # Draw the rectangle around each detected face
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the biggest face based on width
        if w > biggest_face[2]:
            biggest_face = (x, y, w, h)
!
    # Optionally, draw a special rectangle around the biggest face
    if biggest_face != (0, 0, 0, 0):
        x, y, w, h = biggest_face
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('cam', new_frame)

    # Break the loop when 'ESC' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()
