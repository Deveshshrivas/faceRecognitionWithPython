import cv2
import dlib
import os
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection details from environment variables
mongo_uri = os.getenv('MONGO_URI')
database_name = os.getenv('DATABASE_NAME')

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db['users']

def capture_images(name, num_samples=20):
    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)
    count = 0

    while count < num_samples:
        ret, frame = video_capture.read()
        if ret:
            # Convert the image to binary
            _, buffer = cv2.imencode('.jpg', frame)
            binary_image = Binary(buffer)

            # Insert the image into the database
            collection.insert_one({
                "name": name,
                "image_id": count,
                "image_data": binary_image
            })

            print(f"Image {count + 1} captured and saved to database")
            cv2.imshow("Captured Image", frame)
            cv2.waitKey(500)  # Wait for 500 milliseconds before capturing the next image
            count += 1
            if count == 7:
                print("Please turn your face to the left.")
            elif count == 14:
                print("Please turn your face to the right.")
        else:
            print("Failed to capture image")
            break

    video_capture.release()
    cv2.destroyAllWindows()

def recognize_faces():
    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    # Load the captured images and learn how to recognize them
    detector = dlib.get_frontal_face_detector()
    known_face_encodings = []
    known_face_names = []

    for user in collection.find():
        name = user['name']
        binary_image = user['image_data']
        image_data = cv2.imdecode(np.frombuffer(binary_image, np.uint8), cv2.IMREAD_COLOR)
        known_faces = detector(image_data, 1)
        if len(known_faces) > 0:
            known_face_encoding = known_faces[0]
            known_face_encodings.append(known_face_encoding)
            known_face_names.append(name)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect faces in the frame
        faces = detector(frame, 1)

        for face in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

            # Compare the detected face with known faces
            name = "Unknown"
            for known_face_encoding, known_name in zip(known_face_encodings, known_face_names):
                if (face.left() == known_face_encoding.left() and face.top() == known_face_encoding.top() and
                    face.right() == known_face_encoding.right() and face.bottom() == known_face_encoding.bottom()):
                    name = known_name
                    break

            # Draw the name below the face
            cv2.rectangle(frame, (face.left(), face.bottom() - 35), (face.right(), face.bottom()), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (face.left() + 6, face.bottom() - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter the name of the person to capture: ")
    capture_images(person_name)
    recognize_faces()