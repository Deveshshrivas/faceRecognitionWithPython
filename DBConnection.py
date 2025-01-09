import cv2
import dlib
import os
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv
from scipy.spatial import distance

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection details from environment variables
mongo_uri = os.getenv('MONGO_URI')
database_name = os.getenv('DATABASE_NAME')

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db['users']


#Path of Dlib models
# shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")


# Initialize Dlib models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat/dlib_face_recognition_resnet_model_v1.dat")

def capture_images(name, num_samples=100):
    video_capture = cv2.VideoCapture(0)
    count = 0

    while count < num_samples:
        ret, frame = video_capture.read()
        if ret:
            # Detect faces
            faces = detector(frame, 1)
            if len(faces) > 0:
                for face in faces:
                    # Align face using landmarks
                    shape = shape_predictor(frame, face)
                    face_chip = dlib.get_face_chip(frame, shape)

                    # Convert to binary and save in database
                    _, buffer = cv2.imencode('.jpg', face_chip)
                    binary_image = Binary(buffer)

                    collection.insert_one({
                        "name": name,
                        "image_id": count,
                        "image_data": binary_image
                    })

                    print(f"Image {count + 1} captured and saved to database")
                    cv2.imshow("Captured Image", face_chip)
                    cv2.waitKey(500)  # Wait before capturing the next image
                    count += 1

                    if count == 30:
                        print("Please turn your face to the left.")
                    elif count == 70:
                        print("Please turn your face to the right.")
            else:
                print("No face detected.")
        else:
            print("Failed to capture image")
            break

    video_capture.release()
    cv2.destroyAllWindows()

def recognize_faces():
    video_capture = cv2.VideoCapture(0)

    # Load known faces and their encodings
    known_face_encodings = []
    known_face_names = []

    for user in collection.find():
        name = user['name']
        binary_image = user['image_data']
        image_data = cv2.imdecode(np.frombuffer(binary_image, np.uint8), cv2.IMREAD_COLOR)

        # Detect and encode face
        faces = detector(image_data, 1)
        if len(faces) > 0:
            shape = shape_predictor(image_data, faces[0])
            face_encoding = np.array(face_rec_model.compute_face_descriptor(image_data, shape))
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect faces in the frame
        faces = detector(frame, 1)
        for face in faces:
            shape = shape_predictor(frame, face)
            face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))

            # Match the face encoding with known faces
            distances = [distance.euclidean(face_encoding, known_enc) for known_enc in known_face_encodings]
            min_distance = min(distances) if distances else None
            name = "Unknown"

            if min_distance is not None and min_distance < 0.6:  # Adjust threshold as needed
                name = known_face_names[distances.index(min_distance)]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, name, (face.left(), face.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter the name of the person to capture: ")
    capture_images(person_name)
    recognize_faces()
