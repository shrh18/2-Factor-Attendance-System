import cv2
from pyzbar import pyzbar
from pyzbar.pyzbar import decode
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Define paths
face_folder_path = "C:/Users/lenovo/Desktop/Academics/BTech/SEM VIII/Deep Learning/Course Project/Faces" # Folder where face images are stored
excel_file_path = "C:/Users/lenovo/Desktop/Academics/BTech/SEM VIII/Deep Learning/Course Project/Attendance.xlsx" # Path to the Excel file
model_weights_path = "C:/Users/lenovo/Desktop/Academics/BTech/SEM VIII/Deep Learning/Course Project/facenet_keras.h5"  # Update with the path to your model weights file

# Define model parameters
input_shape = (64, 64, 3)
num_classes = len(os.listdir(face_folder_path))

# Define CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='linear'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.save('test.h5')
# model.load_weights(model_weights_path)
# # Load pre-trained weights if available
# if os.path.exists('model_weights.h5'):
#     model.load_weights('model_weights.h5')
#     print("Loaded pre-trained weights")

# Load face images and their filenames
faces = []
face_filenames = []
for face_filename in os.listdir(face_folder_path):
    face = cv2.imread(os.path.join(face_folder_path, face_filename))
    face = cv2.resize(face, input_shape[:2])
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    faces.append(face)
    face_filenames.append(face_filename)

# Convert faces and filenames to numpy arrays
faces = np.array(faces)
face_filenames = np.array(face_filenames)

# Convert face filenames to GR numbers (assuming filenames are in the format "GR-xxx.jpg")
gr_numbers = [filename.split('.')[0] for filename in face_filenames]
print(gr_numbers)

# # Create Excel workbook and sheet
# workbook = Workbook()
# sheet = workbook.active

# Create Excel sheet
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.title = "Attendance"
sheet.append(["GR Number", "Face", "QR"])

# Set column names in Excel sheet
sheet['A1'] = 'GR Number'
sheet['B1'] = 'Face'
sheet['C1'] = 'QR'

# Set font for column names
bold_font = Font(bold=True)
sheet['A1'].font = bold_font
sheet['B1'].font = bold_font
sheet['C1'].font = bold_font

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # ----------------------------------------QR Code---------------------------
    # Capture frame from webcam
    ret, frame = cap.read()
    
    # Convert frame to grayscale for QR code detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect QR codes
    decoded_objs = pyzbar.decode(gray)
    qr_detected = False

    for obj in decoded_objs:
        qr_detected = True
        # Draw bounding box around QR code
        x, y, w, h = obj.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # path2 = 'C:/Users/lenovo/Desktop/Academics/BTech/SEM VIII/Deep Learning/Course Project/QR_codes'
    # images = []
    # classNames2 = []
    # mylist2 = os.listdir(path2)

    # for cls2 in mylist2:
    #     curImg=cv2.imread(f'{path2}/{cls2}')
    #     images.append(curImg)
    #     classNames2.append(os.path.splitext(cls2)[0])
    # print(classNames2)

    # while True:
    #     succ,frame = cap.read()

    #     for code in decode(frame):
    #         print(code.type)
    #         print(code.data.decode('utf-8'))
    #     cv2.imshow('test',frame)
    #     cv2.waitKey(1)
    #     key = cv2.waitKey(1) & 0xFF

    # # Exit loop
    #     if key == ord("q"):
    #         x=1
    #         break
            
    # -----------------------------------------------------------------------------
    # Convert frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces_detected = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = faces_detected.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Extract face from the frame
        face = rgb_frame[y:y + h, x:x + w]
        face = cv2.resize(face, input_shape[:2])

        # Preprocess face for prediction
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Make prediction using the trained CNN model
        prediction = model.predict(face)
        predicted_class_index = np.argmax(prediction)
        predicted_class = gr_numbers[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        # Draw bounding box around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display GR Number on the frame
        cv2.putText(frame, "GR Number: " + predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Record result in Excel sheet
        row = [predicted_class, "True" if confidence > 0.8 else "False", "True" if qr_detected else "False"]
        sheet.append(row)

    # Display frame
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit loop
    if key == ord("q"):
        break
# ---------------------------
        cap.release()
        cv2.destroyAllWindows()
        workbook.save(excel_file_path)
        model.save_weights('model_weights.h5')
        print("Model weights saved")