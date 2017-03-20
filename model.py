import csv
import cv2
import numpy as np

lines = []

# Track1 Data
with open('./data/driving_log.csv') as csvfile1:
   reader1 = csv.reader(csvfile1)
   for line1 in reader1:
      lines.append(line1)

# Track2 Data
with open('./data2/driving_log.csv') as csvfile2:
   reader2 = csv.reader(csvfile2)
   for line2 in reader2:
      lines.append(line2)

images = []
measurements = []

correction = 0.2

for line in lines:

    # Use all 3 Camera Angles
    for angle in range(0,3):
        source_path = line[angle]
        filename = source_path.split('/')[-1]
        current_path = filename
        #print(current_path)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])

        # Add Adjustments for Left Camera and Right Camera.
        if (angle == 1): measurement += correction
        if (angle == 2): measurement -= correction  
        #print("Angle: "+str(angle)+"\t"+"Steer:"+str(measurement))
        measurements.append(measurement)
        
        # Data Augmentation - Flipped Images and Measurements
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)




X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))


model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True,epochs=10,verbose=1)

model.save('model.h5')
exit()


