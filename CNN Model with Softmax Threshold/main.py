
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models
from keras.datasets import cifar10



(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')

'''
model = models.load_model('image_classifier.model')


img = cv.imread('shoes.jpg')

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)


index = np.argmax(prediction)
maxValue = np.max(prediction)
meanValue = np.mean(prediction)
maxDeviationPercent = (meanValue / maxValue) * 100


print('__________STATS__________')
print(f"Values from softmax layer: {prediction}")
print(f"Highest softmax value: {maxValue}")
print(f"Average softmax value: {meanValue}")
print(f"Percentage difference between the Highest and Average Values: {maxDeviationPercent}%")
print()

print("__________PREDICTION__________")
if maxDeviationPercent > 14.5:
    print('Image is not recognized by Neural Network')
else:
    print(f'Prediction is {class_names[index]} ')

'''







