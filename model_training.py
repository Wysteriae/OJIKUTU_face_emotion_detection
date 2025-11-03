import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

print("Loading FER2013 dataset...")
data = pd.read_csv('dataset/fer2013.csv')

print("Preprocessing data...")
pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
X = np.stack(pixels, axis=0)
X = X / 255.0
X = X.reshape(-1, 48, 48, 1)
y = to_categorical(data['emotion'], num_classes=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Building CNN model...")
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=35, batch_size=64, validation_data=(X_test, y_test), verbose=1)

print("Saving model...")
model.save('face_emotionModel.h5')

print("✅ Model training complete!")
