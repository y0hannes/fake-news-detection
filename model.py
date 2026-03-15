from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model(input_size):

    model = Sequential([
        Input(shape=(input_size,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model