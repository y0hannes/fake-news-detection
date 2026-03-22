from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model, X_train, y_train):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks
    )

    return history
