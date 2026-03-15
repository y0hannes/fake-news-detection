
def train_model(model, X_train, y_train):

    history = model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1
    )

    return history
