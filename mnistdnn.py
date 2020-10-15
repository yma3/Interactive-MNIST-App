import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255., X_test/255.

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# Hyperparameters
EPOCHS = 100
LR = 1e-3

# Define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

# Train MNIST
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(2048, activation='relu'),
                                    tf.keras.layers.Dense(1024, activation='relu'),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
optim = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.1)

model.save('./models/overfit_mnist_dnn.h5')
print("Model Saved!")
