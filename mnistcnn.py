import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255., X_test/255.

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# Hyperparameters
EPOCHS = 10
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
                                    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation='softmax')
])

optim = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=128, callbacks=[callbacks], validation_split=0.1)

model.save('./mnist_cnn.h5')
print("Model Saved!")
