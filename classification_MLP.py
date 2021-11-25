# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import pydot

# %%
tf.__version__
# %%
keras.__version__
# %%
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# %%
X_train_full.shape
# %%
X_train_full.dtype
# %%
#divide by 255 to range pixels equal 0 to 1
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# %%
#create the variable class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# %%
#See some imagens of dataset
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
# %%
class_names[y_train[0]]
# %%
#Show an image
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()
# %%
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
# %%
#different way with random seed
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(300, activation="relu"),
keras.layers.Dense(100, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])
# %%
model.summary()
# %%
#name of layers
model.layers
# %%
#show the process of model
keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
# %%
weights, biases = model.layers[1].get_weights()
# %%
weights
# %%
weights.shape
# %%
#all the biases equal to 0. In another moment we will chance this
biases
# %%
model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",
metrics=["accuracy"])
# %%
#Now the model is ready to be trained
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
# %%
history.params
# %%
history.history.keys()
# %%
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
# %%
#evaluate model
model.evaluate(X_test, y_test)
# %%
# predict "deprecated"
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
# %%
#new method to predict
y_pred = np.argmax(model.predict(X_new), axis=-1)
y_pred
# %%
#name of figures predictes
np.array(class_names)[y_pred]
# %%
#show the three figures predicted
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
# %%
#OBS: THIS MODEL IS NOT GOOD BECAUSE IT HAS A BIG LOSS