import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np

path = r"Data"
RANDOM_SEED = 170389
IMAGE_SIZE = (124, 124)
BATCH_SIZE = 32
EPOCHS = 50
INIT_LR = 1e-4

LABELS = ["live","spoof"]


def plot_images(images, labels):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for images, LABELS in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

    plt.show()


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.3,
    subset="training",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.3,
    subset="validation",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

""""This is better for CPU forms
    Going to use Model direct Augmentation to improve performance
"""
# augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

import LivenessModel as LM

model = LM.LivenessModels.build(IMAGE_SIZE + (3,), classes=len(LABELS))



optmizer = keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(
    optimizer=optmizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


callback_es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="auto",
    patience=EPOCHS,
    baseline=None,
    restore_best_weights=True,
)

H = model.fit(
    train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback_es]
)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

img = tf.keras.preprocessing.image.load_img(
    "Data/spoof/0001_00_00_01_198.jpg", target_size=IMAGE_SIZE
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

model.save("model.h5")

model.save("model.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("plot_loss.png")

plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("plot_acc.png")

predictions = model.predict(val_ds.as_numpy_iterator(), batch_size=BATCH_SIZE)

from sklearn.metrics import classification_report

true_categories = tf.concat([y for x, y in val_ds], axis=0)

print(classification_report(true_categories,
	predictions.round(1), target_names=LABELS))