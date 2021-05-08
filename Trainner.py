import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime as dt

path = r"Data"
now = dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
outpath = r"Output/"+now+"/"
os.mkdir(outpath)
RANDOM_SEED = 170389
IMAGE_SIZE = (124, 124)
BATCH_SIZE = 32
EPOCHS = 50
INIT_LR = 1e-4

LABELS = ["live","spoof"]

dir_list = os.listdir(path)
data = []
labels = []

for dir in dir_list:
    tmp_path = path+"/"+dir
    for file in os.listdir(tmp_path):
        label = dir
        image = cv2.imread(tmp_path+"/"+file)
        try:
            image = cv2.resize(image, IMAGE_SIZE)
        except:
            print(file)
            quit()

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)


data = np.array(data, dtype="float")/255.0
# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = keras.utils.to_categorical(labels, 2)
with open(outpath+'classes.npy', 'wb') as np_file:
    np.save(np_file, le.classes_)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.3, random_state=RANDOM_SEED)

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
    patience=2,
    baseline=None,
    restore_best_weights=True,
)

H = model.fit(x=trainX, y=trainY, epochs=EPOCHS, validation_data=(testX,testY), callbacks=[callback_es])

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True, to_file=outpath+"model.png")

img = tf.keras.preprocessing.image.load_img(
    "Data/spoof/0001_00_00_01_198.jpg", target_size=IMAGE_SIZE
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

model.save(outpath+"model.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(outpath+"plot_loss.png")

plt.figure()
plt.plot(H.history["accuracy"], label="train_accuracy")
plt.plot(H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig(outpath+"plot_acc.png")

result = model.evaluate(testX, batch_size=BATCH_SIZE)
print(f"Loss is {result[0]} and accuracy is {result[1]}")


from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(testX, batch_size=BATCH_SIZE)
with open(outpath+"results","w") as result_file:
    result_file.write(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
    result_file.write("\n============Confusion Matrix===================\n")
    result_file.write(str(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))))
    print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))

quit()