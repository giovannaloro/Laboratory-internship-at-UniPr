
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import pandas as pd
import os 
from to_image import to_image
import tensorflow_addons as tfa




 
model = keras.Sequential(
    [
    layers.Conv2D(filters=10, kernel_size=(3,3),),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=1),
    layers.Conv2D(filters=5, kernel_size=(3,3), activation="relu"),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=1),
    layers.GlobalAveragePooling2D(),
    layers.Dense(units=5, activation="softmax")
    ]
)


os.chdir("..")
df = pd.read_csv('augmented_dataset/dataset_ctgan_smotenc_generated_best_onehot.csv')
X_train = df.iloc[:,0:45]
y_train = df.iloc[:,45:50]
X_train["empty1"] = 0
X_train["empty2"] = 0
X_train["empty3"] = 0
X_train["empty4"] = 0
X_train = to_image(X_train,7,7,1)
df = pd.read_csv('dataset/ML_MED_Dataset_validation_preprocessed_full.csv')
X_validation = df.iloc[:,1:46]
y_validation = df.iloc[:,46:52]
X_validation["empty1"] = 0
X_validation["empty2"] = 0
X_validation["empty3"] = 0
X_validation["empty4"] = 0
X_validation = to_image(X_validation,7,7,1)

for i in range(100):
    model = keras.Sequential(
    [
    layers.Conv2D(filters=10, kernel_size=(3,3)),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=1),
    layers.Conv2D(filters=5, kernel_size=(3,3), activation="relu"),
    layers.Activation("relu"),
    layers.BatchNormalization(axis=1),
    layers.GlobalAveragePooling2D(),
    layers.Dense(units=5, activation="softmax")
    ]
    )
    model.compile(optimizer=keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.01), metrics=["categorical_accuracy",tfa.metrics.F1Score(average='macro',num_classes=5,name="macro_f1"),tfa.metrics.F1Score(average='micro',num_classes=5,name="micro_f1"),"F1Score"], loss="categorical_crossentropy" )
    model.fit(x=X_train, y=y_train, batch_size=8, epochs=60, shuffle=True, verbose=0) 
    model.save(f"_model_{i}.h5")
    metric = model.evaluate(x=X_validation, y=y_validation)
    print(f"model {i} stats are: {metric[2]}")
    