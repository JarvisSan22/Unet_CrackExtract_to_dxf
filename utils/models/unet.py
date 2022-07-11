#Standard U Net 
from keras import backend as K
import tensorflow as tf 
from tensorflow.keras import backend as K
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)



def model_UNET(imagesize,LR=1e-4,weights_input=None):

    inputs = inputs = tf.keras.layers.Input(imagesize)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 =tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 =tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 =tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 =tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 =tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 =tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 =tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)
    #Expansive path Back up 　
    up6 = tf.keras.layers.Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")( tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.keras.layers.Concatenate(axis=3)([drop4,up6])
    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = tf.keras.layers.Concatenate(axis=3)([conv3,up7])
    conv7 =tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = tf.keras.layers.Concatenate(axis=3)([conv2,up8])
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = tf.keras.layers.Concatenate(axis=3)([conv1,up9])
    conv9 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 =tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)

    conv10 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR), loss="binary_crossentropy", metrics=["accuracy", jacard_coef])
    #model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.SGD(lr=1e-4), metrics=["accuracy", jacard_coef])
    
    if weights_input:
        model.load_weights(weights_input)
    print(model.summary())
    return model
