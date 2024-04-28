### 3D CNN
### Code from doi: 10.1038/s41597-023-02683-1 and written by Yibin Wang
# Only minor edits

import keras
from keras import layers

def get_model():

    # input_l = keras.Input((82, 66, 39, 1), name="Les_Input")  # Depend on the cropped lesion shape (could be different)
    input_m = keras.Input((82, 66, 39, 1), name="MRI_Input")

    # x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(input_l)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(input_m)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=(3,3,3), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # output_l = layers.Dense(units=128, activation="relu")(x)
    output_m = layers.Dense(units=128, activation="relu")(x)

    input_d = keras.Input((82, 66, 39, 1), name="Dose_Input")  # Dose should keep the same input shape as the lesion

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(input_d)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output_d = layers.Dense(units=128, activation="relu")(x)

    input_c = keras.Input((5), name="Clinical_Input")
    x = layers.Dense(units=32, activation="relu")(input_c)
    output_c = layers.Dense(units=128, activation="relu")(x)

    # merge_both_blocks = layers.concatenate([output_l, output_d, output_c])
    merge_both_blocks = layers.concatenate([output_m, output_d, output_c])

    outputs = layers.Dense(1, activation="sigmoid")(merge_both_blocks)
    # Define the model
    # model = keras.Model(inputs=[input_l, input_d, input_c], outputs = [outputs], name="MultiInput")
    model = keras.Model(inputs=[input_m, input_d, input_c], outputs = [outputs], name="MultiInput")
    return model

# Build model
model = get_model()
# model.summary()

# Plot model
from tensorflow.keras.utils import plot_model
from IPython.core.display import Image
plot_model(model, "MultiInput.png", show_shapes=True)
