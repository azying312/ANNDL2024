# CNN Model
def create_regularized_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

## MRI Class Weights

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Class distb
class_labels = np.unique(mri_y_train_binary)
class_counts = np.bincount(mri_y_train_binary)

# Calculate class weights
class_weights = compute_class_weight(class_weight = 'balanced', classes=np.unique(mri_y_train_binary), y=mri_y_train_binary)
class_weights_dict = dict(zip(np.unique(mri_y_train_binary), class_weights))

### MRI Files -- CNN

# Load images for training and testing
mri_X_train_images = load_images(mri_X_train)
mri_X_test_images = load_images(mri_X_test)

# Make CNN model
input_shape = (82, 66, 39)
# mri_model = create_cnn_model(input_shape)

# Train the CNN model
mri_model = create_regularized_cnn_model(input_shape)
history = mri_model.fit(mri_X_train_images, mri_y_train_binary,
                        class_weight=class_weights_dict,
                        epochs=50,
                        validation_split=0.2,
                        callbacks=[early_stopping])

# Evaluate the model
mri_test_loss, mri_test_accuracy = mri_model.evaluate(mri_X_test_images, mri_y_test_binary)
print("Test Loss:", mri_test_loss)
print("Test Accuracy:", mri_test_accuracy)

### Dose Files -- CNN

# Class Weights

# Class distb
class_labels = np.unique(dose_y_train_binary)
class_counts = np.bincount(dose_y_train_binary)

# Calculate class weights
class_weights = compute_class_weight(class_weight = 'balanced', classes=np.unique(dose_y_train_binary), y=dose_y_train_binary)
class_weights_dict = dict(zip(np.unique(mri_y_train_binary), class_weights))

# Load images for training and testing
dose_X_train_images = load_images(dose_X_train)
dose_X_test_images = load_images(dose_X_test)

# Make CNN model
input_shape = (82, 66, 39)
# dose_model = create_regularized_cnn_model(input_shape)

# Train the CNN model
# history = dose_model.fit(dose_X_train_images, dose_y_train_binary, epochs=10, batch_size=32, validation_split=0.2)
dose_model = create_regularized_cnn_model(input_shape)
history = dose_model.fit(dose_X_train_images, dose_y_train_binary,
                        class_weight=class_weights_dict,
                        epochs=50,
                        validation_split=0.2,
                        callbacks=[early_stopping])
# Evaluate the model
dose_test_loss, dose_test_accuracy = dose_model.evaluate(dose_X_test_images, dose_y_test_binary)
print("Test Loss:", dose_test_loss)
print("Test Accuracy:", dose_test_accuracy)

### Descision Tree

### Process data
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()
X_val_processed = X_val.copy()

# Label encode 'Primary Diagnosis'
label_encoder_diagnosis = LabelEncoder()
X_train_processed['Primary Diagnosis'] = label_encoder_diagnosis.fit_transform(X_train_processed['Primary Diagnosis'])
X_test_processed['Primary Diagnosis'] = label_encoder_diagnosis.fit_transform(X_test_processed['Primary Diagnosis'])
X_val_processed['Primary Diagnosis'] = label_encoder_diagnosis.fit_transform(X_val_processed['Primary Diagnosis'])

# Convert gender to binary 0 - Male, 1 - Female
X_train_processed['Gender'] = (X_train_processed['Gender'] == 'Female').astype(int)
X_test_processed['Gender'] = (X_test_processed['Gender'] == 'Female').astype(int)
X_val_processed['Gender'] = (X_val_processed['Gender'] == 'Female').astype(int)

### Resample the training data to balance the classes
ros = RandomOverSampler(random_state=0)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_processed, y_train_binary)

### Decision tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_resampled, y_train_resampled)
y_pred = decision_tree.predict(X_test_processed)

# Evaluate the model
accuracy = accuracy_score(y_test_binary, y_pred)
print("Decision Tree Accuracy after resampling:", accuracy)

### Assemble results

predictions_cnn_mri_val = mri_model.predict(load_images(mri_X_val))
predictions_cnn_dose_val = dose_model.predict(load_images(dose_X_val))
predictions_decision_tree_val = decision_tree.predict_proba(X_val_processed)[:, 1] # might need to give more weight?

combined_predictions_val = np.column_stack((predictions_cnn_mri_val, predictions_cnn_dose_val, predictions_decision_tree_val))

# Train full model
meta_model = LogisticRegression()
meta_model.fit(combined_predictions_val, y_val_binary)

ensemble_predictions_val = meta_model.predict(combined_predictions_val)
ensemble_accuracy_val = accuracy_score(y_val_binary, ensemble_predictions_val)

print("Ensemble Accuracy:", ensemble_accuracy_val)
