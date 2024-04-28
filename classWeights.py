### Class Weights

# Class distb
class_labels = np.unique(mri_y_train_binary)
class_counts = np.bincount(mri_y_train_binary)

# Calculate class weights
class_weights = compute_class_weight(class_weight = 'balanced', classes=np.unique(mri_y_train_binary), y=mri_y_train_binary)
class_weights_dict = dict(zip(np.unique(mri_y_train_binary), class_weights))
