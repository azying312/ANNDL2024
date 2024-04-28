### Data Processor - Baseline Model

from sklearn.preprocessing import LabelEncoder

### Data Loader - NRRD Files
def load_images(file_paths):
    images = []
    for file_path in file_paths:
        # Load images using nrrd
        image_data, _ = nrrd.read(file_path)
        images.append(image_data)
    return np.array(images)

# Read in data
# baseline_gamaknife_data = pd.read_csv('/content/drive/Shareddrives/ANNDL2024/full_cleaned_cropped_data.csv')
baseline_gamaknife_data = pd.read_csv('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/gamma_knife_data_rotated_cropped')

# Data Split
baseline_train_data, baseline_test_data = CNN_split_data_by_patient(baseline_gamaknife_data)
print("Training data size:", len(baseline_train_data))
print("Testing data size:", len(baseline_test_data))

### MRI Data
# baseline_mri_X_train, baseline_mri_y_train = baseline_train_data['mri_file'], baseline_train_data['mri_type']
# baseline_mri_X_test, baseline_mri_y_test = baseline_test_data['mri_file'], baseline_test_data['mri_type']

augmented_mris = baseline_train_data['augmented_mri']
indices_kept = augmented_mris.dropna().index.tolist()
augmented_mris_cleaned = augmented_mris.dropna().tolist()
baseline_mri_X_train, baseline_mri_y_train = augmented_mris_cleaned, baseline_train_data['mri_type'].iloc[indices_kept]
baseline_mri_X_train, baseline_mri_y_train = augmented_mris_cleaned, baseline_train_data['mri_type'].iloc[indices_kept]


augmented_mris = baseline_test_data['augmented_mri']
indices_kept = augmented_mris.dropna().index.tolist()
augmented_mris_cleaned = augmented_mris.dropna().tolist()
baseline_mri_X_test, baseline_mri_y_test = augmented_mris_cleaned, baseline_test_data['mri_type'].iloc[indices_kept]

baseline_mri_X_train = load_images(baseline_mri_X_train)
baseline_mri_X_test = load_images(baseline_mri_X_test)

### Dose Data
# baseline_dose_X_train, baseline_dose_y_train = baseline_train_data['dose_file'], baseline_train_data['mri_type']
# baseline_dose_X_test, baseline_dose_y_test = baseline_test_data['dose_file'], baseline_test_data['mri_type']

augmented_doses = baseline_train_data['augmented_dose']
indices_kept = augmented_doses.dropna().index.tolist()
augmented_doses_cleaned = augmented_doses.dropna().tolist()
baseline_dose_X_train, baseline_dose_y_train = augmented_doses_cleaned, baseline_train_data['mri_type'].iloc[indices_kept]

augmented_doses = baseline_test_data['augmented_dose']
indices_kept = augmented_doses.dropna().index.tolist()
augmented_doses_cleaned = augmented_doses.dropna().tolist()
baseline_dose_X_test, baseline_dose_y_test = augmented_doses_cleaned, baseline_test_data['mri_type'].iloc[indices_kept]

baseline_dose_X_train = load_images(baseline_dose_X_train)
baseline_dose_X_test = load_images(baseline_dose_X_test)

### Clinical Data
baseline_X_train, baseline_y_train = baseline_train_data.iloc[:,[1, 3, 4, 5, 12]], baseline_train_data['mri_type']
baseline_X_test, baseline_y_test = baseline_test_data.iloc[:,[1, 3, 4, 5, 12]], baseline_test_data['mri_type']

# Convert categorical labels to binary labels
baseline_y_train_binary = (baseline_y_train == 'recurrence').astype(int)
baseline_y_test_binary = (baseline_y_test == 'recurrence').astype(int)

### Process data
baseline_X_train_processed = baseline_X_train.copy()
baseline_X_test_processed = baseline_X_test.copy()

# Label encode 'Primary Diagnosis'
label_encoder_diagnosis = LabelEncoder()
baseline_X_train_processed['Primary Diagnosis'] = label_encoder_diagnosis.fit_transform(baseline_X_train_processed['Primary Diagnosis'])
baseline_X_test_processed['Primary Diagnosis'] = label_encoder_diagnosis.fit_transform(baseline_X_test_processed['Primary Diagnosis'])

# Convert gender to binary 0 - Male, 1 - Female
baseline_X_train_processed['Gender'] = (baseline_X_train_processed['Gender'] == 'Female').astype(int)


### Data Processing - Ensemble Model

# Read in data
ensemble_gamaknife_data = pd.read_csv('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/gamma_knife_data_rotated_cropped')

# Data Split
# train_data, test_data, val_data = split_data_by_patient(ensemble_gamaknife_data)


train_data = ensemble_gamaknife_data[ensemble_gamaknife_data['unique_pt_id'].isin(train_ids)]
test_data = ensemble_gamaknife_data[ensemble_gamaknife_data['unique_pt_id'].isin(test_ids)]
val_data = ensemble_gamaknife_data[ensemble_gamaknife_data['unique_pt_id'].isin(val_ids)]

print("Training data size:", len(train_data))
print("Testing data size:", len(test_data))
print("Validation data size:", len(val_data))

### MRI Data
augmented_mris = train_data['augmented_mri']
indices_kept = augmented_mris.dropna().index.tolist()
augmented_mris_cleaned = augmented_mris.dropna().tolist()
mri_X_train, mri_y_train = augmented_mris_cleaned, train_data['mri_type']

augmented_mris = test_data['augmented_mri']
indices_kept = augmented_mris.dropna().index.tolist()
# print(indices_kept)
# print(len(indices_kept))
augmented_mris_cleaned = augmented_mris.dropna().tolist()
# print(augmented_mris_cleaned)
# print(len(augmented_mris_cleaned))
mri_X_test, mri_y_test = augmented_mris_cleaned, test_data['mri_type']

augmented_mris = val_data['augmented_mri']
indices_kept = augmented_mris.dropna().index.tolist()
augmented_mris_cleaned = augmented_mris.dropna().tolist()
mri_X_val, mri_y_val = augmented_mris_cleaned, val_data['mri_type']

# Convert categorical labels to binary labels
mri_y_train_binary = (mri_y_train == 'recurrence').astype(int)
mri_y_test_binary = (mri_y_test == 'recurrence').astype(int)
# print(mri_y_test)
# print(mri_y_test_binary)
mri_y_val_binary = (mri_y_val == 'recurrence').astype(int)

### Dose Data
augmented_doses = train_data['augmented_dose']
indices_kept = augmented_doses.dropna().index.tolist()
augmented_doses_cleaned = augmented_doses.dropna().tolist()
dose_X_train, dose_y_train = augmented_doses_cleaned, train_data['mri_type']


augmented_doses = test_data['augmented_dose']
indices_kept = augmented_doses.dropna().index.tolist()
augmented_doses_cleaned = augmented_doses.dropna().tolist()
dose_X_test, dose_y_test = augmented_doses_cleaned, test_data['mri_type']

augmented_doses = val_data['augmented_dose']
indices_kept = augmented_doses.dropna().index.tolist()
augmented_doses_cleaned = augmented_doses.dropna().tolist()
dose_X_val, dose_y_val = augmented_doses_cleaned, val_data['mri_type']

# Convert categorical labels to binary labels
dose_y_train_binary = (dose_y_train == 'recurrence').astype(int)
dose_y_test_binary = (dose_y_test == 'recurrence').astype(int)
dose_y_val_binary = (dose_y_val == 'recurrence').astype(int)

### Clinical Data
X_train, y_train = train_data.iloc[:,[1, 3, 4, 5, 12]], train_data['mri_type']
X_test, y_test = test_data.iloc[:,[1, 3, 4, 5, 12]], test_data['mri_type']
X_val, y_val = val_data.iloc[:,[1, 3, 4, 5, 12]], val_data['mri_type']

# Convert categorical labels to binary labels
y_train_binary = (y_train == 'recurrence').astype(int)
y_test_binary = (y_test == 'recurrence').astype(int)
y_val_binary = (y_val == 'recurrence').astype(int)

### Data Loader - NRRD Files
def load_images(file_paths):
    images = []
    for file_path in file_paths:
        # Load images using nrrd
        image_data, _ = nrrd.read(file_path)
        images.append(image_data)
    return np.array(images)
baseline_X_test_processed['Gender'] = (baseline_X_test_processed['Gender'] == 'Female').astype(int)
