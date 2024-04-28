# Fcn to check mixed data in every set

ensemble_gamaknife_data = pd.read_csv('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/gamma_knife_data_rotated_cropped')
unique_patient_ids = ensemble_gamaknife_data['unique_pt_id'].unique()
np.random.shuffle(unique_patient_ids)

def check_recurrence_type(df):
    return 'recurrence' in df['mri_type'].values

while True:
    train_ids, test_val_ids = train_test_split(unique_patient_ids, test_size=0.4)
    test_ids, val_ids = train_test_split(test_val_ids, test_size=0.5)

    train_df = ensemble_gamaknife_data[ensemble_gamaknife_data['unique_pt_id'].isin(train_ids)]
    test_df = ensemble_gamaknife_data[ensemble_gamaknife_data['unique_pt_id'].isin(test_ids)]
    val_df = ensemble_gamaknife_data[ensemble_gamaknife_data['unique_pt_id'].isin(val_ids)]

    if check_recurrence_type(train_df) and check_recurrence_type(test_df) and check_recurrence_type(val_df):
        break

### Split data into train and test sets
def CNN_split_data_by_patient(data, test_size=0.2, random_state=None):
    def check_recurrence_type(df):
        return 'recurrence' in df['mri_type'].values

    while True:
        unique_patient_ids = data['unique_pt_id'].unique()
        # Split indices
        train_ids, test_ids = train_test_split(unique_patient_ids, test_size=test_size, random_state=random_state)

        train_data = data[data['unique_pt_id'].isin(train_ids)]
        test_data = data[data['unique_pt_id'].isin(test_ids)]

        if check_recurrence_type(train_data) and check_recurrence_type(test_data):
            # Reset index
            train_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            return train_data, test_data

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
