### Data Augmentation

import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import nrrd

def augment_data(data_frame, output_dir, mri_file_type, dose_file_type):
    os.makedirs(output_dir, exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=90,
        fill_mode='nearest'
    )

    # Create empty lists to store the augmented data
    augmented_data = []
    for index, row in data_frame.iterrows():
        mri_file_name = row[mri_file_type]
        dose_file_name = row[dose_file_type]

        mri_original_file_name = os.path.basename(mri_file_name)
        dose_original_file_name = os.path.basename(dose_file_name)
        mri_data, mri_header = nrrd.read(mri_file_name)
        dose_data, dose_header = nrrd.read(dose_file_name)

        # Perform data augmentation for MRI
        for rotation_angle in [90, 180, 270]:
            augmented_mri_images = []
            for i in range(mri_data.shape[2]):
                augmented_mri_data = datagen.apply_transform(np.expand_dims(mri_data[:, :, i], axis=-1), {'theta': rotation_angle})
                augmented_mri_images.append(augmented_mri_data.squeeze())
            augmented_mri_combined = np.stack(augmented_mri_images, axis=-1)
            augmented_mri_path = f"{output_dir}/augmented_mri_{mri_original_file_name}_{index}_rotation_{rotation_angle}.nrrd"

            # Write augmened files
            nrrd.write(augmented_mri_path, augmented_mri_combined, mri_header)

            augmented_row = row.copy()
            augmented_row['augmented_mri_path'] = augmented_mri_path
            augmented_data.append(augmented_row)

        ### Dose
        for rotation_angle in [90, 180, 270]:
            augmented_dose_images = []
            for i in range(dose_data.shape[2]):
                augmented_dose_data = datagen.apply_transform(np.expand_dims(dose_data[:, :, i], axis=-1), {'theta': rotation_angle})
                augmented_dose_images.append(augmented_dose_data.squeeze())
            augmented_dose_combined = np.stack(augmented_dose_images, axis=-1)
            augmented_dose_path = f"{output_dir}/augmented_dose_{dose_original_file_name}_{index}_rotation_{rotation_angle}.nrrd"

            # Write augmented dose slices to NRRD file
            nrrd.write(augmented_dose_path, augmented_dose_combined, dose_header)
            augmented_row = row.copy()
            augmented_row['augmented_dose_path'] = augmented_dose_path
            augmented_data.append(augmented_row)

    # Create a DataFrame from the augmented data
    augmented_df = pd.DataFrame(augmented_data)

    return augmented_df

output_directory = "/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/images_augmented/"

gamaknife_data = pd.read_csv('/content/drive/Shareddrives/ANNDL2024/full_cleaned_cropped_data.csv')

mri_file_type = 'mri_file'
dose_file_type = 'dose_file'

augmented_df = augment_data(gamaknife_data, output_directory, mri_file_type, dose_file_type)

### Get Augmented Data

import os
import pandas as pd

recurrence_subset = gamaknife_data[gamaknife_data['mri_type'] == 'recurrence']
stable_subset = gamaknife_data[gamaknife_data['mri_type'] == 'stable']

file_list = recurrence_subset['les_file']
rotated_files_dir = "/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/images_augmented/"
rotated_files = os.listdir(rotated_files_dir)

new_rows = []
for index, row in recurrence_subset.iterrows():
    lesion_filename = os.path.basename(row['les_file'])
    matching_files = [f for f in rotated_files if f.startswith("augmented_mri_" + lesion_filename) or f.startswith("augmented_dose_" + lesion_filename)]

    for rotated_file in matching_files:
        new_row = row.copy()
        if 'augmented_mri_' in rotated_file:
            augmented_type = 'augmented_mri'
        elif 'augmented_dose_' in rotated_file:
            augmented_type = 'augmented_dose'
        else:
            continue
        new_row[augmented_type] = os.path.join(rotated_files_dir, rotated_file)
        new_rows.append(new_row)

expanded_df = pd.DataFrame(new_rows)

def combine_rows(group):
    # Get the non-rotated MRI and dose files
    mri_file = group['mri_file'].iloc[0]
    dose_file = group['dose_file'].iloc[0]

    # Get the rotated MRI and dose files
    rotated_mri_files = group['augmented_mri'].dropna().tolist()
    rotated_dose_files = group['augmented_dose'].dropna().tolist()

    # Combine the rotated files into three pairs
    combined_data = []
    for i in range(3):
        combined_data.append({
            'les_file': group['les_file'].iloc[0],
            'mri_file': mri_file,
            'dose_file': dose_file,
            'augmented_mri': rotated_mri_files[i],
            'augmented_dose': rotated_dose_files[i]
        })
    other_cols = group.iloc[0].drop(['les_file', 'mri_file', 'dose_file', 'augmented_mri', 'augmented_dose']).to_dict()
    combined_data = [{**data, **other_cols} for data in combined_data]

    return pd.DataFrame(combined_data)

original_cols = expanded_df.columns.tolist()
grouped_df = expanded_df.groupby('les_file')
combined_rows = grouped_df.apply(combine_rows).reset_index(drop=True)
combined_rows = combined_rows[original_cols]

# Merge with stable data
gamaknife_data['augmented_mri'] = gamaknife_data['mri_file'].copy()
gamaknife_data['augmented_dose'] = gamaknife_data['dose_file'].copy()

merged_df = pd.concat([combined_rows, gamaknife_data], ignore_index=True)

merged_df.to_csv('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/gamma_knife_data_rotated_cropped', index=False)
