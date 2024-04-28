# Read in data
gamaknife_data = pd.read_csv('/content/drive/Shareddrives/ANNDL2024/full_cleaned_data.csv')

### Mapping Lesion Masks to the MRI images

# match the lesion to mri images
lesion_to_mri_mapping = {}

for mri_file in MR_files:
  directory, filename = os.path.split(mri_file)
  all_files = os.listdir(directory)
  other_files = [os.path.join(directory, file) for file in all_files if "_MR_" not in os.path.join(directory, file) and "_dose" not in os.path.join(directory, file)]
  lesion_to_mri_mapping[mri_file] = other_files

### Functions for Data Processing
max_size = (82, 66, 39)

# Pad Function
def pad_image(image, target_size):
    # Calculate the amount of padding needed on each side
    pad_height = max(0, target_size[0] - image.shape[0])
    pad_width = max(0, target_size[1] - image.shape[1])
    pad_depth = max(0, target_size[2] - image.shape[2])

    # Calculate the padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_front = pad_depth // 2
    pad_back = pad_depth - pad_front

    # Pad the image
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back)), mode='constant')

    return padded_image

def overlay(scan_file, les_mask):
  scan_data, header = nrrd.read(scan_file)
  les_data, header = nrrd.read(les_mask)
  overlayed_image = scan_data * les_data
  return overlayed_image

# Crop function
def crop_les(d):
    true_points = np.argwhere(d)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = d[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]
    return cropped_arr

### Cropping the Lesion Masks for MRI

# loop through keys
for mri_image in lesion_to_mri_mapping.keys():
  les_list = lesion_to_mri_mapping[mri_image]

  for les_mask in les_list:
    # Overlaying the Lesion Masks over MRI & Cropping regions of interest
    overlayed_image = overlay(mri_image, les_mask)
    cropped_mri_image = crop_les(overlayed_image)
    padded_mri_image = pad_image(cropped_mri_image, max_size)

    directory, filename = os.path.split(les_mask)
    dir2, fi2 = os.path.split(directory)
    real_dir, fi2 = os.path.split(dir2)

    cropped_directory = os.path.join(real_dir, "mri_cropped/")
    os.makedirs(cropped_directory, exist_ok=True)

    # Save to new folder
    new_file_path = os.path.join(cropped_directory, filename)
    nrrd.write(new_file_path, padded_mri_image)

### Dose cropping

### Mapping Lesion Masks to the Dose images

# match the lesion to mri images
lesion_to_dose_mapping = {}

for dose_file in DOSE_files:
  directory, filename = os.path.split(dose_file)
  all_files = os.listdir(directory)
  other_files = [os.path.join(directory, file) for file in all_files if "_MR_" not in os.path.join(directory, file) and "_dose" not in os.path.join(directory, file)]
  lesion_to_dose_mapping[dose_file] = other_files

# loop through keys
for dose_image in lesion_to_dose_mapping.keys():
  les_list = lesion_to_dose_mapping[dose_image]

  for les_mask in les_list:
    # Overlaying the Lesion Masks over MRI & Cropping regions of interest
    overlayed_image = overlay(dose_image, les_mask)
    cropped_dose_image = crop_les(overlayed_image)
    padded_dose_image = pad_image(cropped_dose_image, max_size)

    directory, filename = os.path.split(les_mask)
    dir2, fi2 = os.path.split(directory)
    real_dir, fi2 = os.path.split(dir2)

    cropped_directory = os.path.join(real_dir, "dose_cropped/")
    os.makedirs(cropped_directory, exist_ok=True)

    # Save to new folder
    new_file_path = os.path.join(cropped_directory, filename)
    nrrd.write(new_file_path, padded_dose_image)

mri_crops_list = os.listdir('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/mri_cropped')
dose_crops_list = os.listdir('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/dose_cropped')

### Match Cropped Data

for index, row in gamaknife_data.iterrows():
    file_name = os.path.basename(row['les_file'])

    # MRI
    new_path = os.path.join('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/mri_cropped', file_name)
    gamaknife_data.at[index, 'mri_file'] = new_path

    # Dose
    new_path = os.path.join('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/dose_cropped', file_name)
    gamaknife_data.at[index, 'dose_file'] = new_path

### Write Cropped and Cleaned Table
gamaknife_data.to_csv('/content/drive/Shareddrives/ANNDL2024/full_cleaned_cropped_data.csv', index=False)
