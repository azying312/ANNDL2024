### Retrieve Data

MR_files = [] # list of all MR .nrrd files
DOSE_files = [] # list of dose .nrrd files
LES_files = [] # list of remaining files (non MR or dose)

MR = "_MR_"
DOSE = '_dose'

for file in file_names:
  # print(file)
  if MR in file:
    # print("MRI")
    MR_files.append(file)
  elif DOSE in file:
    # print("dose")
    DOSE_files.append(file)
  else:
    # print("les")
    LES_files.append(file)

### Load Clinical Data

clinical_dir = '/content/drive/Shareddrives/ANNDL2024/Brain-TR-GammaKnife-Clinical-Information.xlsx'
clinical_data = pd.read_excel(clinical_dir)

# Cut NaN cols
clinical_data = clinical_data.iloc[:,:6]

clinical_data = clinical_data.sort_values(by=['unique_pt_id', 'Course #'])

### Clean data

data_folders = os.listdir(folder_path)

# remove .DS
rmv_file = '.DS_Store'
index_to_remove = data_folders.index(rmv_file)
data_folders = data_folders[:index_to_remove] + data_folders[index_to_remove+1:]

data_folders = sorted(data_folders)

les_list = []

# Add cols for DOSE, MRI, Lesion Mask
MR_files = sorted(MR_files)
DOSE_files = sorted(DOSE_files)

clinical_data['mri_file'] = MR_files
clinical_data['dose_file'] = DOSE_files


for folder in data_folders:
  # print('folder', folder)
  full_folder_path = os.path.join(folder_path, folder)
  full_folder = os.path.join(folder_path, folder)
  files = os.listdir(full_folder)

  full_files = [os.path.join(full_folder_path, file_name) for file_name in files]

  overlapping_indices = [index for index, file_name in enumerate(full_files) if file_name in MR_files or file_name in DOSE_files]
  filtered_files = [file_name for index, file_name in enumerate(full_files) if index not in overlapping_indices]

  les_list.append(filtered_files)

clinical_data['les_file'] = les_list

# Expand dataframe to have one lesion file per row
clinical_data = clinical_data.explode('les_file')
clinical_data = clinical_data.sort_values(by=['unique_pt_id', 'Course #', 'les_file'])

# Get Lesion Name in NRRD files
les_file_name = [str.split(os.path.split(file)[1], '.nrrd')[0] for file in clinical_data['les_file']]

clinical_data['Lesion Name in NRRD files'] = les_file_name

### Load lesion data
lesion_data = pd.read_excel('/content/drive/Shareddrives/ANNDL2024/lesion_data.xlsx')

### Match response variable
lesion_data = lesion_data.sort_values(by=['unique_pt_id', 'Treatment Course', 'Lesion Name in NRRD files'])

# Clean clinical_data: GK.270_3_L_RtOccipital11302018
clinical_data['Lesion Name in NRRD files']
to_replace = 'GK.270_3_L_RtOccipital11302018'
lesion_prefix = to_replace[:-8]  # Remove the last 8 characters

# Replace the lesion name in the DataFrame
clinical_data['Lesion Name in NRRD files'] = clinical_data['Lesion Name in NRRD files'].str.replace(to_replace, lesion_prefix)

# Not case sensitive matching
lesion_data['Lesion Name in NRRD files_lower'] = lesion_data['Lesion Name in NRRD files'].str.lower()
clinical_data['Lesion Name in NRRD files_lower'] = clinical_data['Lesion Name in NRRD files'].str.lower()

merged_data = pd.merge(clinical_data, lesion_data[['Lesion Name in NRRD files_lower', 'mri_type', 'duration_tx_to_imag (months)']],
                       how='left', on='Lesion Name in NRRD files_lower')

### Missing data

# Find rows in merged_data where 'mri_type' or 'duration_tx_to_imag (months)' is NaN
missing_rows = merged_data[merged_data['mri_type'].isna() | merged_data['duration_tx_to_imag (months)'].isna()]

# Extract 'Lesion Name in NRRD files' of missing rows
missing_lesion_names = missing_rows['Lesion Name in NRRD files']
print('missing_lesion_names: ')
print(missing_lesion_names)

# Find corresponding rows in clinical_data
missing_clinical_data = clinical_data[clinical_data['Lesion Name in NRRD files'].isin(missing_lesion_names)]
print("missing_clinical_data")
print(missing_clinical_data['Lesion Name in NRRD files'])

# Find corresponding rows in lesion_data
missing_lesion_data = lesion_data[lesion_data['Lesion Name in NRRD files'].isin(missing_lesion_names)]
print("missing_lesion_data")
print(missing_lesion_data['Lesion Name in NRRD files'])

# Note: they aren't in the clinical data file, drop rows
cleaned_data = merged_data.dropna(subset=['mri_type'])

# Write Cleaned Table
cleaned_data.to_csv('/content/drive/Shareddrives/ANNDL2024/full_cleaned_data.csv', index=False)
