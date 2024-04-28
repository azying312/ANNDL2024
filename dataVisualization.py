# Lesion Image

print(first_nrrd_file)

slice_x = data[data.shape[0] // 2, :, :]  # Slice along the X axis
slice_y = data[:, data.shape[1] // 2, :]  # Slice along the Y axis
slice_z = data[:, :, data.shape[2] // 2]  # Slice along the Z axis

# Plot the slices using Matplotlib
plt.figure(figsize=(10, 10))

plt.subplot(131)
plt.imshow(slice_x, cmap='gray')
plt.title('Slice along X axis')

plt.subplot(132)
plt.imshow(slice_y, cmap='gray')
plt.title('Slice along Y axis')

plt.subplot(133)
plt.imshow(slice_z, cmap='gray')
plt.title('Slice along Z axis')

plt.show()

# MRI Scan

nrrd_file = file_names[3]
print(nrrd_file)
data2, header2 = nrrd.read(nrrd_file)

slice_x = data2[data2.shape[0] // 2, :, :]  # Slice along the X axis
slice_y = data2[:, data2.shape[1] // 2, :]  # Slice along the Y axis
slice_z = data2[:, :, data2.shape[2] // 2]  # Slice along the Z axis

# Plot the slices using Matplotlib
plt.figure(figsize=(10, 10))

plt.subplot(131)
plt.imshow(slice_x, cmap='gray')
plt.title('Slice along X axis')

plt.subplot(132)
plt.imshow(slice_y, cmap='gray')
plt.title('Slice along Y axis')

plt.subplot(133)
plt.imshow(slice_z, cmap='gray')
plt.title('Slice along Z axis')

plt.show()

# Dose Image

nrrd_file = file_names[4]
print(nrrd_file)
data3, header3 = nrrd.read(nrrd_file)

slice_x = data3[data3.shape[0] // 2, :, :]  # Slice along the X axis
slice_y = data3[:, data3.shape[1] // 2, :]  # Slice along the Y axis
slice_z = data3[:, :, data3.shape[2] // 2]  # Slice along the Z axis

# Plot the slices using Matplotlib
plt.figure(figsize=(10, 10))

plt.subplot(131)
plt.imshow(slice_x, cmap='gray')
plt.title('Slice along X axis')

plt.subplot(132)
plt.imshow(slice_y, cmap='gray')
plt.title('Slice along Y axis')

plt.subplot(133)
plt.imshow(slice_z, cmap='gray')
plt.title('Slice along Z axis')

plt.show()

clinical_data = pd.read_excel('/content/drive/Shareddrives/ANNDL2024/Brain-TR-GammaKnife-Clinical-Information.xlsx')
clinical_data = clinical_data.iloc[:,:6]

# Gender vs Age at Diagnosis Plot

figsize = (12, 1.2 * len(clinical_data['Gender'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(clinical_data, x='Age at Diagnosis', y='Gender', inner='stick', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)

# Plot post augmentation data distribution
ensemble_gamaknife_data = pd.read_csv('/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/gamma_knife_data_rotated_cropped')
ensemble_gamaknife_data.groupby('mri_type').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

lesion_data.groupby('mri_type').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)
