from google.colab import drive
drive.mount('/content/drive')

%pip install pynrrd
%pip install SimpleITK
%pip install mayavi
%pip install keras
%pip install tensorflow
%pip install gspread

import os
import SimpleITK as sitk
import nrrd
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import gspread

folder_path = '/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/Brain-TR-GammaKnife-processed'
file_names = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Append the file name to the list
        if file.endswith(".nrrd"):
          file_names.append(os.path.join(root, file))

first_nrrd_file = file_names[0]

data, header = nrrd.read(first_nrrd_file)
