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

# Data Processing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Data Visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Performance Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Meta learner
from sklearn.linear_model import LogisticRegression

folder_path = '/content/drive/Shareddrives/ANNDL2024/PKG-Brain-TR-GammaKnife-processed/Brain-TR-GammaKnife-processed'
file_names = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Append the file name to the list
        if file.endswith(".nrrd"):
          file_names.append(os.path.join(root, file))

first_nrrd_file = file_names[0]

data, header = nrrd.read(first_nrrd_file)

lesion_data = pd.read_excel('/content/drive/Shareddrives/ANNDL2024/lesion_data.xlsx')
