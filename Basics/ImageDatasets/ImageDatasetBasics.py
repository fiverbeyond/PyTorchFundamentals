# Practice building and manipulating datasets of images.

from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

directory = "./resources/data"
csv_file ='index.csv'
csv_path = os.path.join(directory, csv_file)

# Build a dataset from the CSV
data_name = pd.read_csv(csv_path)
print("Head of the dataset:",  data_name.head())


# 'iloc' stands for integer location (although iloc has apparently recently been deprecated).
print('File name:', data_name.iloc[1,1])
print('Class or y:', data_name.iloc[1,0])

image_name = data_name.iloc[1,1]
image_path = os.path.join(directory, image_name)

# Image objects can be opened after putting them into their own variable.
image = Image.open(image_path)
plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)
plt.title(data_name.iloc[1,0])
plt.show()
