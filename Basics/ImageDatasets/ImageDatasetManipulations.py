import torch
import matplotlib.pylab as plot
torch.manual_seed(0) # forces the same seed on all random operations

from PIL import Image
import pandas as pd
import os

def show_data(data_sample, shape = (28, 28)):
    plot.imshow(data_sample[0].numpy().reshape(shape), cmap = 'gray')
    plot.title('y = ', + data_sample[1])

# Read the CSV file f9rom the URL and print out the first five samples
directory = "./resources/data"
csv_file = 'index.csv'
csv_path = os.path.join(directory, csv_file)

# Load the file into a dataframe
data_frame = pd.read_csv(csv_path)
print("Head: ", data_frame.head())

print('File name: ', data_frame.iloc[0, 0])
print('Class type: ', data_frame.iloc[0, 1])

print('Rows in the whole dataset:', data_frame.shape[0])


# Load the image as a variable.
image_name = data_frame.iloc[1,1]
image_path = os.path.join(directory, image_name)
print('Working with \'', image_name, '\' in ', image_path)

# Show the training image
image = Image.open(image_path)
plot.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)
plot.title(data_frame.iloc[1,0])
plot.show()






