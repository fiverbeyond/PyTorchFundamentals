from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image


class ImageDataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform = None):
        # Image directory
        self.data_dir = data_dir

        # the transform to be used on the image
        self.transform = transform
        csv_path = os.path.join(self.data_dir, csv_file)
        # Load the CSV file containing image info
        self.data_frame = pd.read_csv(str(csv_path))

        # Number of images in dataset
        self.len = self.data_frame.shape[0]

    # Get length
    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):

        # Image file path
        img_name = os.path.join(self.data_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name)

        # The class label for the image
        label = self.data_frame.iloc[idx ,0]

        # If there is any transform method, apply it onto the immage
        if self.transform:
            image = self.transform(image)

        return image, label