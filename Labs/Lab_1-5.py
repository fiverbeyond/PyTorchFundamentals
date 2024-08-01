# This is a private implementation of Lab 5 "Torch Vision Datasets"
# from the Coursera course "Deep Neural Networks with Pytorch".
# MOST OF THE CODE HERE IS NOT MY OWN, but it was typed by me (not copied) as I worked through each line in the lab.


import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
torch.manual_seed(0)

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))

dataset = dsets.MNIST(
    root = '/data',
    download = False,
    transform = transforms.ToTensor()
)

print("Type of the first element: ", type(dataset[0]))

# I stopped the Lab here, as I'm currently unable to get the dataset onto my machine for local testing.