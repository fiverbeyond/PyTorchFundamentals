# This is a private implementation of Lab 2 "Tensors 2D"
# from the Coursera course "Deep Neural Networks with Pytorch".
# MOST OF THE CODE HERE IS NOT MY OWN, but it was typed by me (not copied) as I worked through each line in the lab.
# Most of these lines have been adapted to let me practice parts of pytorch I find interesting.

import torch
import pandas

# Convert 2D List to 2D tensor
a_list = [[11,12,13],[21,22,23],[31,32,33]]
a_tensor = torch.tensor(a_list)
print("'A' Tensor: \n:", a_tensor)

dataframe = pandas.DataFrame({'a':[11,12,13], 'b':[12,22,31]})

print("Pandas Dataframe to numpy: ", dataframe.values)
print("Type before conversion: ", dataframe.values.dtype)

print("\n===========================================")

new_tensor = torch.from_numpy(dataframe.values)
print("\nTensor after converting: \n", new_tensor)
print("Type after conversion: ", new_tensor.dtype)

print("Slicing practice\n")
print("What is the value on the first row, first two columns?", a_tensor[0, 0:2])
print("What is the value on the first row, first two cloumns? ", a_tensor[0][0:2])

