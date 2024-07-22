# This is a private implementation of Lab 1 "Tensors 1D"
# from the Coursera course "Deep Neural Networks with Pytorch".
# MOST OF THE CODE HERE IS NOT MY OWN, but it was typed by me (not copied) as I worked through each line in the lab.

import torch

import matplotlib.pyplot as plt

def plotVec(vectors):
    ax = plt.axes()
    for vec in vectors:
        ax.arrow(0,0,*vec["vector"], head_width = 0.05, color = vec["color"], head_length = 0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2,2)
    plt.xlim(-2,)

def main():
    ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
    print("The dtype of tensor object after converting it to tensor: ", ints_to_tensor.dtype)
    print("The type of tensor object after converting it to tensor: ", ints_to_tensor.type())
    print("The Python type of the tensor object ", type(ints_to_tensor))

    # Convert float list with length 5 to a tensor
    floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    print("The dtype of tensor object after converting it to tensor: ", floats_to_tensor.dtype)
    print("The type of tensor object after converting it to tensor: ", floats_to_tensor.type())

    list_floats = [0.0, 1.0, 2.0, 4.0, 4.0]
    floats_int_tensor = torch.tensor(list_floats, dtype=torch.int64)
    print("The dtype of tensor object after converting it to tensor: ", floats_int_tensor.dtype)
    print("The type of tensor object after converting it to tensor: ", floats_int_tensor.type())

if __name__ == "__main__":
    main()