import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # To use pandas series, treat the values attribute as a numpy array.
    pandas_series = pd.Series([1.0,2.0,3.0,4.0,5.0])
    pandas_to_torch = torch.from_numpy(pandas_series.values)

    # Tensors can be indexed and sliced just like Python arrays
    sub_set = pandas_to_torch[1:4] # Ending index is exclusive
    print("Subset tensor: " + str(sub_set))

    # Hadamard product (entrywise) and dot product are straightforward
    torch_tensor = torch.DoubleTensor([0.0, 1.0, 2.0, 3.0, 4.0])
    product = pandas_to_torch * torch_tensor # Recall that in hadamard product, the tensors must be the same dimensions.
    print("Hadamard product tensor dimensions:" + str(product.ndimension()))
    print("Dot product result: " + str(pandas_to_torch.dot(torch_tensor)));

    # The ability to add a scalar value to a vector value is called 'Broadcasting'
    print(str(sub_set + 3))

    # Functions are applied element-wise
    base = torch.tensor([0, np.pi/2, np.pi])
    sined = torch.sin(base)
    print("Transformed tensor: " + str(sined))

    # The linspace function is a useful way to generate evenly-distributed stepwise samples.
    increments = torch.linspace(5, 100, 20)
    print("Linenspace: " + str(increments))
    x = torch.linspace(0, 2 * np.pi, 100)
    y = torch.sin(x)
    plt.plot(x.numpy(), y.numpy())
    plt.show()

if __name__ == "__main__":
    main()