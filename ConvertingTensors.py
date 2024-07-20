import torch
import numpy as np
import pandas as pd

def main():

    # Tensors may be converted to larger n-dimensional tensors using the view() method
    one_dim = torch.Tensor([1,2,3,4,5])
    # -1 in the first parameter will infer the number of rows.
    two_dim = one_dim.view(5,1) # Specifies five rows and one column.
    print("Converted tensor: " + str(two_dim))
    print("Converted tensor dimensions: " + str(two_dim.ndimension()))

    # Conversions to and from numpy arrays are deliberately easy.
    numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    torch_tensor = torch.from_numpy((numpy_array))
    back_to_numpy = torch_tensor.numpy()

    # Note that (I think) in these examples, torch_tensor and back_to_numpy
    # are both actually references! If I change the original numpy_array,
    # both derived arrays will be simultaneously changed.
    # This feels... very weird, in python. Does Python even know what a reference is?
    numpy_array[3] = 45
    print("torch_tensor: " + str(torch_tensor))
    print("back_to_numpy = " + str(back_to_numpy))

    # To use pandas series, treat the values attribute as a numpy array.
    pandas_series = pd.Series([1.0,2.0,3.0,4.0,5.0])
    pandas_to_torch = torch.from_numpy(pandas_series.values)





if __name__ == "__main__":
    main()