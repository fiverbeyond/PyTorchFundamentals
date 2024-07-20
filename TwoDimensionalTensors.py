# Some practice with 2D Tensors in Pytorch
# Tester line to make sure github works.

import torch

def main():
    a_list = [[1,1,1],[1,1,1],[1,1,1]]
    a_tensor = torch.tensor(a_list)

    print("Multidimensional tensor:\n" + str(a_tensor))
    print("Number of dimensios: " + str(a_tensor.ndimension()))
    print("Shape of tensor: " + str(a_tensor.shape))
    print("Size of tensor: " +str (a_tensor.size()))
    print("Number of elements: " + str(a_tensor.numel()))

    # Multidimensional tensosr of the same shape can be added. The result is identical to matrix addition.
    b_tensor = torch.tensor([[1,0,0], [0,1,0], [0,0,1]])
    sum_tensor = a_tensor + b_tensor
    print("B Tensor: \n" + str(b_tensor))
    print("\nSummed tensor:\n " + str(sum_tensor) )

    # Matrix multiplication works to. Recall that in matrix multiplication, the number of columns in the first matrix
    # match the number of columns in the second matrix.
    product_tensor = torch.mm(a_tensor, b_tensor)
    print("\nProduct tensor:\n" + str(sum_tensor))


if __name__ == "__main__":
    main()