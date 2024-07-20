import torch

def main():

    # Interestingly, all elements of tensors are themselves tensors.
    a=torch.tensor([7,4,3,2,6])
    print("Example tensor: " + str(a))
    print("This tensor is of size " + str(a.size()) + " and dimension " + str(a.ndimension()))
    print("\nExample tensor element:" + str(a[3]))

    # To get the actual python values themselves, use item()
    print("Example of item tensor element: " + str(a[3].item()))

    # Tensor types are contained within the tensor object
    print("Tensor type: " + str(a.type()))

    # Whereas datatype stores the type of data held within each cell of the tensor.
    # For example, a tensor storing datatype torch.float32 or torch.float
    # is stored in a tensor of type torch.*.FloatTensor
    print ("Datatype: " + str(a.dtype))


    # Note that datatypes can be 'hard cast'. This tensor's cells hold values of type torch.int32
    # even though the values declared are floats.
    # Note that this implicit convesrion generates a future deprecation warning in Python
    hard_cast = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.int32)
    print("\n\t-- Tensor cast example --")
    print("Hard cast tensor type: " + str(hard_cast.type()))
    print("Hard cast Datatype: " + str(hard_cast.dtype))

    # Tensors can also be 'cast' by explicit type definition at creation.
    # This is a float tensor, even though the passed-in values areintegers.
    floater = torch.FloatTensor([1,2,3,4,5])
    print(floater) # Note that the printed out tensor includes decimals.

    # Tensor types can be converted using the type method
    ints =  torch.IntTensor([1,2,3,4,5])
    print("\n\t-- Tensor conversion example 2: --")
    print("Original integer tensor: " + str(ints))
    converted = ints.type(torch.FloatTensor)
    print("...converted to a float tensor: " + str(converted))




if __name__ == "__main__":
    main()