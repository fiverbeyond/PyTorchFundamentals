import torch

def main():

    a=torch.tensor([7,4,3,2,6])
    print("Example tensor" + str(a[3]))
    print("Tensor type: " + str(a.type()))
    print ("Datatype: " + str(a.dtype))


if __name__ == "__main__":
    main()