from torch.utils.data import Dataset
import torch


class toy_set(Dataset):
    def __init__(self, length=100, transform=None):
        self.x=2*torch.ones(length,2)
        self.y=torch.ones(length, 1)
        self.len=length
        self.transform=transform

    # Overrides use of [] selectors
    def __getitem__(self,index):
        sample=self.x[index] ,self.y[index]
        if self.transform:
            sample= self.transform(sample)
        return sample

    def __len__(self):
        return self.len

# An example of a custom transform
class add_mult(object):

    def __init__ (self, add_x = 1, mult_y = 1):
        self.add_x = add_x
        self.mult_y = mult_y

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = (x[0] + self.add_x, x[1] + self.add_x)
        y = y * self.mult_y
        sample = x, y
        return sample

class mult(object):
    def __init__(selfself, mult=100):
        self.mult=mult

    def __call__(self, sample):
        x=sample[0]
        y=sample[1]
        x= x * self.mult
        y= y * self.mult
        sample = x, y
        return sample