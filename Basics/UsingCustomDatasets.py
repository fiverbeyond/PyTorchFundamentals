import CustomDatasets
from torchvision import transforms

print('Hello World')
dataset = CustomDatasets.toy_set()
print("x dimensions: ", dataset.x.ndimension())
print("x size: ", dataset.x.size())
print("y dimensions: ", dataset.y.ndimension())
print("y size: ", dataset.y.size())

singleRecord = dataset[1]
print("Record size: ", len(singleRecord))
print("Record: ", singleRecord)

# Transforms can be applied directly, as in this case
transform = CustomDatasets.add_mult()
trans_x, trans_y = transform(dataset)
print("Transformed x:", trans_x)
print("Transformed y:", trans_y)


# Transforms can be combined into a single transform operation by using
# the transform compose method.
combo_transform = transforms.Compose(CustomDatasets.add_mult(),
                                     CustomDatasets.mult())