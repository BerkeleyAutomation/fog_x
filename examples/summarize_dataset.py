import fog_x 
from fog_x.loader import RLDSLoader

path = "/home/kych/datasets/rtx"
dataset_name = "fractal20220817_data"
version = "0.1.0"
split = "train"

loader = RLDSLoader(path=f"{path}/{dataset_name}/{version}", split=split, shuffling=False)

data = loader[0][0]
for k, v in data.items():
    print(k)
    if k == "observation" or k == "action":
        for k2, v2 in v.items():
            print(k, k2, v2.shape, v2.dtype)
    else:
        print(k, v.shape, v.dtype)

