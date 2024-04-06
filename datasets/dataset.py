import os
import deeplake
import numpy as np
from PIL import Image
from tqdm import tqdm

class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == "CUB":
            self.images, self.masks = load_cub()
        elif dataset == "ECSSD":
            ds = deeplake.load("hub://activeloop/ecssd")
            self.images = ds["images"]
            self.masks = ds["masks"]
        self.size = len(self.images)

    def load_samples(self):
        for imagep, true_maskp in zip(self.images, self.masks):
            if self.dataset == "CUB":
                img = np.asarray(Image.open(imagep))
                seg = np.asarray(Image.open(true_maskp).convert('L'))
                true_mask = np.where(seg >= 200,1,0)
            elif self.dataset == "ECSSD":
                img = np.asarray(imagep)
                seg = np.asarray(true_maskp)
                true_mask = np.where(seg == True, 1, 0)
            yield img, true_mask


def load_cub():
    cp = os.getcwd()
    
    fold = f'{cp}/segmentations'
    file_paths = []
    for root, _, files in os.walk(fold):
        for file in files:
            file_paths.append(os.path.join(root,file))

    fold2 = f'{cp}/CUB_200_2011/images'
    fp2 = []
    for root, _, files in os.walk(fold2):
        for file in files:
            fp2.append(os.path.join(root,file))

    fp2  = sorted(fp2)
    file_paths = sorted(file_paths)

    with open(f'{cp}/CUB_200_2011/train_test_split.txt') as f:
        count = {}
        pretest = set()
        for line in f:
            x = line.split()[1]
            if x in count:
                count[x]+=1
            else:
                count[x] = 1
            if x == "0":
                pretest.add(line.split()[0])

    with open(f'{cp}/CUB_200_2011/images.txt') as u:
        test = []
        for line in u:
            x,y  = line.split()[0],line.split()[1]
            if x in pretest:
                test.append(y)

    masks = sorted([f'{cp}/segmentations/' +x[:len(x)-3] + 'png' for x in test])
    test = sorted([f'{cp}/CUB_200_2011/images/' +x for x in test])

    return test, masks

