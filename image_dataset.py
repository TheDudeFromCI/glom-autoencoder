import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class ImageDataset(Dataset):
    def __init__(self, data_folder):
        self.files = []
        for folder in os.listdir(data_folder):
            folder = os.path.join(data_folder, folder)
            for file in os.listdir(folder):
                file = os.path.join(folder, file)
                self.files.append(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        return F.to_tensor(img)
