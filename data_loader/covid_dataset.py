import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torchxrayvision as xrv
from torchvision.transforms import transforms
import torchvision.datasets

class CovidDataset(Dataset):
    def __init__(self, config, mode, dim=(224, 224)):
        self.config = config
        self.root = self.config.dataset.input_data + '/'
        self.mode = mode
        self.dim = dim

        data_set = xrv.datasets.COVID19_Dataset(views=["PA", "AP"],
                                                imgpath=self.root + "images",
                                                csvpath=self.root + "metadata.csv")

        self.paths = "images/" + data_set.csv["filename"]
        self.paths = self.paths.to_list()
        self.class_dict = {"COVID": 1, "NON-COVID": 0}
        self.ind2class = {v: k for (k, v) in self.class_dict.items()}
        self.classes = list(self.class_dict.keys())
        self.labels = data_set.labels[:, 3]

        split_len = int(0.1 * len(self.paths))

        if mode == 'test':
            split = split_len
            self.labels = self.labels[:split]
            self.paths = self.paths[:split]
            self.do_augmentation = False
        if mode == 'val':
            split = split_len
            self.labels = self.labels[split:2*split]
            self.paths = self.paths[split:2*split]
            self.do_augmentation = False
        elif mode == 'train':
            split = 2 * split_len
            self.labels = self.labels[split:]
            self.paths = self.paths[split:]
            self.do_augmentation = True

        print("{} examples =  {}".format(mode, len(self.paths)))

    def __getitem__(self, index) -> T_co:
        image_tensor = self.load_image(self.root + self.paths[index], self.dim)
        # label_tensor = torch.zeros(len(self.classes),)
        # label_tensor[int(self.labels[index])] = 1
        return image_tensor, int(self.labels[index])

    def __len__(self):
        return len(self.paths)

    def load_image(self, img_path, dim):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        if self.do_augmentation:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        image_tensor = transform(image)

        return image_tensor


if __name__ == '__main__':
    pass
