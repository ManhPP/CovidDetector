import os

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import transforms

from utils import reproducibility, select_model

cwd = os.getcwd()
config_file = '../config/trainer_config.yml'

config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']
config.cwd = str(cwd)
reproducibility(config)

checkpoint = "_model_last_checkpoint.pth.tar"

model = select_model(config, 2)
checkpoint = torch.load(checkpoint)

state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

image = transform(Image.open('img.jpg').convert('RGB'))
image = image.unsqueeze(0)
output = model(image)

print(output)


