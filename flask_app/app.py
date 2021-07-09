#!/usr/bin/env python
# coding: utf-8

import os

import torch
from PIL import Image
from flask import Flask, render_template, request
from flask import send_from_directory
from omegaconf import OmegaConf
from torchvision import transforms

from utils import reproducibility, select_model

cwd = os.getcwd()
config_file = '../config/trainer_config.yml'

config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']
config.cwd = str(cwd)
reproducibility(config)
checkpoint = '/Users/macbookpro/Workspace/covid19/CovidDetector/checkpoints/model_COVIDNet_small/dataset_COVID' \
             '/date_08_07_2021_14.10.36/_model_best_checkpoint.pth.tar'

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
checkpoint = torch.load(checkpoint, map_location=device)

state_dict = checkpoint['state_dict']

model = select_model(config, 2)
model.load_state_dict(state_dict)
print('Model loaded')
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

normalize = transforms.Normalize(mean=[0.5], std=[0.5])

class_dict = {1: "COVID", 0: "NON-COVID"}

app = Flask(__name__, static_folder='.')

root_dir = os.getcwd()
print(root_dir)


@app.route('/assets/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(root_dir, 'assets/'), filename)


@app.route('/result')
def serve_result():
    return send_from_directory(os.path.join(root_dir), 'result.png')


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print(request.method)
    f = request.files['image']

    file_path = os.path.join(root_dir, 'upload', f.filename)
    print(file_path)
    f.save(file_path)
    image = Image.open(file_path).convert('RGB')

    image = transform(image)

    # normalize_image = normalize(image)
    normalize_image = image

    normalize_image = normalize_image.unsqueeze(0)
    if torch.cuda.is_available():
        normalize_image = normalize_image.cuda()

    output = torch.softmax(model(normalize_image), 1)
    print(output)

    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()

    response = {}
    print('output : ', output)
    print(pred_label_idx)
    print(prediction_score)

    response['class'] = class_dict[pred_label_idx.item()]
    response['score'] = str(prediction_score.item())

    return response


if __name__ == '__main__':
    app.run(debug=True, port=8000)

