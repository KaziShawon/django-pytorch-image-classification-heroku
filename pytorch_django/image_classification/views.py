from django.shortcuts import render

# Create your views here.

import io
import os
import json
import base64
from glob import glob

import torch
from torch._C import _import_ir_module_from_package
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from django.conf import settings
from django.shortcuts import render
from .forms import ImageUploadForm

# Load model architecture
# Select GPU if available, if not select CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
json_path = os.path.join(settings.STATIC_ROOT, "classes.json")
classes_mapping = json.load(open(json_path))


def transforms(image_bytes):
    im = Image.open(io.BytesIO(image_bytes))
    im = im.resize((224, 224), Image.ANTIALIAS)
    im_tensor = torch.from_numpy(np.array(im))
    im_tensor = im_tensor.permute(2, 0, 1)
    im_tensor = im_tensor.unsqueeze(0)
    im_tensor = im_tensor.to(device, dtype=torch.float32)
    return im_tensor


num_classes = 10
# Use Inception V3
model = models.inception_v3(pretrained=False)
path2weights = "/home/kz/Projects_Learning/medina-tech-portfolio/django-pytorch-image-classification-heroku/pytorch_django/vgg16_pretrained.pt"
classes = ["cane", "cavallo", "elefante", "farfalla", "gallina",
           "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
             "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
             "ragno": "spider", "scoiattolo": "squirrel"}


def load_model(path2weights):
    model_vgg = models.vgg16(pretrained=False)
    for param in model_vgg.parameters():
        param.requires_grad = False
    model_vgg.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model_vgg.classifier = nn.Sequential(nn.Flatten(),
                                         nn.Linear(512, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(256, 10))
    # Use the file path of the trained model
    if torch.cuda.is_available():
        model_vgg.load_state_dict(torch.load(path2weights))
    else:
        model_vgg.load_state_dict(torch.load(path2weights, map_location='cpu'))
    return model_vgg.to(device)


def get_prediction(image_bytes, path2weights):
    model = load_model(path2weights)
    image_tensor = transforms(image_bytes)
    model_output = model(image_tensor.to(device))
    output_scaled = torch.softmax(model_output, dim=1)
    _, preds_tensor = torch.max(output_scaled, 1)
    predicted_idx = (np.squeeze(preds_tensor.numpy()) if not torch.cuda.is_available(
    ) else np.squeeze(preds_tensor.cpu().numpy()))
    # predicted_class = classes_mapping[predicted_idx]
    predicted_class = translate[classes[predicted_idx]]
    return predicted_class


def index(request):
    image_uri = None
    predicted_label = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Passing the image to avoid storing it to DB or filesystem
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64, %s' % ('image/jpeg', encoded_img)

            # get predicted label
            try:
                predicted_label = get_prediction(image_bytes, path2weights)
            except RuntimeError as re:
                print(re)
    else:
        form = ImageUploadForm()
    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }
    return render(request, 'image_classification/index.html', context)
