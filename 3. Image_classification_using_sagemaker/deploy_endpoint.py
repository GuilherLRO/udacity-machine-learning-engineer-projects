import subprocess, smdebug, json, logging, sys, os
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as transforms
from PIL import Image, io, requests

# Loads the trained ResNet50 model and replaces its last layer with a custom classifier
def model_fn(model_dir):
    # Load the pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    # Freeze all the pre-trained layers to avoid retraining them
    for param in model.parameters():
        param.required_grad = False
    # Modify the last layer to output the correct number of classes
    num_features, num_classes = model.fc.in_features, 133
    model.fc = nn.Sequential(nn.Linear(num_features, 256), nn.ReLU(),
                              nn.Linear(256, 128), nn.ReLU(),
                              nn.Linear(128, num_classes), nn.LogSoftmax(dim=1))
    # Load the saved state dictionary of the trained model
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    # Set the model to evaluation mode
    model.eval()
    return model

# Preprocesses the input data and returns it as a PIL image object
def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Applies some transformations on the input image, feeds it into the ResNet50 model, and returns the predicted class scores as a tensor
def predict_fn(input_object, model):
    # Apply the necessary transformations on the input image
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object = transform(input_object)
    # Feed the input image into the ResNet50 model and get the predicted class scores
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction