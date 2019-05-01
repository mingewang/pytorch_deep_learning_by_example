# feature extractor from any layer example

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# extract feature from a layer in this model
# model: pretrained model
# layer_name: layer_name inside that pre-trained model
# output_shape from that layer
# img_path: file path for that image
def extract_feature(model, layer_name, output_shape, img_path):
  with open(img_path, 'rb') as f:
    with Image.open(f) as img:
      img = img.convert('RGB')
      # 1. Load the image with Pillow library
    img = Image.open(img_path)

  # according to: https://pytorch.org/docs/stable/torchvision/models.html
  # we need to transform before feeding into pretrained model
  transform = transforms.Compose([
                                 transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                                 ])

  input_tensor = transform(img)         # 3x400x225 -> 3x244x244 size may differ
  input_tensor = input_tensor.unsqueeze(0) # 3x244x244 -> 1x3x244x244
  # Create a PyTorch Variable with the transformed image
  t_input = torch.autograd.Variable(input_tensor, requires_grad=False)

  # 3. Create a vector of zeros that will hold our feature vector
  #    The 'avgpool' layer has an output size of 512
  my_embedding = torch.zeros(output_shape)
  # 4. Define a function that will copy the output of a layer
  def copy_data(m, i, o):
      print (o.data.shape)
      my_embedding.copy_(o.data)

  # Use the model object to select the desired layer
  layer = model._modules.get(layer_name)
  # 5. Attach that function to our selected layer
  h = layer.register_forward_hook(copy_data)
  # 6. Run the model on our transformed image
  model(t_input)
  # 7. Detach our copy function from the layer
  h.remove()
  # 8. Return the feature vector
  return my_embedding

# By default, models will be downloaded to your $HOME/.torch folder.
# You can modify this behavior using the $TORCH_MODEL_ZOO variable as follow:
# export TORCH_MODEL_ZOO="/local/pretrainedmodels
# load a pre-trained imagenet model
model = models.resnet18(pretrained=True)

# we need to watch the output of model carefully
# then we can choose/decide which feature/layer we want
# in this case, we use: avgpool
# the avgpool is just one layer output from print(model)
# that we want to extract a feature from.
# that is an arbitrary layer for the purpose of this demo.
# feel free to try any other layer
# Need to pay special attention to shape of this layer
print( model )

# Set model to evaluation mode
model.eval()

# please download this sample file
# for example on linux
# wget https://en.wikipedia.org/wiki/File:African_Bush_Elephant.jpg
img_path = 'elephant.jpg'

avg_layer = "avgpool"
# observered from its previous layer called layer4[1].bn2
# BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
output_shape = [1, 512, 1, 1]

# Now extract feature
img_feature = extract_feature(model, avg_layer, output_shape, img_path)

print( img_feature )



