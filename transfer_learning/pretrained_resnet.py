# pytorch transfer_learning example

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# By default, models will be downloaded to your $HOME/.torch folder.
# You can modify this behavior using the $TORCH_MODEL_ZOO variable as follow:
# export TORCH_MODEL_ZOO="/local/pretrainedmodels
# load a pre-trained imagenet model
model = models.resnet18(pretrained=True)
# since we are just evaluate, put the model to eval mode
model.eval()

# load class_names for imagenet
# https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt
with open("imagenet1000_clsidx_to_labels.txt") as f:
    class_names = eval(f.read())

# please download this sample file
# for example on linux
# wget https://en.wikipedia.org/wiki/File:African_Bush_Elephant.jpg
img_path = 'elephant.jpg'

with open(img_path, 'rb') as f:
  with Image.open(f) as img:
    img = img.convert('RGB')

#plt.show()

# according to: https://pytorch.org/docs/stable/torchvision/models.html
# we need to transform before feeding into pretrained model
transform = transforms.Compose([
                               transforms.Resize([224,224]),
                               transforms.ToTensor(), 
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                               ])

input_tensor = transform(img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor, requires_grad=False)

# now do the prediction
output_logits = model(input)

#_, preds = torch.max(output_logits, 1)
top_preds = torch.topk(output_logits, k=3, dim=1)

probs = F.softmax(output_logits, dim=1)[0]

#print( output_logits  )
print( top_preds  )

for pred in top_preds[1][0]:
  real_idx = pred.item()
  print("It is: ", class_names[real_idx], " with prob:", probs[real_idx].item())


# output, we can see the African elephan has the highest score
# African elephant, Loxodonta africana
# tusker
# Indian elephant, Elephas maximus
