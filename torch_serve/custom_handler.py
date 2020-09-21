"""
Custom handler for pytorch serve. 
reference documents:
https://pytorch.org/serve/custom_service.html
https://github.com/pytorch/serve/blob/master/docs/custom_service.md
https://github.com/FrancescoSaverioZuppichini/torchserve-tryout
"""
import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

# Custom handler with class level entry point
# The handler can extend from any of following:
# BaseHander, object, image_classifier, image_segementer, 
# object_detector, text_classifier
class MyHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms. CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    
    # usually do not need to overload this  
    """
    def initialize(self, context):
        #Initialize model. This will be called during model loading time
        #:param context: Initial context contains model server system properties.
        #:return:
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details
        ...
        """

    def preprocess_one_req(self, req):
        """
        Process one single image.
        """
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")       
         # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        Process all the images from the requests and batch them in a Tensor.
        """
        requests = data
        images = [self.preprocess_one_req(req) for req in requests]
        images = torch.cat(images)    
        return images

    def inference(self, model_input):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        outs = self.model.forward(model_input)
        probs = F.softmax(outs, dim=1) 
        preds = torch.argmax(probs, dim=1)
        return preds

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        res = []
        # inference_output  [BATCH_SIZE, 1]
        # convert it to list
        preds = inference_output.cpu().tolist()
        # index_to_name.json will be loaded
        # and automatically accessible as self.mapping
        # print(self.mapping)
        for pred in preds:
            label = self.mapping[str(pred)]
            res.append({'label' : label, 'index': pred })
        return res

    # usually do not need to overload this  
    """
    def handle(self, data, context):
        #Invoke by TorchServe for prediction request.
        #Do pre-processing of data, prediction using model and postprocessing of prediciton output
        #:param data: Input data for prediction
        #:param context: Initial context contains model server system properties.
        #:return: prediction output
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
    """

_service = MyHandler()

# module level entry point
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
