from super_gradients.training import models

import torch
import os

def load_model(model: str, pretrained_coco: bool = False):
    """
    Load a model from the models module.
    - model: The model name to load or the local path to the model
    - pretrained_coco: Whether to load the model with COCO pretrained weights.
    """

    # Check if model is a path
    if os.path.exists(model):
        loaded_model = torch.load(model)
        loaded_model.eval()
        return loaded_model

    return models.get("yolo_nas_l", pretrained_weights="coco" if pretrained_coco else None)
