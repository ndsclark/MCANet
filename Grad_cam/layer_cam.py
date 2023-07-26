import cv2
import numpy as np
import torch
from Grad_cam.base_cam import BaseCAM
from Grad_cam.utils.svd_on_activations import get_2d_projection


# https://ieeexplore.ieee.org/document/9462463
class LayerCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(LayerCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        spatial_weighted_activations = np.maximum(grads, 0) * activations 
        
        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam

