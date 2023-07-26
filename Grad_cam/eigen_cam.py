import cv2
import numpy as np
import torch
from Grad_cam.base_cam import BaseCAM
from Grad_cam.utils.svd_on_activations import get_2d_projection


# https://arxiv.org/abs/2008.00299
class EigenCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
