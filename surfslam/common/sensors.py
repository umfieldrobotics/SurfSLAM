"""
File: src/common/sensors.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import torch
from typing import Union, TypeAlias

from pyturtlmap import ImuMeasurement, DvlMeasurement, BarometerMeasurement


BackendData: TypeAlias = Union[ImuMeasurement, DvlMeasurement, BarometerMeasurement]

class Image:
    """ Image class for holding images.

    A simple wrapper containing an image and a timestamp
    """

    ## Constructor
    # @param image: a torch Tensor of RGB or Binary data
    # @param timestamp: the time at which the image was captured
    def __init__(self, image: torch.Tensor, timestamp: float):
        self.image = image
        self.timestamp = timestamp

        self.shape = self.image.shape
    
    def state_dict(self) -> dict:
        return {"image": self.image.cpu().numpy(), "timestamp": float(self.timestamp), "device": str(self.image.device)}

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "Image":
        return cls(torch.Tensor(state_dict["image"]).to(state_dict['device']), state_dict["timestamp"])
    
    ## @returns a copy of the current image
    def clone(self) -> "Image":
        if isinstance(self.timestamp, torch.Tensor):
            new_ts = self.timestamp.clone()
        else:
            new_ts = self.timestamp
        return Image(self.image.clone(), new_ts)

    ## Moves all items in the image to the specified device, in-place. Also returns the current image.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    def to(self, device: Union[int, str]) -> "Image":
        self.image = self.image.to(device)

        if isinstance(self.timestamp, torch.Tensor):
            self.timestamp = self.timestamp.to(device)
        else:
            self.timestamp = torch.Tensor([self.timestamp]).to(device)
        return self


class StereoImage:
    """ Image class for holding images.

    A simple wrapper containing an image and a timestamp
    """

    ## Constructor
    # @param image: a torch Tensor of RGB or Binary data
    # @param timestamp: the time at which the image was captured
    def __init__(self, left_image: torch.Tensor, right_image: torch.Tensor, timestamp: float):
        self.left_image = left_image
        self.right_image = right_image
        self.timestamp = timestamp

        self.shape = self.left_image.shape

    ## @returns a copy of the current image
    def clone(self) -> "StereoImage":
        if isinstance(self.timestamp, torch.Tensor):
            new_ts = self.timestamp.clone()
        else:
            new_ts = self.timestamp
        return StereoImage(self.left_image.clone(), self.right_image.clone(), new_ts)

    ## Moves all items in the image to the specified device, in-place. Also returns the current image.
    # @param device: Target device, as int (GPU) or string (CPU or GPU)
    def to(self, device: Union[int, str]) -> "StereoImage":
        self.left_image = self.left_image.to(device)
        self.right_image = self.right_image.to(device)

        if isinstance(self.timestamp, torch.Tensor):
            self.timestamp = self.timestamp.to(device)
        else:
            self.timestamp = torch.Tensor([self.timestamp]).to(device)
        return self
    
    def state_dict(self) -> dict:
        return {
            "left_image": self.left_image.cpu().numpy(),
            "right_image": self.right_image.cpu().numpy(),
            "timestamp": self.timestamp.item() if isinstance(self.timestamp, torch.Tensor) else self.timestamp,
            "device": self.left_image.device
        }
    
    @classmethod
    def from_state_dict(cls, state_dict: dict) -> None:
        device = state_dict["device"]
        left_image = torch.from_numpy(state_dict["left_image"]).to(device)
        right_image = torch.from_numpy(state_dict["right_image"]).to(device)
        timestamp = state_dict["timestamp"]
        return cls(left_image, right_image, timestamp).to(device)
