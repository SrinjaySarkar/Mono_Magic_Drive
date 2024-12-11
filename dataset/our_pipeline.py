import os
import logging
import warnings
from typing import Any, Dict, Tuple

import h5py
import numpy as np
from numpy import random
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image
import PIL.ImageDraw as ImageDraw
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import CameraInstance3DBoxes,DepthInstance3DBoxes,LiDARInstance3DBoxes



