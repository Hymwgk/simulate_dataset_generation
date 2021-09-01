# -*- coding: utf-8 -*-
# 使用python3
import os
import sys
import re
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import GraspableObject3D
import numpy as np
from dexnet.visualization.visualizer3d import DexNetVisualizer3D as Vis
from dexnet.grasping import RobotGripper
from autolab_core import YamlConfig
from mayavi import mlab
from dexnet.grasping import GpgGraspSampler  # temporary way for show 3D gripper using mayavi
import glob


def show_points(point, name='raw_pc',color='lb', scale_factor=.004):
    mlab.figure(figure=name,bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81),size=(1800,1800))
    if color == 'b':
        color_f = (0, 0, 1)
    elif color == 'r':
        color_f = (1, 0, 0)
    elif color == 'g':
        color_f = (0, 1, 0)
    elif color == 'lb':  # light blue
        color_f = (0.22, 1, 1)
    else:
        color_f = (1, 1, 1)
    if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
        point = point.reshape(3, )
        mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
    else:  # vis for multiple points
        mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)


