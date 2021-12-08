# -*- coding: utf-8 -*-
'''对虚拟生成的点云进行后续处理，
- 剔除NAN点
- 剔除桌腿点
- 剔除离群点
- 对点云添加随机噪声
- 对点云进行随机降采样，使得所有点云总数都保持一致
'''
from logging import raiseExceptions
from math import pi
import math
import os
import sys
import argparse
import time
import mayavi
import numpy as np
import pickle
import glob
import random
from tqdm import tqdm
import torch
from tqdm import tqdm
import multiprocessing
from autolab_core import RigidTransform


#解析命令行参数
parser = argparse.ArgumentParser(description='Point Cloud Post Preceeding')
parser.add_argument('--gripper', type=str, default='baxter')   #
parser.add_argument('--process_num', type=int, default=50)  #设置同时处理几个场景
parser.add_argument('--points_num', type=int, default=15000)  #保存多少个场景点
parser.add_argument('--noise', type=float, default=0.002)  #设置最大随机噪声幅度，默认2mm


args = parser.parse_args()
home_dir = os.environ['HOME']
#场景文件夹
scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(args.gripper)
raw_pc_paths = glob.glob(scenes_dir+'*/raw_pc.npy')
raw_pc_paths.sort()

#获取相机在世界坐标系下的位置姿态
world_to_scaner_path_list = glob.glob(scenes_dir+'*/world_to_scaner.npy') 
world_to_scaner_path_list.sort()


#目前单线程处理
for scene_index in tqdm(range(len(raw_pc_paths)),desc='Point Cloud Post Preceeding: '):

    #剔除NAN值
    raw_pc = np.load(raw_pc_paths[scene_index])
    pc = raw_pc[~np.isnan(raw_pc).any(axis=1)]#删除NAN点云

    #剔除桌腿
    world_to_scaner = np.load(world_to_scaner_path_list[scene_index])
    world_to_scaner_quaternion = world_to_scaner[3:7]#四元数
    world_to_scaner_rot = RigidTransform.rotation_from_quaternion(world_to_scaner_quaternion)#转换到旋转矩阵
    world_to_scaner_trans =world_to_scaner[0:3]#平移向量
    world_to_scaner_T =  RigidTransform(world_to_scaner_rot,world_to_scaner_trans).matrix #构造WTC刚体变换对象

    trans = world_to_scaner_T[0:3,3].reshape(3,1)#平移
    rot = world_to_scaner_T[0:3,0:3] #旋转

    W_pc = np.transpose(np.dot(rot,pc.T)+trans)  #先旋转再平移  之后再次转置[-1,3]

    index_ = np.where(W_pc[:,2] >=0.75)   #z坐标高于0.75m的桌面以上点云
    pc = pc[index_]#检索出原始点云桌面以上的点云table
    scene_in = raw_pc_paths[scene_index].split('/')[-2]
    path = os.path.join(scenes_dir,scene_in,'dense_pc.npy')
    np.save(path,pc)

    
    #随机采样，采样出n个点; 保持每一帧点云的数量都是一致的
    pc = pc[np.random.choice(len(pc), size=args.points_num, replace=True)]

    #为三个轴方向添加随机高斯噪声
    noise = np.random.random(size=(len(pc),3))*args.noise #[len(pc),3]
    pc = pc+noise  #


    #将处理后的点云保存下来
    
    path = os.path.join(scenes_dir,scene_in,'pc.npy')
    np.save(path,pc)
    
