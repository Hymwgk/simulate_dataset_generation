# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time
import numpy as np
import pickle

#解析命令行参数
parser = argparse.ArgumentParser(description='Get legal grasps with score')
parser.add_argument('--gripper', type=str, default='panda')
args = parser.parse_args()


def get_raw_pc_path_list(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        for file in files:
            if file=='raw_pc.npy':
                file_list.append(os.path.join(root,file))
    #排序
    file_list.sort()
    return file_list

def get_meshes_pose_path_list(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        for file in files:
            if file=='table_meshes_with_pose.pickle':
                file_list.append(os.path.join(root,file))
    #排序
    file_list.sort()
    return file_list

if __name__ == '__main__':
    home_dir = os.environ['HOME']
    scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(args.gripper)
    #获取点云路径列表
    raw_pc_path_list = get_raw_pc_path_list(scenes_dir)
    #获取对应的mesh&pose 列表
    meshes_pose_path_list = get_meshes_pose_path_list(scenes_dir)


    #对每一帧场景处理
    for scene_index,raw_pc_path in enumerate(raw_pc_path_list):
        #读取当前场景raw点云
        pc_raw = np.load(raw_pc_path)
        #剔除NAN值
        pc = pc_raw[~np.isnan(pc_raw).any(axis=1)]

        #打开当前场景的'table_meshes_with_pose.pickle'
        with open(meshes_pose_path_list[scene_index],'rb') as f:
            table_meshes_with_pose = pickle.load(f)

        #获取当前场景的mesh 列表
        table_mesh_list = table_meshes_with_pose[0]
        table_mesh_poses_array = table_meshes_with_pose[1]

        #读取当前mesh物体上采样得到的grasp列表


        #对每一个场景
        for mesh_index,mesh in enumerate(table_mesh_list):
            #获取某个mesh的姿态，需要么？是不是可以直接使用numpy的矩阵运算啊
            table_mesh_poses_array[mesh_index]
            #与采样得到的grasp姿态进行相乘

            #乘完之后，把这些grasp 姿态合并到与当前帧对应的一个大的np.array中


        #获得当前帧旋转后的grasp pose 的np.array之后，进行碰撞检测

        #进行劣质剔除

        #保存为'legal_grasp_with_score.npy'


