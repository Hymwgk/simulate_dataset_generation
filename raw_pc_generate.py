# -*- coding: utf-8 -*-
import os
import sys
import bpy
import math
import numpy as np

from bpy import data as D
from bpy import context as C
import  mathutils 
from math import *
import blensor
import glob
import pickle
import time
import multiprocessing
#from autolab_core import RigidTransform   导入不了


import argparse
#解析命令行参数
'''
parser = argparse.ArgumentParser(description='Sample grasp for meshes')
parser.add_argument('--gripper', type=str, default='baxter')
args = parser.parse_args()
'''


def do_job(scene_index):
    #print("==================do job", scene_index)
    global table_obj_poses_path
    for path in  table_obj_poses_path:
        if scene_index==int(path.split('/')[-2]):
            meshes_with_pose_path = path


    with open(meshes_with_pose_path,'rb') as f:
        table_meshes_with_pose = pickle.load(f)

    table_obj_list = table_meshes_with_pose[0]
    table_obj_poses_array = table_meshes_with_pose[1]
        
    #去掉除了相机和光源之外的mesh物体
    for item in bpy.data.objects:
        if item.type == 'MESH':
            bpy.data.objects.remove(item)





    #在场景中导入该帧场景中所有待抓取的物体
    for ind in range(len(table_obj_list)):
        bpy.ops.import_scene.obj(filepath=table_obj_list[ind],split_mode ="OFF")
    print("Scene {} has {} meshes ".format(meshes_with_pose_path.split('/')[-2],len(bpy.data.objects)-2))
    #print("已导入{}个模型".format(len(bpy.data.objects)-2))

    #批量修改模型的姿态
    for index, item in enumerate(bpy.data.objects):
        if item.type == 'MESH':
            #
            mesh_name = item.name.split('_')[0]

            for mesh_index,path in enumerate(table_obj_list):
                #
                if path.find(mesh_name)!=-1:
                    #获取位置
                    location = table_obj_poses_array[mesh_index,:3]
                    #获取姿态
                    rotation_quaternion = mathutils.Quaternion(table_obj_poses_array[mesh_index,3:])
                    rotation_euler = rotation_quaternion.to_euler('XYZ')

                    item.rotation_euler=rotation_euler
                    #item.rotation_quaternion = rotation_quaternion
                    item.location=(location[0],location[1],location[2])



    #设置相机
    scanner = bpy.data.objects["Camera"]
    #设置为kinect相机
    scanner.scan_type = "kinect"
    #设置相机的位置姿态
    location = [0.0,-0.465,1.691]
    quaternion = [0.978974,0.198347,-0.026844,-0.039333]
    np.save(os.path.join(os.path.split(meshes_with_pose_path)[0],'world_to_scaner.npy'),np.array(location+quaternion))
    
    scanner.location = location
    scaner_quaternion = mathutils.Quaternion(quaternion)
    scaner_euler = scaner_quaternion.to_euler('XYZ')
    scanner.rotation_euler = scaner_euler


    '''
    #使用blender导入不了
    world_to_scaner_rot=RigidTransform.rotation_from_quaternion([0.978974,0.198347,-0.026844,-0.039333])
    world_to_scaner_trans = np.array(scanner.location)
    world_to_scaner = RigidTransform(world_to_scaner_rot,world_to_scaner_trans,from_frame='world',to_frame='scaner')
    world_to_scaner.save(os.path.join(os.path.split(meshes_with_pose_path)[0],'world_to_scaner.tf'))
    '''

    #设置点云是否添加噪声
    scanner.add_noise_scan_mesh


    """clear all scanning datas  """
    for item in bpy.data.objects:
        #print(item.name)
        if item.type == 'MESH' and item.name.startswith('Scan'):
            bpy.data.objects.remove(item)



                
    """clear the scanning in view windows and start newly scan"""
    bpy.ops.blensor.delete_scans()
    print("Scaning scene {} ... ... ... ".format(meshes_with_pose_path.split('/')[-2]))
    bpy.ops.blensor.scan()

    points_list =[]
    for item in  bpy.data.objects:
        if item.type == 'MESH' and item.name.startswith('Scan'):
            for sp in item.data.vertices:
                points_list += sp.co[0:3]
            points_array =np.reshape(np.array(points_list),(-1,3))
            np.save(os.path.join(os.path.split(meshes_with_pose_path)[0],'raw_pc.npy'),points_array)
            print('===saved '+os.path.join(os.path.split(meshes_with_pose_path)[0],'raw_pc.npy'))


    #休眠1s用于debug
    #time.sleep(1)
def do_jobs(scene_index):
   for path in  table_obj_poses_path:
        if path.split('/')[-2]==str(scene_index):
            scene_obj_list_path = path
            print(path)



if __name__ == '__main__':


    #gripper_name = args.gripper
    #这个得在这里改动夹爪名称
    gripper_name='baxter'

    home_dir = os.environ['HOME']

    scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(gripper_name)
    print(scenes_dir)
    #get 
    table_obj_poses_path = glob.glob(scenes_dir+'*/table_meshes_with_pose.pickle')

    print("导入了{}帧场景".format(len(table_obj_poses_path)))
    #print(scenes_obj_list_path)
    time.sleep(2)

    #设置同时访问几个场景
    pool_size=multiprocessing.cpu_count() #
    if pool_size>len(table_obj_poses_path):
        pool_size = len(table_obj_poses_path)
    scene_index = 0
    pool = []
    for i in range(pool_size):  
        pool.append(multiprocessing.Process(target=do_job,args=(scene_index,)))
        scene_index+=1
    [p.start() for p in pool]  #启动多线程

    while scene_index<len(table_obj_poses_path):    #如果有些没处理完
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                p = multiprocessing.Process(target=do_job, args=(scene_index,))
                scene_index+=1
                p.start()
                pool.append(p)
                break
    [p.join() for p in pool]  #启动多线程


    print('All job done!')


