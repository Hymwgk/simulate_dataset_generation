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


def do_job(scene_index):
    #print("==================do job", scene_index)
    #对所有的场景进行点云扫描
    for path in  scenes_obj_list_path:
        if path.find(str(scene_index))!=-1:
            scene_obj_list_path = path

    for path in  scenes_poses_list_path:
        if path.find(str(scene_index))!=-1:
            scene_poses_list_path = path
        #
    scene_poses_list =[]
        #读取场景中的物体路径
    with open(scene_obj_list_path,'rb') as f:
        scene_obj_list=pickle.load(f)
    #读取场景物体姿态列表
    scene_poses_list=np.load(scene_poses_list_path)
        
    #去掉除了相机和光源之外的mesh物体
    for item in bpy.data.objects:
        if item.type == 'MESH':
            bpy.data.objects.remove(item)





    #在场景中导入该帧场景中所有待抓取的物体
    for ind in range(len(scene_obj_list)):
        bpy.ops.import_scene.obj(filepath=scene_obj_list[ind],split_mode ="OFF")
    print("Scene {} has {} meshes ".format(scene_obj_list_path.split('/')[-2],len(bpy.data.objects)-2))
    #print("已导入{}个模型".format(len(bpy.data.objects)-2))

    #批量修改模型的姿态
    for index, item in enumerate(bpy.data.objects):
        if item.type == 'MESH':
            #
            mesh_name = item.name.split('_')[0]

            for mesh_index,path in enumerate(scene_obj_list):
                #
                if path.find(mesh_name)!=-1:
                    #获取位置
                    location = scene_poses_list[mesh_index,:3]
                    #获取姿态
                    rotation_quaternion = mathutils.Quaternion(scene_poses_list[mesh_index,3:])
                    rotation_euler = rotation_quaternion.to_euler('XYZ')

                    item.rotation_euler=rotation_euler
                    #item.rotation_quaternion = rotation_quaternion
                    item.location=(location[0],location[1],location[2])



    #设置相机
    scanner = bpy.data.objects["Camera"]
    #设置为kinect相机
    scanner.scan_type = "kinect"
    #设置相机的位置姿态
    scanner.location = (0,-3,0)
    scanner.rotation_euler= (90/180*pi,0,0/180*pi)
    #设置点云是否添加噪声
    scanner.add_noise_scan_mesh


    """clear all scanning datas  """
    for item in bpy.data.objects:
        #print(item.name)
        if item.type == 'MESH' and item.name.startswith('Scan'):
            bpy.data.objects.remove(item)



                
    """clear the scanning in view windows and start newly scan"""
    bpy.ops.blensor.delete_scans()
    print("Scaning scene {} ... ... ... ".format(scene_obj_list_path.split('/')[-2]))
    bpy.ops.blensor.scan()

    points_list =[]
    for item in  bpy.data.objects:
        if item.type == 'MESH' and item.name.startswith('Scan'):
            for sp in item.data.vertices:
                points_list += sp.co[0:3]
            points_array =np.reshape(np.array(points_list),(-1,3))
            np.save(os.path.join(os.path.split(scene_obj_list_path)[0],'raw_pc.npy'),points_array)
            print('===saved '+os.path.split(scene_obj_list_path)[0])


    #休眠1s用于debug
    #time.sleep(1)



if __name__ == '__main__':

    gripper_name=""
    if len(sys.argv) > 1:
        gripper_name = sys.argv[1]
    else:
        #默认panda夹爪
        gripper_name = "panda"

    gripper_name = "panda"

    home_dir = os.environ['HOME']

    scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(gripper_name)
    print(scenes_dir)
    #get 
    scenes_obj_list_path = glob.glob(scenes_dir+'*/legal_meshes.pickle')
    scenes_poses_list_path = glob.glob(scenes_dir+'*/legal_poses.npy')

    print("导入了{}帧场景".format(len(scenes_poses_list_path)))
    #print(scenes_obj_list_path)
    time.sleep(5)

    #设置同时访问几个场景
    pool_size=10  #
    if pool_size>len(scenes_obj_list_path):
        pool_size = len(scenes_obj_list_path)
    scene_index = 0
    pool = []
    for i in range(pool_size):  
        pool.append(multiprocessing.Process(target=do_job,args=(scene_index,)))
        scene_index+=1
    [p.start() for p in pool]  #启动多线程

    while scene_index<len(scenes_obj_list_path):    #如果有些没处理完
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


