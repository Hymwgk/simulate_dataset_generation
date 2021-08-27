# -*- coding: utf-8 -*-
import  os
import sys
import random
import numpy as np
import mujoco_py
import pickle
import argparse

#解析命令行参数
parser = argparse.ArgumentParser(description='Simulate poses for meshes')
parser.add_argument('--gripper', type=str, default='panda')
args = parser.parse_args()

def get_file_name(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        for file in files:
            #查找场景xml文件
            if file.find('scene_') != -1 and file.find('.xml') != -1:
                #保存下来该文件的路径
                file_list.append(os.path.join(root,file))
    #排序
    file_list.sort()
    return file_list

def get_mesh_list(xml_string):
    mesh_list=[]
    #按照回车分割
    xml_string_list = xml_string.split('\n')
    for string in xml_string_list:
        if string.find('<mesh')!=-1:
            mesh_name = string.split('\"')[-2].split('.')[0]
            #去除背景mesh
            if mesh_name.find('bg_')==-1:
                #名称带有后缀，比如   mesh1.stl  
                mesh_list.append(mesh_name)
    return mesh_list

def get_meshes_dir(xml_string):
    #按照回车分割
    xml_string_list = xml_string.split('\n')
    for string in xml_string_list:
        if string.find('meshdir=')!=-1:
            meshes_dir = string.split('\"')[-2]
    return meshes_dir

def get_table_pose(xml_string):
    strings=[]
    #按照回车分割
    strings = xml_string.split('\n')
    for string in strings:
        if string.find('<body name="table"')!=-1:
            pos_string = string.split('\"')[3].split()
            quat_string = string.split('\"')[5].split()
            pos = [float(x) for x in pos_string]
            quat = [float(x) for x in quat_string]

            pos = np.array(pos)
            quat =  np.array(quat)
            pose = np.concatenate((pos,quat))
            break

    return pose




if __name__ == '__main__':

    gripper_name = args.gripper

    home_dir = os.environ['HOME']

    scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/"+gripper_name+"/scenes"
    #获取所有场景的xml文件的路径列表
    scenes_xml_path_list = get_file_name(scenes_dir)
    print("为夹爪{}找到{}帧场景".format(gripper_name,len(scenes_xml_path_list)))
    #用于debug
    #simulate_file = scene_xml_dir+"test.xml"


    for scene_i_xml_path in scenes_xml_path_list:
        #读取场景i的 xml内容
        with open(scene_i_xml_path,'r') as f:
            scene_i_xml_string = f.read()
        #读取场景i的xml文件中的字符串，提取xml中的mesh名称列表
        scene_i_meshes_list = get_mesh_list(scene_i_xml_string)
        #获取所含meshes文件夹地址
        scene_i_meshes_dir = get_meshes_dir(scene_i_xml_string)
        #存放scene_i_xml文件的文件夹地址
        scene_i_xml_root_path,_= os.path.split(scene_i_xml_path)

        #获得背景中的桌子的姿态，一会而添加到姿态列表中
        table_pose = get_table_pose(scene_i_xml_string)


        #根据xml创建仿真器
        model = mujoco_py.load_model_from_path(scene_i_xml_path)
        print("Simulating {}".format(scene_i_xml_path.split('/')[-1]))
        #对该模型创建模拟器对象, 迭代次数为1000，step可以简单认为是时间间隔
        #选择一个足够长的step，保证每个物体有足够时间落在指定的桌面上
        #有时候例如圆柱体会在桌面滚动
        sim = mujoco_py.MjSim(model,nsubsteps=5000)
        #sim.data.set_joint_qvel("joint1", np.array([0, 0, -0.5, 0, 0, 0]))
        #sim.forward()

        pos_old = sim.data.qpos
        #pos_old = sim.data.qpos.reshape(-1,7)[:,2]

        steady = False
        #等待场景稳定
        while not steady:
            #打印初始状态下各个物体的状态，格式是p_x,p_y,p_z,q_i,q_j,q_k,w  
            #print(sim.data.qpos)

            #使用step函数开始仿真
            sim.step()
            #获取每个模型当前时刻的z坐标值
            pos_now = sim.data.qpos
            #pos_now = sim.data.qpos.reshape(-1,7)[:,2]
            #计算差值
            pos_delta = pos_now - pos_old
            #如果z坐标值变动小于0.005m，就认为环境已经稳定，退出仿真
            if not np.sum(pos_delta>0.001):
                steady = True
            else:
                pos_old = pos_now

        pos_now=pos_now.reshape(-1,7)
        mesh_on_table = pos_now[:,2]>0.5  #找到稳定姿态的高度低于0.5m的mesh
        #抽取出满足某个条件的行（保留合法的姿态）
        pos_legal = pos_now[mesh_on_table,:]
        table_pose=np.reshape(table_pose,(-1,7))
        #拼接起来啊
        pos_legal = np.concatenate((table_pose,pos_legal),axis = 0)



        #保留合法的mesh名称（带有路径）
        mesh_legal = []
        #首先把桌子的路径先加上去
        mesh_legal.append(os.path.join(scene_i_meshes_dir,'bg_table.obj'))
        #检索桌面上的物体
        for index,item  in enumerate(mesh_on_table):
            if item:
                mesh_legal.append(os.path.join(scene_i_meshes_dir,scene_i_meshes_list[index]+'.obj'))
        

        #使用npy保存位置合法姿态列表
        #np.save(os.path.join(scene_i_xml_root_path,"legal_poses.npy"),pos_legal)

        #直接用pickel保存
        table_meshes_with_pose=(mesh_legal,pos_legal)
        with open(os.path.join(scene_i_xml_root_path,"table_meshes_with_pose.pickle"),'wb') as f:
            pickle.dump(table_meshes_with_pose, f)



        #保存稳定后的场景,仅仅用于debug
        #f= open(simulate_file,'w+')
        #sim.save(f,format='xml')


