# -*- coding: utf-8 -*-
import  os
import sys
import random
import numpy as np
import mujoco_py
import pickle
import argparse
import multiprocessing
import shutil
import glob

#解析命令行参数
parser = argparse.ArgumentParser(description='Simulate poses for meshes')
parser.add_argument('--gripper', type=str, default='baxter')
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



def do_job(scene_index):
    scene_i_xml_path = scenes_xml_path_list[scene_index]

    try:
        simulation(scene_i_xml_path)
    except:
        print(scene_i_xml_path," failed")
        failed_list.append(scene_index)



def simulation(scene_i_xml_path):
    #scene_i_xml_path = scenes_xml_path_list[scene_index]
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
    print("Simulating {}".format(scene_i_xml_path))
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
    print("警告：这里需要优化，一些物体可能在空中",__file__,sys._getframe().f_lineno)
    mesh_on_table = pos_now[:,2]>0.5  #找到稳定姿态的高度高于0.5m的mesh
    #mesh_on_table = mesh_on_table[:,0]

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


def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))



if __name__ == '__main__':

    gripper_name = args.gripper

    home_dir = os.environ['HOME']

    scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/"+gripper_name+"/scenes"
    #获取所有场景的xml文件的路径列表
    scenes_xml_path_list = get_file_name(scenes_dir)
    print("为夹爪{}找到{}帧场景".format(gripper_name,len(scenes_xml_path_list)))
    #用于debug
    #simulate_file = scene_xml_dir+"test.xml"
    mangaer = multiprocessing.Manager()
    failed_list =mangaer.list()

    #设置同时仿真几个场景
    pool_size= multiprocessing.cpu_count() #
    if pool_size>len(scenes_xml_path_list):
        pool_size = len(scenes_xml_path_list)
    #pool_size = 1
    scnen_index = 0
    pool = []
    for i in range(pool_size):  
        pool.append(multiprocessing.Process(target=do_job,args=(scnen_index,)))
        scnen_index+=1
    [p.start() for p in pool]  #启动多进程
    #refull
    while scnen_index<len(scenes_xml_path_list):    #如果有些没处理完
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                p = multiprocessing.Process(target=do_job, args=(scnen_index,))
                scnen_index+=1
                p.start()
                pool.append(p)
                break
    [p.join() for p in pool]  #等待所有进程结束

    '''处理仿真仿真失败的情况
    有一些xml文件在进行仿真时，会出现仿真失败的情况
    这里将周边成功场景的仿真结果拷贝到失败文件夹中
    '''
    if len(failed_list)!=0:
        print(failed_list)
        failed_list = [x for x in failed_list] #失败场景的索引

        success_list = np.arange(len(scenes_xml_path_list)) #
        success_list = list(set(success_list).difference(set(failed_list)))#成功场景的索引
        choice_list = random. sample(success_list,len(failed_list))#从成功场景中挑出几个
        print(choice_list)
        max_digits = len(str(len(scenes_xml_path_list)))#场景的最大值

        for i in range(len(choice_list)):
            #copy
            failed_dir =  os.path.join(scenes_dir,str(failed_list[i]).zfill(max_digits))+'/'  #目标
            success_dir  = os.path.join(scenes_dir,str(choice_list[i]).zfill(max_digits))+'/'  #源
            #拷贝成功文件到失败文件夹内
            success_files = glob.glob(success_dir + '*')  
            for files in success_files:
                mycopyfile(files,failed_dir)
        
    print('All job done.')












