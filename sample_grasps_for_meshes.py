#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python3 
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 20/05/2018 2:45 PM 
# File Name  : generate-dataset-canny.py
# 运行之前需要对各个cad文件生成sdf文件

import numpy as np
import sys
import pickle
from dexnet.grasping.quality import PointGraspMetrics3D
from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler, grasp, graspable_object
from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
import dexnet
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os
import glob

import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # for the convenient of run on remote computer

import argparse

#解析命令行参数
parser = argparse.ArgumentParser(description='Sample grasp for meshes')
parser.add_argument('--gripper', type=str, default='panda')
args = parser.parse_args()

#sys.path()

#输入文件夹地址，返回一个列表，其中保存的是文件夹中的文件名称
def get_file_name(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        #将下一层子文件夹的地址保存到 file_list 中
        if root.count('/') == file_dir_.count('/') + 1:
            file_list.append(root)
    #排序
    file_list.sort()
    return file_list

def do_job(job_id,grasps_with_score):      #处理函数  处理index=i的模型
    """
    对目标物体进行采样的子进程
    """    
    print("Object {}, worker {} start ".format(object_name,job_id))
    
    #对CAD模型进行Antipod采样，grasps_with_score_形式为 [(grasp pose, score),...]
    grasps_with_score_ = grasp_sampler.generate_grasps_score(
        dex_net_graspable, 
        target_num_grasps=10,  #每轮采样的目标抓取数量
        grasp_gen_mult=10,                                         
        max_iter=5,                 #为了达到目标数量，最多进行的迭代次数
        vis=False,
        random_approach_angle=True)


    print("Worker {} got {} grasps.".format(job_id,len(grasps_with_score_)))

    #添加到外部的采集库中
    grasps_with_score+=grasps_with_score_
    print("Now we have {} grasps in total".format(len(grasps_with_score)))




def get_dex_net_graspable(object_path):
    '''构建符合dex_net标准的可抓取类对象
    '''
    #设置obj模型文件与sdf文件路径
    if os.path.exists(object_path + "/google_512k/nontextured.obj") and os.path.exists(object_path + "/google_512k/nontextured.sdf"):
        of = ObjFile(object_path + "/google_512k/nontextured.obj")
        sf = SdfFile(object_path + "/google_512k/nontextured.sdf")
    else:
        print("can't find any cad_model or sdf file!")
        raise NameError("can't find any cad_model or sdf file!")
    #根据路径读取模型与sdf文件
    mesh = of.read()
    sdf = sf.read() 
    #构建被抓取的CAD模型数据
    dex_net_graspable = GraspableObject3D(sdf, mesh)   
    print("Log: opened object", object_name)

    return dex_net_graspable



def grasp_sort(grasps_with_score):
    '''按照分数从大到小对采样得到的抓取集合进行排序
    '''
    for i in range(1,len(grasps_with_score)):
        for j in range(0,len(grasps_with_score)-i):
            if grasps_with_score[j][1]<grasps_with_score[j+1][1]:
                grasps_with_score[j],grasps_with_score[j+1] = grasps_with_score[j+1],grasps_with_score[j]
    return grasps_with_score

def redundant_check(grasps_with_score,mini_grasp_amount_per_score):
    '''根据分数分布去除多余的抓取
        将分数划分为几个区间，每个区间仅仅保留一定数量的分数
    '''
    checked_grasps=[]
    #对分数进行区间划分
    score_list_sub1 = np.arange(0.0, 0.6, 0.2)   
    score_list_sub2 = np.arange(0.6,1, 0.1)
    end = np.array([1.0])
    score_list = np.concatenate([score_list_sub1, score_list_sub2,end])
    #为每个区间创建一个计数器
    good_count_perfect = np.zeros(len(score_list)-1)

    #如果每个摩擦系数下，有效的抓取(满足力闭合或者其他判断标准)小于要求值，就一直循环查找，直到所有摩擦系数条件下至少都存在20个有效抓取
    for grasp_with_score in grasps_with_score:
        #对当前抓取循环判断区间
        for k in range(len(good_count_perfect)):
            #如果第k个区间内部的抓取数量还不够
            if good_count_perfect[k] < mini_grasp_amount_per_score:
                #判断当前抓取是否属于这个区间
                if grasp_with_score[1]>=score_list[k] and grasp_with_score[1]<score_list[k+1]:
                    good_count_perfect[k]+=1
                    checked_grasps.append(grasp_with_score)
                    break    

    return checked_grasps


if __name__ == '__main__':

    gripper_name=args.gripper
    home_dir = os.environ['HOME']
    
    #存放CAD模型的文件夹
    file_dir = home_dir + "/dataset/simulate_grasp_dataset/ycb/google_512k/"   #获取模型的路径
    file_list_all = get_file_name(file_dir)   #返回所有cad模型所处的文件夹的路径列表

    #存放采样结果的目录
    grasps_file_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/antipodal_grasps/".format(gripper_name)
    if not os.path.exists(grasps_file_dir):
        os.makedirs(grasps_file_dir)


    #尝试获取外部文件列表
    all_objects_original_grasps = []
    objects_name_list =[]
    original_grasp_files = glob.glob(grasps_file_dir+'original_*')
    #如果外部有指定文件,就从外部读取之前生成好的抓取
    if len(original_grasp_files)!=0:
        print("There is {} original grasp files".format(len(original_grasp_files)))
        print(original_grasp_files)

        for file in original_grasp_files:
            objects_name_list.append(file.split('original_')[-1].split('.')[0])
            with open(file, 'rb') as f:
                all_objects_original_grasps.append(pickle.load(f))
    else:
        print("There is no original grasp files!")



    #读取采样器初始化配置文件
    yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")
    #加载夹爪配置参数，初始化夹爪对象
    gripper = RobotGripper.load(gripper_name, home_dir + "/code/dex-net/data/grippers") 
    #读取夹爪对象与采样器配置，初始化指定的采样器
    grasp_sampler= AntipodalGraspSampler(gripper, yaml_config)

    #
    mangaer = multiprocessing.Manager()
    #如果从外部没有读取到数据
    if len(original_grasp_files)==0:
        #对cad模型按顺序一个一个检测抓取
        for obj_index, object_path in enumerate(file_list_all):
                
            grasps_with_score =mangaer.list()

            #截取目标对象名称
            object_name = object_path.split('/')[-1]  
            #加载cad模型
            dex_net_graspable =  get_dex_net_graspable(object_path)

            #获得计算机的核心数
            cores = multiprocessing.cpu_count()
            #在这里修改同时使用多少个进程执行采样，最好不超过计算机的核心数
            processes_num = 10

            pool =[]
            for i in range(processes_num):
                pool.append(multiprocessing.Process(target=do_job, args=(i,grasps_with_score)))
            #启动多线程
            [p.start() for p in pool]                  
            #等待所有进程结束，返回主进程
            [p.join() for p in pool]                  
            #pool.join()

            print("===========共获得{}个grasp=============".format(len(grasps_with_score)))
            #转化成为普通list
            grasps_with_score  = [x for x in grasps_with_score]

            #按照分数从高到低对采样得到的抓取进行排序
            original_grasps = grasp_sort(grasps_with_score)

            #先保存下来这些原始的采样数据，采样一次挺不容易的
            original_grasp_file_name =  grasps_file_dir+"original_{}.pickle".format(object_name)
            with open(original_grasp_file_name, 'wb') as f:
                pickle.dump(original_grasps, f)


            #按照分数区间，剔除冗余抓取
            final_grasps = redundant_check(grasps_with_score,20)
            #==========保存最终结果=========
            #保存grasp的完整配置和分数
            final_grasp_file_name =  grasps_file_dir+object_name
            with open(final_grasp_file_name+ '.pickle', 'wb') as f:
                pickle.dump(final_grasps, f)
            #只保存姿态和分数
            tmp = []
            for grasp in final_grasps:
                grasp_config = grasp[0].configuration
                score = np.array([grasp[1]])
                tmp.append(np.concatenate([grasp_config, score]))
            np.save(final_grasp_file_name + '.npy', np.array(tmp))
            print("finished job ", object_name)

    else:#如果从外部读取到了数据
        for index,grasps_with_score in enumerate(all_objects_original_grasps):
            #按照分数区间，剔除冗余抓取
            final_grasps = redundant_check(grasps_with_score,20)

            #==========保存最终结果=========
            #保存grasp的完整配置和分数
            final_grasp_file_name =  grasps_file_dir+objects_name_list[index]
            with open(final_grasp_file_name+ '.pickle', 'wb') as f:
                pickle.dump(final_grasps, f)
            #只保存姿态和分数
            tmp = []
            for grasp in final_grasps:
                grasp_config = grasp[0].configuration
                score = np.array([grasp[1]])
                tmp.append(np.concatenate([grasp_config, score]))
            np.save(final_grasp_file_name + '.npy', np.array(tmp))
            print("finished job ", objects_name_list[index])
        
    print('All job done.')
    