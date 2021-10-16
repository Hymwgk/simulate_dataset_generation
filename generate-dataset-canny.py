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
from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler, grasp
from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
import dexnet
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os

import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # for the convenient of run on remote computer


import argparse

#解析命令行参数
parser = argparse.ArgumentParser(description='Old grasp sample method')
parser.add_argument('--gripper', type=str, default='baxter')
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

def do_job(i):      #处理函数  处理index=i的模型
    """
    创建多线程，多线程调用worker函数，处理指定的模型，
    """    
    #根据id号，截取对应的目标模型名称
    object_name = file_list_all[i].split('/')[-1]
    #存放采样出的good抓取的列表，元素形式是二元素元组  (grasp pose, score)
    good_grasp=[]

    #对CAD模型进行Antipod采样，将采样的抓取结果放到good_grasp中
    #worker(
    # i,   处理第i个模型
    # sample_nums_per_round,  
    # max_iter_per_round,
    # mini_grasp_amount_per_score, 
    # max_rounds, good_grasp)
    #worker(i, 100, 5,20, 30,good_grasp)
    worker(i, 100, 5,20, 30,good_grasp)

    #存放结果文件的路径&名称
    good_grasp_file_name =  os.environ['HOME']+"/dataset/simulate_grasp_dataset/{}/grasp_sampled/{}_{}".format(gripper_name, str(object_name), str(len(good_grasp)))
    
    #创建一个pickle文件，将good_grasp保存起来
    with open(good_grasp_file_name + '.pickle', 'wb') as f:
        pickle.dump(good_grasp, f)

    tmp = []
    for grasp in good_grasp:
        grasp_config = grasp[0].configuration
        score = grasp[1]
        tmp.append(np.concatenate([grasp_config, score]))
    np.save(good_grasp_file_name + '.npy', np.array(tmp))
    print("finished job ", object_name)


def worker(i, sample_nums_per_round, max_iter_per_round,
                        mini_grasp_amount_per_score,max_rounds, good_grasp):  #主要是抓取采样器以及打分    100  20
    """
    brief: 对指定模型，利用随机采样算法，进行抓取姿态的检测和打分
    param [in]  索引为i的CAD模型
    param [in]  采样器单次采样最少返回sample_nums_per_round个抓取
    param [in]  每轮采样最多迭代max_iter_per_round次
    param [in]  每个分数区间最少有mini_grasp_amount_per_score个抓取
    param [in]  结果存放在good_grasp中
    """
    #截取目标对象名称
    object_name = file_list_all[i].split('/')[-1]  
    print('a worker of task {} start, index = {}'.format(object_name, i))    

    #读取采样器初始化配置文件
    yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")
    #加载夹爪配置参数，初始化夹爪对象
    gripper = RobotGripper.load(gripper_name, home_dir + "/code/dex-net/data/grippers") 

    #设置抓取采样的方法
    grasp_sample_method = "antipodal"

    if grasp_sample_method == "uniform":
        grasp_sampler = UniformGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gaussian":
        grasp_sampler = GaussianGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "antipodal":
        #读取夹爪对象与采样器配置，初始化指定的采样器
        grasp_sampler = AntipodalGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gpg":
        grasp_sampler = GpgGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "point":
        grasp_sampler = PointGraspSampler(gripper, yaml_config)
    else:
        raise NameError("Can't support this sampler")
    #print("Log: do job", i)
    #设置obj模型文件与sdf文件路径
    if os.path.exists(str(file_list_all[i]) + "/google_512k/nontextured.obj") and os.path.exists(str(file_list_all[i]) + "/google_512k/nontextured.sdf"):
        of = ObjFile(str(file_list_all[i]) + "/google_512k/nontextured.obj")
        sf = SdfFile(str(file_list_all[i]) + "/google_512k/nontextured.sdf")
    else:
        print("can't find any cad_model or sdf file!")
        raise NameError("can't find any cad_model or sdf file!")

    #根据路径读取模型与sdf文件
    mesh = of.read()
    sdf = sf.read() 
    #构建被抓取的CAD模型数据
    cad_model = GraspableObject3D(sdf, mesh)   
    print("Log: opened object", i + 1, object_name)

#########################################
    #生成一个起点是2.0终点是0.75   步长为-0.4  （递减）的等距数列score_list_sub1 (2.0, 0.75, -0.4) 
    score_list_sub1 = np.arange(0.0, 0.6, 0.2)   
    #生成一个起点是0.5终点是0.36   步长为-0.05的等距数列score_list_sub2  (0.5, 0.36, -0.05)
    score_list_sub2 = np.arange(0.6,1, 0.1)
    end = np.array([1.0])

    #将上面两个向量接起来，变成一个长条向量，使用不同的步长，目的是为了在更小摩擦力的时候，有更多的分辨率
    score_list = np.concatenate([score_list_sub1, score_list_sub2,end])
    #####################准备开始采样############################
    #填充一个与摩擦数量相同的数组，每个对应的元素都是0
    good_count_perfect = np.zeros(len(score_list)-1)
    count = 0
    #如果每个摩擦系数下，有效的抓取(满足力闭合或者其他判断标准)小于要求值，就一直循环查找，直到所有摩擦系数条件下至少都存在20个有效抓取
    while np.sum(good_count_perfect < mini_grasp_amount_per_score) != 0:    
        #开始使用antipodes sample获得对映随机抓取，此时并不判断是否满足力闭合，只是先采集满足夹爪尺寸的抓取
        #如果一轮多次随机采样之后，发现无法获得指定数量的随机抓取，就会重复迭代计算3次，之后放弃，并把已经找到的抓取返回来
        grasps = grasp_sampler.generate_grasps_score(cad_model, target_num_grasps=sample_nums_per_round, grasp_gen_mult=10,max_iter=max_iter_per_round,
                                     vis=False, random_approach_angle=True)
        for grasp in grasps:
            for k in range(len(good_count_perfect)):
                #如果第k个区间内部的抓取数量还不够
                if good_count_perfect[k] < mini_grasp_amount_per_score:
                    #判断当前抓取是否属于这个区间
                    if grasp[1]>=score_list[k] and grasp[1]<score_list[k+1]:
                        good_count_perfect[k]+=1
                        good_grasp.append(grasp)
                        break            
        count += 1
        print('Object:{} GoodGrasp:{}'.format(object_name, good_count_perfect))  
        #如果现在对某个物体的检测轮数太多了，还是找不全，就退出，防止一直在找
        if count >max_rounds:  #如果检测轮数大于30轮了
            break

    print('Gripper:{} Object:{}  After {} rounds found {} good grasps.'.
          format(gripper_name, object_name, count,len(good_grasp)))


if __name__ == '__main__':

    #获取夹爪名称
    gripper_name=args.gripper
    home_dir = os.environ['HOME']
    #存放CAD模型的文件夹
    file_dir = home_dir + "/dataset/simulate_grasp_dataset/ycb/google_512k/"   #获取模型的路径
    file_list_all = get_file_name(file_dir)   #返回所有cad模型所处的文件夹的路径列表
    object_numbers = file_list_all.__len__()  #获取cad模型的数量

    job_list = np.arange(object_numbers)   #返回一个长度为object_numbers的元组 0 1 2 3 ... 
    job_list = list(job_list)                                    #转换为列表
    #设置同时对几个模型进行采样
    pool_size = 50
    if pool_size>object_numbers:
        pool_size = object_numbers
    # Initialize pool
    pool = []     #创建列表
    count =0 
    for i in range(pool_size):   #想多线程处理多个模型，但是实际上本代码每次只处理一个
        count += 1
        pool.append(multiprocessing.Process(target=do_job, args=(i,)))  #在pool末尾添加元素
    [p.start() for p in pool]                  #启动多线程
    # refill
    
    while count < object_numbers:    #如果有些没处理完
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                p = multiprocessing.Process(target=do_job, args=(count,))
                count += 1
                p.start()
                pool.append(p)
                break
    print('All job done.')
    