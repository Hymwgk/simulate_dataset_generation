#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python3 
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 20/05/2018 2:45 PM 
# File Name  : generate-dataset-canny.py
# 运行之前需要对各个cad文件生成sdf文件

from get_legal_grasps_with_score import get_rot_mat
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
import time
import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # for the convenient of run on remote computer

import argparse

#解析命令行参数
parser = argparse.ArgumentParser(description='Sample grasp for meshes')
parser.add_argument('--gripper', type=str, default='baxter')
#b : breakpoint sample  断点采样
#r:  re-sample  重采样
#p: post-process  后续处理
parser.add_argument('--mode', type=str, choices=['b', 'r', 'p'],default='p')
'''
每个模型的抓取期望数量为 target_n = rounds*process_n*grasp_n
可以根据自己的电脑配置来手动更改这几个参数，参考期望数量为6000以上
'''
#设置采集几轮
parser.add_argument('--rounds', type=int,default=1)
#设置每轮用多少个线程同时采样一个mesh的抓取
parser.add_argument('--process_n', type=int,default=70) #60
#设置每个线程采集的目标抓取数量
parser.add_argument('--grasp_n', type=int,default=200) #200




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
    对目标物体进行采样的子进程，所有的进程将会同时处理同一个目标物体
    """    
    print("Object {}, worker {} start ".format(object_name,job_id))
    
    #对CAD模型进行Antipod采样，grasps_with_score_形式为 [(grasp pose, score),...]
    #其中的分数是regnet中使用的打分方式
    grasps_with_score_ = grasp_sampler.generate_grasps_score(
        dex_net_graspable, #目标抓取对象
        target_num_grasps=args.grasp_n,  #每轮采样的目标抓取数量
        grasp_gen_mult=10,                                         
        max_iter=20,                 #为了达到目标数量，最多进行的迭代次数
        vis=False, #不显示采样结果(多进程该选项无效)
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


def grasp_marking(obj_i):
    """打分筛选函数，可以选择不同的算法对某个模型的抓取进行处理
    """
    object_name = objects_name_list[obj_i]#目标物体名称
    grasps_with_score=all_objects_original_grasps[obj_i]#raw grasps_score
    dex_net_graspable =  dex_net_graspables[object_name]#
    final_grasp_file_name =  grasps_file_dir+object_name
    #选择pointnetgpd的打分方法对物体obj_i进行打分
    point_net_gpd_(grasps_with_score,dex_net_graspable,object_name,grasps_file_dir)
    #或者选择cos alpha的方法

    #print()



def point_net_gpd(grasps_with_score,obj,final_grasp_file_name):
    """对每个物体的抓取进行打分，但是不进行剔除等操作
    """
    object_name = final_grasp_file_name.split('/')[-1]
    print("Start:{}, Grasp num {}".format(object_name,len(grasps_with_score)))
    #设置
    force_closure_quality_config = {}   #设置力闭合  字典
    canny_quality_config = {}
    good_grasp=[]

    #将摩擦力划分为两端不同step间距的array
    fc_list_sub1 = np.arange(2.0, 0.75, -0.4)   
    fc_list_sub2 = np.arange(0.5, 0.36, -0.05)

    #将上面两个向量接起来，变成一个长条向量，使用不同的步长，目的是为了在更小摩擦力的时候，有更多的分辨率
    fc_list = np.concatenate([fc_list_sub1, fc_list_sub2])#axis=0
    #print("判断摩擦系数")
    #print(fc_list)
    #保存字典对
    for value_fc in fc_list:
        #对value_fc保留2位小数，四舍五入
        value_fc = round(value_fc, 2)
        #更改内存中配置中的摩擦系数，而没有修改硬盘中的yaml文件
        yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
        yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc
        #把每个摩擦力值当成键，
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['force_closure'])
        canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['robust_ferrari_canny'])

    #填充一个与摩擦数量相同的数组，每个对应的元素都是0
    good_count_perfect = np.zeros(len(fc_list))
    count = 0
    #设置每个摩擦值需要计算的最少抓取数量 （根据指定输入值20）
    minimum_grasp_per_fc = 30     
    #处理每个原始抓取
    for i,grasp_with_score in enumerate(grasps_with_score):
        grasp = grasp_with_score[0]
        tmp, is_force_closure = False, False
        #循环对某个采样抓取应用不同的抓取摩擦系数，判断是否是力闭合
        for ind_, value_fc in enumerate(fc_list):
            value_fc = round(value_fc, 2) #
            tmp = is_force_closure
            #判断在当前给定的摩擦系数下，抓取是否是力闭合的
            is_force_closure = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                    force_closure_quality_config[value_fc], vis=False)
            #假设当前,1号摩擦力为1.6 抓取不是力闭合的，但是上一个0号摩擦系数2.0 条件下抓取是力闭合的
            if tmp and not is_force_closure:
                #当0号2.0摩擦系数条件下采样的good抓取数量还不足指定的最低数量20
                if good_count_perfect[ind_ - 1] < minimum_grasp_per_fc:
                    #以0号摩擦系数作为边界
                    canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                        canny_quality_config[
                                                                            round(fc_list[ind_ - 1], 2)],
                                                                        vis=False)
                    good_grasp.append((grasp, round(fc_list[ind_ - 1], 2), canny_quality))
                    #在0号系数的good抓取下计数加1
                    good_count_perfect[ind_ - 1] += 1
                #当前抓取j的边界摩擦系数找到了，退出摩擦循环，判断下一个抓取
                break
            #如果当前1号摩擦系数1.6条件下，该抓取j本身就是力闭合的，且摩擦系数是列表中的最后一个（所有的摩擦系数都判断完了）
            elif is_force_closure and value_fc == fc_list[-1]:
                if good_count_perfect[ind_] < minimum_grasp_per_fc:
                    #以当前摩擦系数作为边界
                    canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                        canny_quality_config[value_fc], vis=False)
                    good_grasp.append((grasp, value_fc, canny_quality))
                    good_count_perfect[ind_] += 1
                #当前抓取j关于当前摩擦系数1.6判断完毕，而且满足所有的摩擦系数，就换到下一个摩擦系数
                break

    alpha = 1.0
    beta = 0.01
    tmp = []
    for grasp in good_grasp:
        grasp_config = grasp[0].configuration
        score_friction = grasp[1]
        score_canny = grasp[2]
        #使用pointnetgpd计算的分数
        final_score = alpha/score_friction + beta*score_canny
        tmp.append(np.concatenate([grasp_config, np.array([final_score])]))
    np.save(final_grasp_file_name + '_pgpd.npy', np.array(tmp))

    print('Object:{} GoodGrasp:{} Total:{}'.format(object_name, good_count_perfect,len(good_grasp)))  #判断

   
def point_net_gpd_(grasps_with_score,obj,object_name,grasps_file_dir):
    """对每个物体的抓取使用PointNetGPD的方式进行打分
    1.打分，生成score
    2.覆盖率检查，去掉过分相似的抓取
    3.削减每个score分段内的抓取数量
    """
    print("Start:{}, original_grasp num {}".format(object_name,len(grasps_with_score)))
    #
    force_closure_quality_config = {}   #设置力闭合  字典
    canny_quality_config = {}
    all_grasp=[]

    #将摩擦力划分为两端不同step间距的array
    fc_list_sub1 = np.arange(2.0, 0.75, -0.4)   
    fc_list_sub2 = np.arange(0.5, 0.36, -0.05)

    #将上面两个向量接起来，变成一个长条向量，使用不同的步长，目的是为了在更小摩擦力的时候，有更多的分辨率
    fc_list = np.concatenate([fc_list_sub1, fc_list_sub2])#axis=0
    #print("判断摩擦系数")
    #print(fc_list)
    #保存字典对
    for value_fc in fc_list:
        #对value_fc保留2位小数，四舍五入
        value_fc = round(value_fc, 2)
        #更改内存中配置中的摩擦系数，而没有修改硬盘中的yaml文件
        yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
        yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc
        #把每个摩擦力值当成键，
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['force_closure'])
        canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['robust_ferrari_canny'])

    good_count_perfect = np.zeros(len(fc_list))#填充一个与摩擦数量相同的数组，每个对应的元素都是0
    count = 0
    minimum_grasp_per_fc = 30#设置每个摩擦值需要计算的最少抓取数量 （根据指定输入值20）     

    
    '''执行力闭合&Canny打分 将中间打分结果保存下来
    为每个抓取分配两部分分数：力闭合分数，canny分数    
    
    '''
    temp_file_name =grasps_file_dir+object_name+ '_pgpd-temp.pickle'#中间打分文件路径
    if os.path.exists(temp_file_name):#读取之前生成好的文件
        with open(temp_file_name, 'rb') as f:
            all_grasp = pickle.load(f)
            print('Load ', temp_file_name)
    else:
        for i,grasp_with_score in enumerate(grasps_with_score):
            print('模型{}打分:{}/{}'.format(object_name,i,len(grasps_with_score)))
            grasp = grasp_with_score[0]
            tmp, is_force_closure = False, False
            #循环对某个采样抓取应用不同的抓取摩擦系数，判断是否是力闭合
            for ind_, value_fc in enumerate(fc_list):
                value_fc = round(value_fc, 2) #
                tmp = is_force_closure
                #判断在当前给定的摩擦系数下，抓取是否是力闭合的
                is_force_closure = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                        force_closure_quality_config[value_fc], vis=False)
                #假设当前,1号摩擦力为1.6 抓取不是力闭合的，但是上一个0号摩擦系数2.0 条件下抓取是力闭合的
                if tmp and not is_force_closure:
                    #当0号2.0摩擦系数条件下采样的good抓取数量还不足指定的最低数量20
                    #以0号摩擦系数作为边界
                    canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                        canny_quality_config[
                                                                            round(fc_list[ind_ - 1], 2)],
                                                                        vis=False)
                    all_grasp.append((grasp, round(fc_list[ind_ - 1], 2), canny_quality))
                    #当前抓取j的边界摩擦系数找到了，退出摩擦循环，判断下一个抓取
                    break
                #如果当前1号摩擦系数1.6条件下，该抓取j本身就是力闭合的，且摩擦系数是列表中的最后一个（所有的摩擦系数都判断完了）
                elif is_force_closure and value_fc == fc_list[-1]:
                    #以当前摩擦系数作为边界
                    canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                        canny_quality_config[value_fc], vis=False)
                    all_grasp.append((grasp, value_fc, canny_quality))
                    #当前抓取j关于当前摩擦系数1.6判断完毕，而且满足所有的摩擦系数，就换到下一个摩擦系数
                    break
        #把打好分的抓取保存下来
        with open(temp_file_name, 'wb') as f:
            pickle.dump(all_grasp, f)


    #力闭合与Canny打分完毕，生成单一的score，并保存
    alpha = 1.0
    beta = 0.01
    tmp = np.empty(shape=(0,12))
    for grasp in all_grasp:
        grasp_config = grasp[0].configuration.reshape(1,-1)#(1,10) 原始的抓取配置向量
        score_friction = np.array([grasp[1]])
        score_canny = np.array([grasp[2]])
        #使用pointnetgpd计算的分数
        final_score = alpha/score_friction + beta*score_canny
        tmp=np.concatenate((tmp,np.c_[grasp_config,np.c_[final_score,score_friction]]),axis=0)#(-1,12)
        #tmp.append(np.c_[grasp_config,np.c_[final_score,score_friction]])

    '''
    #排序完了，进行距离覆盖率检查#不在这里进行覆盖率检查
    tmp = coverage_check(tmp)
    final_grasps = np.empty(shape=(0,tmp.shape[1]-1))
    #根据摩擦力划分
    for i,grasp_with_score in enumerate(tmp):
        tmp, is_force_closure = False, False
        for ind_, value_fc in enumerate(fc_list):
            if grasp_with_score[-1]<value_fc :#
                
                if value_fc== fc_list[-1]:#已经最小的了
                    if good_count_perfect[ind_] < minimum_grasp_per_fc:#分段内的抓取还不够
                        good_count_perfect[ind_] += 1
                        final_grasps = np.concatenate((final_grasps,grasp_with_score[0:-1].reshape(1,-1)),axis=0)
                        break
            else:
                if good_count_perfect[ind_] < minimum_grasp_per_fc:
                    good_count_perfect[ind_] += 1
                    final_grasps = np.concatenate((final_grasps,grasp_with_score[0:-1].reshape(1,-1)),axis=0)
                    break
    
    grasp_file_name =grasps_file_dir+object_name+ '_pgpd{}.npy'.format(len(final_grasps))
    np.save(grasp_file_name, final_grasps)
    '''
    #筛选在某最低分数阈值以上的抓取，并保存
    tmp= tmp[tmp[:,-2]>2]
    final_grasps = tmp
    grasp_file_name =grasps_file_dir+object_name+ '_pgpd{}.npy'.format(len(final_grasps))
    np.save(grasp_file_name, final_grasps)


    print('Object:{} FinalGrasp:{} Total:{}'.format(object_name, good_count_perfect,len(final_grasps)))  #判断

def coverage_check(grasps_score):
    """进行覆盖率检查，剔除掉一些抓取中心太近的抓取
    先排序
    """
    #先排序
    #grasps_score = grasps_score[grasps_score[:,-2].argsort()] #按照队后一列，对grasps_score的每一行进行排序
    grasps_score = grasps_score[np.argsort(-grasps_score[:,10],),:]
    #保留抓取
    grasp_stay =np.empty(shape=(0,grasps_score.shape[1]))
    #保留trans
    trans_stay = np.empty(shape=(0,3))
    #
    trans = grasps_score[:,0:3]  #(-1,3)
    #rot = get_rot_mat(grasps_score).reshape(-1,9) #(-1,3,3)

    #每两个抓取之间的抓取中心点间距不小于5mm
    trans_threshold = 0.005

    #覆盖率筛选
    for i in range(len(grasps_score)):
        if trans_stay.shape[0]==0: #
            trans_stay = np.copy(trans[[i]])
        else:
            trans_delta = np.absolute(np.sum((trans_stay-trans[[i]]),axis=1)) #(-1,)
            #距离差异要大
            if np.sum(trans_delta<trans_threshold)==0: 
                trans_stay = np.concatenate((trans_stay,trans[[i]]),axis = 0)
                grasp_stay=np.concatenate((grasp_stay,grasps_score[[i]]),axis = 0)
    
    return grasp_stay


if __name__ == '__main__':

    gripper_name=args.gripper
    home_dir = os.environ['HOME']
    
    #读取stl模型路径列表
    file_dir = home_dir + "/dataset/simulate_grasp_dataset/ycb/google_512k/"   #获取模型的路径
    file_list_all = get_file_name(file_dir)   #返回所有cad模型所处的文件夹的路径列表

    #设置grasp sample result 保存路径
    grasps_file_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/antipodal_grasps/".format(gripper_name)
    if not os.path.exists(grasps_file_dir):
        os.makedirs(grasps_file_dir)

    #设置夹爪尺寸参数，选择使用Antipodal grasp sampler
    yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")#读取采样器初始化配置文件
    gripper = RobotGripper.load(gripper_name, home_dir + "/code/dex-net/data/grippers") #加载夹爪配置参数，初始化夹爪对象
    grasp_sampler= AntipodalGraspSampler(gripper, yaml_config)#读取夹爪对象与采样器配置，初始化指定的采样器

    '''利用CAD模型构造符合dex-net库的待抓取对象
    并保存到硬盘，目的是加快二次计算，减少计算量
    '''
    dex_net_graspables ={}#
    for object_path in file_list_all:
        object_name = object_path.split('/')[-1] #获取模型名称
        pickel_name = os.path.join(object_path,"google_512k","dex_net_graspable.pickle")#待抓取对象的路径
        if os.path.exists(pickel_name):#读取之前生成好的文件
            with open(pickel_name, 'rb') as f:
                dex_net_graspables[object_name] = pickle.load(f)
        else:#如果是第一次执行就生成文件并保存到硬盘
            dex_net_graspables[object_name]=get_dex_net_graspable(object_path)
            with open(pickel_name, 'wb') as f:
                pickle.dump(dex_net_graspables[object_name], f)


    
    mangaer = multiprocessing.Manager()
    #b模式（断点采样模式）和r模式（重新采样模式）
    if args.mode!='p':
        #对每个待抓取模型采样候选抓取
        for obj_index, object_path in enumerate(file_list_all):
            grasps_with_score =mangaer.list()#多进程共享列表
            object_name = object_path.split('/')[-1] #截取目标对象名称
            print("{}开始采样".format(object_name))
            time.sleep(1)
            #
            original_grasp_file_name =  grasps_file_dir+"original_{}.pickle".format(object_name)#预备生成的文件名称 

            #断点生成模式时，会跳过已经生成好的结果
            if os.path.exists(original_grasp_file_name) and args.mode=='b':
                print("{}已存在".format(original_grasp_file_name))
                #time.sleep(1)
                continue
            
            #对某个模型   进行多进程抓取采样
            dex_net_graspable =  dex_net_graspables[object_name]#加载待抓取模型
            cores = multiprocessing.cpu_count()#获得计算机的核心数
            #在这里修改同时使用多少个进程执行采样，最好不超过计算机的核心数
            processes_num = args.process_n
            if processes_num>=cores:#防止设置的进程数超过cpu核心数
                print('\'process_n\'  too large!  Set  \'process_n\' less than {}'.format(cores))
                raise NameError()
            
            for _ in range(args.rounds):#进行args.rounds轮抓取 默认为1轮
                pool =[]
                for i in range(processes_num):
                    pool.append(multiprocessing.Process(target=do_job, args=(i,grasps_with_score)))
                #启动多线程
                [p.start() for p in pool]                  
                #等待所有进程结束，返回主进程
                [p.join() for p in pool]                  
                #pool.join()
            print("==========={}共获得{}个grasp=============".format(object_name,len(grasps_with_score)))
            #转化成为普通list
            grasps_with_score  = [x for x in grasps_with_score]

            #按照分数从高到低对采样得到的抓取进行排序
            original_grasps = grasp_sort(grasps_with_score)

            #保存下来这些原始的采样数据，采样一次挺不容易的
            with open(original_grasp_file_name, 'wb') as f:
                pickle.dump(original_grasps, f)
        print("All job done")

    else:#process mode，读取预先生成好的原始抓取文件做分数筛选等处理
        print('打分处理模式 -p')
        #尝试获取外部文件列表
        objects_name_list =[]  # 
        original_grasp_files = glob.glob(grasps_file_dir+'original_*')  #raw  grasp files path
        all_objects_original_grasps = [] #raw grasps
        #读取b/r模式采样出的抓取文件
        if len(original_grasp_files)!=0:
            print("There is {} original grasp files".format(len(original_grasp_files)))
            for file in original_grasp_files:
                objects_name_list.append(file.split('original_')[-1].split('.')[0])
                with open(file, 'rb') as f:
                    all_objects_original_grasps.append(pickle.load(f))
        else:
            print("There is no original grasp files!")
            sys.exit(1)#表示异常退出程序

        #接着多进程对原始抓取文件进行处理
        pool_size= multiprocessing.cpu_count() #
        if pool_size>len(objects_name_list):#限制进程数小于目标物体数量
            pool_size = len(objects_name_list)
        #pool_size = 1#调试
        obj_index = 0
        pool = []
        for i in range(pool_size):  
            pool.append(multiprocessing.Process(target=grasp_marking,args=(obj_index,)))
            obj_index+=1
        [p.start() for p in pool]  #启动多进程
        #refull
        while obj_index<len(objects_name_list):    #如果有些没处理完
            for ind, p in enumerate(pool):
                if not p.is_alive():
                    pool.pop(ind)
                    p = multiprocessing.Process(target=grasp_marking, args=(obj_index,))
                    obj_index+=1
                    p.start()
                    pool.append(p)
                    break
        [p.join() for p in pool]  #等待所有进程结束
        print('All job done.')
    