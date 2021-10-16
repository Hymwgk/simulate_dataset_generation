# -*- coding: utf-8 -*-
#读取采样后每个mesh上的抓取数量，筛选出满足某些要求的mesh，并把这些合法mesh的路径保存在
# 某夹爪内部的legal_meshes_for_panda.txt中
# 远程测试
import  os
import sys
import pickle
import argparse
import numpy as np
import random

#解析命令行参数
parser = argparse.ArgumentParser(description='Check good meshes')
parser.add_argument('--gripper', type=str, default='baxter')
args = parser.parse_args()


#返回all_16k_stls文件夹下除了背景模型的其他所有stl路径
def get_stls_path(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        for file in files:
            #剔除掉背景模型以及非stl文件
            if file.find('bg_')==-1 and file.find('.stl')!=-1:
                file_list.append(os.path.join(root,file))
    #排序
    #file_list.sort()
    return file_list

#返回
def get_grasp_files_path_name(file_dir_,method):  
    file_list = []
    names = []
    for root, dirs, files in os.walk(file_dir_):
        for file in files:
            #
            if file.find(method)!=-1 and file.find('.npy')!=-1:
                file_list.append(os.path.join(root,file))
                names.append(file.split('_'+method)[0])
    #排序
    #file_list.sort()
    return file_list,names



if __name__ == '__main__':

    gripper_name = args.gripper
    #可以手动指定一些需要删除的mesh
    manual_delet_list = ['059_chain']

    print("Checking good meshes for {}.".format(gripper_name))
    
    home_dir = os.environ['HOME']
    meshes_16k_dir = home_dir+"/dataset/simulate_grasp_dataset/ycb/all_16k_meshes"
    #grasp 目录
    grasp_files_dir = home_dir+"/dataset/simulate_grasp_dataset/"+gripper_name+"/antipodal_grasps"
    #grasp files， name
    grasp_files_path,obj_names = get_grasp_files_path_name(grasp_files_dir,'pgpd')

    #获取{stl name:stl path}
    stls_path_list = get_stls_path(meshes_16k_dir)
    stl_name_path ={}
    for stl_path in stls_path_list:
        name = stl_path.split('/')[-1].split('.')[0]
        stl_name_path[name] = stl_path


    #获取{stl path: grasps}
    all_obj_grasps ={}
    #设定最小的分数阈值
    minimum_score = 2
    #首先是仅仅只将具有高分抓取的obj_name:grasps  保存起来
    for index,path in enumerate(grasp_files_path):
        grasp_with_score = np.load(path)   #(-1,10+1)
        #是希望物体有高分，不仅仅是抓取多
        if np.sum(grasp_with_score[:,-1]>minimum_score)>5:
            obj_name = obj_names[index]
            try:
                all_obj_grasps[stl_name_path[obj_name]]=grasp_with_score  #obj_name:grasps 
            except:
                pass
            else:
                pass
    







    #剔除指定名称
    for index,del_name in enumerate(manual_delet_list):
        for stl_path,_ in all_obj_grasps.items(): 
            if stl_path.find(del_name)!=-1:
                break
        del all_obj_grasps[stl_path]


    #取出list
    final_paths=[]
    for stl_path,_ in all_obj_grasps.items(): 
        final_paths.append(stl_path)


    #随即从合法模型库中抽取一半放在里面
    #final_paths = final_paths+random.sample(final_paths,int(len(final_paths)/2))

    target = home_dir+"/dataset/simulate_grasp_dataset/"+gripper_name+"/good_meshes.pickle"
    #直接用pickel保存
    with open(target,'wb') as f:
        pickle.dump(final_paths, f)


