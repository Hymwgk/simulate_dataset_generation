# -*- coding: utf-8 -*-
#读取采样后每个mesh上的抓取数量，筛选出满足某些要求的mesh，并把这些合法mesh的路径保存在
# 某夹爪内部的legal_meshes_for_panda.txt中
import  os
import sys
import pickle

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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        gripper_name = sys.argv[1]
    else:
        #默认panda夹爪
        gripper_name = "panda"

    print("Checking good meshes for {}.".format(gripper_name))
    
    home_dir = os.environ['HOME']
    meshes_16k_dir = home_dir+"/dataset/simulate_grasp_dataset/ycb/all_16k_meshes"
    target = home_dir+"/dataset/simulate_grasp_dataset/"+gripper_name+"/good_meshes.pickle"

    stls_path_list = get_stls_path(meshes_16k_dir)

    #直接用pickel保存
    with open(target,'wb') as f:
        pickle.dump(stls_path_list, f)

