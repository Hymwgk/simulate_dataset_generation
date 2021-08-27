# -*- coding: utf-8 -*-
#从一个指定的文件中，读取所列的mesh的路径；生成mujoco需要的xml场景文件
#将生成的xml文件，分别保存在各自的场景文件夹中
#
import  os
from string import digits
import sys
import random
import numpy as np
import pickle
import argparse

#解析命令行参数
parser = argparse.ArgumentParser(description='Generate Mujoco xml files')
parser.add_argument('--gripper', type=str, default='panda')
parser.add_argument('--mesh_num', type=int, default=10)
parser.add_argument('--scene_num', type=int, default=10)
args = parser.parse_args()


def get_file_name(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        for file in files:
            if file.find('bg_')==-1:
                file_list.append(file.split('.')[0])
    #排序
    #file_list.sort()
    return file_list

if __name__ == '__main__':

    #夹爪名称
    gripper_name = args.gripper
    #场景中模型数量
    mesh_num = args.mesh_num
    #场景数量
    scene_num = args.scene_num


    print("为{}生成{}帧场景，每帧{}个物体".format(gripper_name,scene_num,mesh_num))
    #获取最大的位数，用来补0
    max_digits = len(str(scene_num))

    home_dir = os.environ['HOME']

    #设定legal_meshes的路径记录文件地址
    legal_meshes_pickle = home_dir+'/dataset/simulate_grasp_dataset/'+gripper_name+'/good_meshes.pickle'
    #读取legal_meshes的路径记录文件
    with open(legal_meshes_pickle,'rb') as mesh:
        mesh_list=pickle.load(mesh)

    #获取16k mesh库目录
    meshes_dir ,_=os.path.split(mesh_list[0])
    #提取出mesh文件名称，目的是去写到xml文件中
    mesh_list = [ mesh.split('/')[-1].split('.')[0]    for mesh in mesh_list]


    #场景文件夹主目录
    scenes_dir = home_dir+'/dataset/simulate_grasp_dataset/panda/scenes/'


    xml_template_string = """<?xml version="1.0" ?>
    <mujoco model="MuJoCo Model">
        <!-- 设置角度用弧度制，设定mesh文件夹路径 -->
        <compiler angle="radian" meshdir=/>
        <!-- 每一step间隔0.005s，重力加速度，黏度 -->
        <option timestep="0.005" gravity="0 0 -1.0" viscosity="0.01" impratio="1" density="0" tolerance="1e-10" jacobian="sparse" iterations="30"/>
        <size njmax="500" nconmax="100"/>
        <visual>
            <quality shadowsize="2048"/>
            <map stiffness="700" fogstart="10" fogend="15" zfar="40" shadowscale="0.5"/>
            <rgba haze="0.15 0.25 0.35 1"/>
        </visual>
        <statistic meansize="0.05" extent="2"/>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="texplane" builtin="checker" mark="cross" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="512" height="512"/>
            <material name="matplane" texture="texplane" texuniform="true" reflectance="0.3"/>
            <!-- 桌子的模型 -->
            <mesh name="bg_table" file="bg_table.stl"/>
            <!-- 四块桌子围栏的模型 -->
            <mesh name="bg_table_fance1" file="bg_table_fance.stl"/>
            <mesh name="bg_table_fance2" file="bg_table_fance.stl"/>
            <mesh name="bg_table_fance3" file="bg_table_fance.stl"/>
            <mesh name="bg_table_fance4" file="bg_table_fance.stl"/>
            <!--下面的mesh标签是待仿真物体的模型占位符，将会被替换为具体的值-->
            <mesh/>
        </asset>
        <worldbody>
            <geom name="ground" size="0 0 1" type="plane" condim="1" material="matplane"/>
            <light pos="0 0 5" dir="0 0 -1" directional="true" castshadow="false" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
            <light pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
            <body name="table" pos="0 0 0.755" quat="0.707107 0.707107 0 0">
                <geom type="mesh" mesh="bg_table"/>
            </body>
            <body name="bg_table_fance1" pos="0.0 0 0.75" quat="0.707107 0.707107 0 0">
                <geom type="mesh" mesh="bg_table_fance1"/>
            </body>
            <body name="bg_table_fance2" pos="0 0.0 0.75" quat="0.5  0.5  0.5  0.5">
                <geom type="mesh" mesh="bg_table_fance2"/>
            </body>
            <body name="bg_table_fance3" pos="0 0 0.75" quat="0 0 0.707107 0.707107">
                <geom type="mesh" mesh="bg_table_fance3"/>
            </body>
            <body name="bg_table_fance4" pos="0 0 0.75" quat="-0.5  -0.5  0.5  0.5">
                <geom type="mesh" mesh="bg_table_fance4"/>
            </body>
            <!--body name="bg_funnel_part1" pos="0.0 0 1" quat="0.707107 0.707107 0 0">
                <geom type="mesh" mesh="bg_funnel_part1"/>
            </body>
            <body name="bg_funnel_part2" pos="0 0.0 1" quat="0.5  0.5  0.5  0.5">
                <geom type="mesh" mesh="bg_funnel_part2"/>
            </body>
            <body name="bg_funnel_part3" pos="0 0 1" quat="0 0 0.707107 0.707107">
                <geom type="mesh" mesh="bg_funnel_part3"/>
            </body>
            <body name="bg_funnel_part4" pos="0 0 1" quat="-0.5  -0.5  0.5  0.5">
                <geom type="mesh" mesh="bg_funnel_part4"/>
            </body-->
            <!--下面的标签是待仿真物体的参数设定占位符，决定初始的位姿，将会被替换为具体的值-->
            <body/>
        </worldbody>
    </mujoco>  
    """

    #生成n种场景，每种场景中的物体种类都重新挑选
    for scene_n in range(scene_num):
        #打乱mesh_list顺序
        random.shuffle(mesh_list)
        #截取指定的k个文件,抽选子集
        mesh_list_ = mesh_list[:mesh_num]

        xml_string = xml_template_string

        #===修改meshes文件夹地址
        temp_string="meshdir=\""+meshes_dir+"\""
        #查找mesh地址标签并分割
        temp_list = xml_string.split('meshdir=')
        xml_string = temp_string.join(temp_list)
        #===修改asset标签，导入stl模型
        temp_string=""
        for i in range(mesh_num):
            temp_string +="     <mesh file=\""+mesh_list_[i]+".stl\" />\n"
        #查找mesh标签并分割
        temp_list = xml_string.split("     <mesh/>\n")
        xml_string = temp_string.join(temp_list)

        #===修改body标签
        temp_string=""
        #先采样各个位置，防止初始位置相互触碰，导致物理模拟出现失败
        pos_group = np.zeros([1,3])
        maximum_rounds = 10
        #最大可以检测10轮
        for _ in range(maximum_rounds):
            #对每个物体进行采样
            for i in range(mesh_num):
                #每个物体最大可以采样20次
                for j in range(20):
                    #采样的约束范围
                    pos_x = round((random.random()-0.5)*0.3,2)
                    pos_y =round((random.random()-0.5)*0.3,2)
                    pos_z = round(random.random()*1.5+1.2,2)  
                    pos = np.array([[pos_x,pos_y,pos_z]])
                   #如果是第一个
                    if not pos_group[0][2]:
                        pos_group = pos
                        break
                    else:
                        #计算与已有的每个位置之间的差值
                        pos_delta = np.linalg.norm(pos_group - pos,axis=1)
                        if np.sum(pos_delta<0.2):#限制每两个物体坐标系原点之间的最小距离
                            continue
                        else:#和已有的位置都挺远的
                            #把采样出来的位置添加到group中
                            pos_group = np.concatenate((pos_group,pos),axis=0)
                            break
            #如果采样出的合法位置少于要求的数量
            if pos_group.shape[0]<mesh_num:
                #清除本轮的内容
                pos_group = np.empty([1,3])
                continue
            else:#已经都检测出合适的位置了，就退出
                break
        #如果所有轮数都检测完了，还是没有齐全，就报错
        if pos_group.shape[0]<mesh_num:
            print("初始位置检测失败，修改初始位置约束范围或减少场景模型数量")
    
        #开始根据随机抽选的模型以及初始位置替换xml中具体内容    
        for i in range(mesh_num):
            pos = str(pos_group[i][0])+" "+str(pos_group[i][1])+" "+str(pos_group[i][2])
            #随机分配初始姿态
            rot_x = round(random.random()*3.14,2)
            rot_y = round(random.random()*3.14,2)
            rot_z = round(random.random()*3.14,2)
            rot = str(rot_x)+" "+str(rot_y)+" "+str(rot_z)

            temp_string +="    <body name=\""+mesh_list_[i]+"\" pos=\""+pos+"\" euler=\""+rot+"\">\n      \
                <joint name=\"joint"+str(i)+"\" type=\"free\"/>\n      <geom type=\"mesh\" mesh=\""+mesh_list_[i]+"\"   />\n    </body>\n"
        #查找body标签并分割
        temp_list = xml_string.split("    <body/>\n")
        xml_string = temp_string.join(temp_list)


        #场景的具体名称
        new_xml_name = ""  #清零
        new_xml_name = "scene_"+str(scene_n)+".xml"

        #找到对应的场景文件夹目录，没有的话就创建
        scene_dir = os.path.join(scenes_dir+str(scene_n).zfill(max_digits))
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)

        #创建新的xml文件并写入修改后的内容
        xml = open(os.path.join(scene_dir,new_xml_name),'w+')
        xml.write(xml_string)
        xml.close()










