# -*- coding: utf-8 -*-
import os
import sys
import argparse
import time
import numpy as np
import pickle
import glob
from autolab_core import RigidTransform
from autolab_core import YamlConfig
from dexnet.grasping import GpgGraspSampler  
from mayavi import mlab

from dexnet.grasping import RobotGripper


#解析命令行参数
parser = argparse.ArgumentParser(description='Get legal grasps with score')
parser.add_argument('--gripper', type=str, default='panda')
args = parser.parse_args()


def get_files_path(file_dir_,filename = None):  
    file_list = []
    if filename !=None:
        for root, dirs, files in os.walk(file_dir_):
            for file in files:
                if file==filename:
                    file_list.append(os.path.join(root,file))
        #排序
        file_list.sort()
        return file_list
    else:
        return False

def get_rot_mat(poses_vector):
    center_point = poses_vector[:,0:3]    #夹爪中心(指尖中心) 
    major_pc = poses_vector[:,3:6]  # binormal
    angle = poses_vector[:,[7]]#

    # cal approach
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    zeros= np.zeros(cos_t.shape)
    ones = np.ones(cos_t.shape)

    #绕世界y轴的旋转矩阵
    R1 = np.c_[cos_t, zeros, sin_t,zeros, ones, zeros,-sin_t, zeros, cos_t].reshape(-1,3,3).swapaxes(1,2)
    #print(R1)
    axis_y = major_pc

    #设定一个与y轴垂直且与世界坐标系x-o-y平面平行的单位向量作为初始x轴
    axis_x = np.c_[axis_y[:,[1]], -axis_y[:,[0]], zeros]
    #查找模为0的行，替换为[1,0,0]
    axis_x[np.linalg.norm(axis_x,axis=1)==0]=np.array([1,0,0])
    #单位化
    axis_x = axis_x / np.linalg.norm(axis_x,axis=1,keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y,axis=1,keepdims=True)
    #右手定则，从x->y  
    axis_z = np.cross(axis_x, axis_y)

    #这个R2就是一个临时的夹爪坐标系，但是它的姿态还不代表真正的夹爪姿态
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]].reshape(-1,3,3).swapaxes(1,2)
    #将现有的坐标系利用angle进行旋转，就得到了真正的夹爪坐标系，
    # 抽出x轴作为approach轴(原生dex-net夹爪坐标系)
    #由于是相对于运动坐标系的旋转，因此需要右乘
    R3=np.matmul(R2,R1)
    '''
    approach_normal =R3[:, :,0]
    #print(np.linalg.norm(approach_normal,axis=1,keepdims=True))
    approach_normal = approach_normal / np.linalg.norm(approach_normal,axis=1,keepdims=True)
    #minor_pc=R3[:, :,2]  是一样的
    minor_pc = np.cross( approach_normal,major_pc)
    '''
    #然后把平移向量放在每个旋转矩阵的最右侧，当成一列
    return R3


def display_grasps(center, pose, color):
    #
    center_point = center
    approach_normal = pose[:,0]
    major_pc = pose[:,1]
        
    #计算夹爪bottom_center
    grasp_bottom_center = -ags.gripper.hand_depth * approach_normal + center_point
    #
    hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
    #固定夹爪作为参考系时
    #local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    ags.show_grasp_3d(hand_points, color=color)
        #ags.show_grasp_norm_oneside(center_point,approach_normal,axis_y,minor_pc)

    return True

def show_points(point, name='raw_pc',color='lb', scale_factor=.004):
    if color == 'b':
        color_f = (0, 0, 1)
    elif color == 'r':
        color_f = (1, 0, 0)
    elif color == 'g':
        color_f = (0, 1, 0)
    elif color == 'lb':  # light blue
        color_f = (0.22, 1, 1)
    else:
        color_f = (1, 1, 1)
    if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
        point = point.reshape(3, )
        mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
    else:  # vis for multiple points
        mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    return point.shape[0]





if __name__ == '__main__':
    home_dir = os.environ['HOME']

    yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")
    gripper = RobotGripper.load(args.gripper, home_dir + "/code/dex-net/data/grippers")
    ags = GpgGraspSampler(gripper, yaml_config)

    scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(args.gripper)
    #获取采样grasp所在文件夹地址
    candidate_grasp_score_dir =  home_dir+"/dataset/simulate_grasp_dataset/{}/antipodal_grasps/".format(args.gripper)
    #获取点云路径列表
    raw_pc_path_list = get_files_path(scenes_dir,filename='raw_pc.npy')
    #获取对应的mesh&pose 列表
    meshes_pose_path_list = get_files_path(scenes_dir,filename='table_meshes_with_pose.pickle')
    #获取相机在世界坐标系下的位置姿态
    world_to_scaner_path_list = get_files_path(scenes_dir,filename='world_to_scaner.npy')
    #获取采样grasp&score
    candidate_grasp_score_list = glob.glob(candidate_grasp_score_dir+'*.npy')

    #对每一帧场景处理
    for scene_index,raw_pc_path in enumerate(raw_pc_path_list):
        #读取当前场景raw点云
        pc_raw = np.load(raw_pc_path)
        #剔除NAN值
        pc = pc_raw[~np.isnan(pc_raw).any(axis=1)]

        #WTC
        world_to_scaner = np.load(world_to_scaner_path_list[scene_index])
        world_to_scaner_quaternion = world_to_scaner[3:7]#四元数
        world_to_scaner_rot = RigidTransform.rotation_from_quaternion(world_to_scaner_quaternion)#转换到旋转矩阵
        world_to_scaner_trans =world_to_scaner[0:3]#平移向量
        world_to_scaner_T =  RigidTransform(world_to_scaner_rot,world_to_scaner_trans) #构造WTC刚体变换对象
        #CTW
        scaner_to_world_T = world_to_scaner_T.inverse().matrix #得到逆矩阵

        #打开当前场景的'table_meshes_with_pose.pickle'
        with open(meshes_pose_path_list[scene_index],'rb') as f:
            table_meshes_with_pose = pickle.load(f)

        #读取当前帧包含有那些mesh，把列表读取出来
        table_mesh_list = table_meshes_with_pose[0]
        table_mesh_poses_array = table_meshes_with_pose[1]
        #以点云坐标系为参考系的变换后的grasp列表
        grasps_center = np.empty(shape=(0,3))
        grasps_pose    =np.empty(shape=(0,3,3))

        #对场景中每一个模型
        for mesh_index,mesh in enumerate(table_mesh_list):
            #WTM
            world_to_mesh_7d = table_mesh_poses_array[mesh_index]
            world_to_mesh_rot = RigidTransform.rotation_from_quaternion(world_to_mesh_7d[3:])
            world_to_mesh_trans = world_to_mesh_7d[0:3]
            world_to_mesh_T = RigidTransform(world_to_mesh_rot,world_to_mesh_trans).matrix
            
            #从grasp库中查找并读取当前mesh的抓取采样结果
            mesh_name = mesh.split('/')[-1].split('.')[0]
            if mesh_name =='bg_table':
                continue
            for path in candidate_grasp_score_list:
                if path.find(mesh_name)!=-1:
                    grasps_with_score = np.load(path)

            #MTG 列表
            mesh_to_grasps_rot=get_rot_mat(grasps_with_score)
            mesh_to_grasps_trans =  grasps_with_score[:,0:3]   
            mesh_to_grasps_rot_trans = np.concatenate((mesh_to_grasps_rot,mesh_to_grasps_trans.reshape(-1,3,1)),axis=2) 
            temp = np.array([0,0,0,1]).reshape(1,1,4).repeat(mesh_to_grasps_rot.shape[0],axis=0) #补第四行
            mesh_to_grasps_T =np.concatenate((mesh_to_grasps_rot_trans,temp),axis=1) #再拼成标准4X4

            #计算CTG
            scaner_to_grasps_T =np.matmul(np.matmul(scaner_to_world_T,world_to_mesh_T),mesh_to_grasps_T)
            scaner_to_grasps_rot = scaner_to_grasps_T[:,0:3,0:3]
            scaner_to_grasps_trans = scaner_to_grasps_T[:,0:3,3].reshape(-1,3)

            '''
            scaner_to_grasp_T=scaner_to_grasps_T[0]
            scaner_to_grasp_rot = scaner_to_grasp_T[0:3,0:3]
            scaner_to_grasp_trans = scaner_to_grasp_T[0:3,3].reshape(-1,3)

            Gp = np.array([1,2,3]) #随便设置一个点Gp
            #分解为R t 两个变换步骤
            Cp_r_t=scaner_to_grasp_rot.dot(Gp) + scaner_to_grasp_trans#先旋转再平移
            Cp_t_r=scaner_to_grasp_rot.dot(Gp+ scaner_to_grasp_trans)#先平移再旋转
            #直接使用4x4变换矩阵计算
            Gp_q = np.array([1,2,3,1])#将Gp拓展为齐次向量
            Cp_q = scaner_to_grasp_T.dot(Gp_q)

            print(Cp_r_t)
            print(Cp_t_r)
            print(Cp_q[0:3])
            '''
            #将所有的模型的grasp  统一添加到场景的grasp 列表中
            grasps_center = np.r_[grasps_center,scaner_to_grasps_trans]
            grasps_pose = np.r_[grasps_pose,scaner_to_grasps_rot]


        #对旋转后的抓取，逐个进行碰撞检测，并把没有碰撞的抓取保存下来

        #创建debug显示，仅限于单线程
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
        _ = show_points(pc)
        for index,center in enumerate(grasps_center):
            display_grasps(center,grasps_pose[index],color="d")
        mlab.show()





        #获得当前帧旋转后的grasp pose 的np.array之后，进行碰撞检测

        #进行劣质剔除

        #保存为'legal_grasp_with_score.npy'


