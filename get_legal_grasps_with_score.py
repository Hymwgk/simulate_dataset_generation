# -*- coding: utf-8 -*-
from math import pi
import os
import sys
import argparse
import time
from matplotlib.pyplot import axis
import mayavi
import numpy as np
import pickle
import glob
import random
from autolab_core import RigidTransform
from autolab_core import YamlConfig
from dexnet.grasping import GpgGraspSampler  
from mayavi import mlab
from tqdm import tqdm
from dexnet.grasping import RobotGripper
from numpy.core.fromnumeric import swapaxes, transpose
import torch
import open3d as o3d

#解析命令行参数
parser = argparse.ArgumentParser(description='Get legal grasps with score')
parser.add_argument('--gripper', type=str, default='baxter')   #
parser.add_argument('--load_npy',action='store_true')  #设置同时处理几个场景

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

home_dir = os.environ['HOME']

yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")
gripper = RobotGripper.load(args.gripper, home_dir + "/code/dex-net/data/grippers")
ags = GpgGraspSampler(gripper, yaml_config)


class GetLegalGrasps:
    def __init__(self,use_dense=False) -> None:
        self.minimum_points_num=10 #限制夹爪内部最少点数
        self.minimum_insert_dist=0.01 #限制夹爪内部点数最小嵌入距离
        self.table_hight = 0.75 #桌面高度，用于检测夹爪桌面碰撞
        self.safe_dis = 0.005  #与桌面的最小安全距离
        self.max_angle=50  #夹爪接近角度限制

        scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(args.gripper)
        #获取采样grasp所在文件夹地址
        candidate_grasp_score_dir =  home_dir+"/dataset/simulate_grasp_dataset/{}/antipodal_grasps/".format(args.gripper)
        #获取点云路径列表
        pc_path_list = glob.glob(scenes_dir+'*/pc.npy')
        pc_path_list.sort()

        table_pc_path_list = glob.glob(scenes_dir+'*/table_pc.npy')
        table_pc_path_list.sort()
        #获取对应的mesh&pose 列表
        meshes_pose_path_list = glob.glob(scenes_dir+'*/table_meshes_with_pose.pickle')
        meshes_pose_path_list.sort()
        #获取相机在世界坐标系下的位置姿态
        world_to_scaner_path_list = glob.glob(scenes_dir+'*/world_to_scaner.npy')
        world_to_scaner_path_list.sort()
        #获取采样grasp&score
        candidate_grasp_score_list = glob.glob(candidate_grasp_score_dir+'*.npy')

        #对每一帧场景处理
        for scene_index,pc_path in enumerate(pc_path_list):
            #读取当前场景raw点云
            self.pc= np.load(pc_path)
            self.dense_pc = np.load(table_pc_path_list[scene_index])

            if use_dense:
                self.used_pc = self.dense_pc
            else:
                self.used_pc = self.pc

            #获取场景抓取信息
            self.grasp_centers,self.bottom_centers,self.grasp_poses,self.grasp_scores,\
                self.local_hand_points,self.local_hand_points_extend,self.hand_points=\
                    self.get_scene_grasps(world_to_scaner_path_list,meshes_pose_path_list,\
                        candidate_grasp_score_list,scene_index)

            print('Start job ', pc_path)

            #self.show_grasps_pc(title='raw')
            self.collision_check_table_cuda()
            #self.show_grasps_pc(title='table_check')
            self.restrict_approach_angle()
            #self.show_grasps_pc(title='angle_check')
            self.collision_check_pc_cuda()
            #self.show_grasps_pc(title='pc_check')
            #保存不与桌子碰撞的抓取
            #限制抓取approach轴与桌面垂直方向的角度
            #验证结果
            #self.collision_check_validate()

            #保存为'legal_grasp_with_score.npy'
            save_path =os.path.join(os.path.split(pc_path)[0],'legal_grasps_with_score.npy')

            #先批量取出虚拟夹爪各点相对于相机坐标系C的坐标

            grasp_dict ={}
            grasp_dict['bottom_centers'] = self.bottom_centers
            grasp_dict['grasp_centers'] = self.grasp_centers
            grasp_dict['grasp_poses'] = self.grasp_poses
            grasp_dict['grasp_score'] = self.grasp_scores
            grasp_dict['local_hand_points'] =self.local_hand_points
            grasp_dict['local_hand_points_extend'] =self.local_hand_points_extend
            grasp_dict['hand_points'] = self.hand_points#所有抓取姿态下各个夹爪的坐标
            grasp_dict['pc'] = self.pc
            grasp_dict['dense_pc'] = self.dense_pc

            np.save(save_path,grasp_dict)
            print('Job done ',save_path,'  good grasps num',len(self.grasp_poses))

    def get_scene_grasps(self,world_to_scaner_path_list,meshes_pose_path_list,grasp_files_path,scene_index):
        '''获取每一帧场景中的抓取信息
        '''
        #WTC
        world_to_scaner = np.load(world_to_scaner_path_list[scene_index])
        world_to_scaner_quaternion = world_to_scaner[3:7]#四元数
        self.world_to_scaner_rot = RigidTransform.rotation_from_quaternion(
                                                        world_to_scaner_quaternion)#转换到旋转矩阵
        self.world_to_scaner_trans =world_to_scaner[0:3]#平移向量


        #构造WTC刚体变换对象
        world_to_scaner_T =  RigidTransform(self.world_to_scaner_rot,self.world_to_scaner_trans) 

        #CTW   世界在相机坐标系中的位置姿态
        self.scaner_to_world_T = world_to_scaner_T.inverse().matrix #得到逆矩阵

        #GTB   碰撞坐标系在抓取典范坐标系下的位置姿态
        translation=np.array([-ags.gripper.hand_depth,0,0])
        grasp_to_bottom_T =  RigidTransform(np.eye(3),translation).matrix


        #打开当前场景的'table_meshes_with_pose.pickle'
        with open(meshes_pose_path_list[scene_index],'rb') as f:
            table_meshes_with_pose = pickle.load(f)

        #读取当前帧包含有那些mesh，把列表读取出来
        self.mesh_list = table_meshes_with_pose[0]
        self.mesh_poses_array = table_meshes_with_pose[1]

        grasp_centers = np.empty(shape=(0,3))
        grasp_poses    =np.empty(shape=(0,3,3))
        grasp_scores = np.empty(shape=(0,1))
        #计算所有夹爪的点
        hand_points = np.empty([0,21,3])

        #本地夹爪各点坐标
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))#[21,3]


        #对场景中每一个模型
        for mesh_index,mesh in enumerate(self.mesh_list):
            #从grasp库中查找并读取当前mesh的抓取采样结果
            mesh_name = mesh.split('/')[-1].split('.')[0]
            if mesh_name =='bg_table': #跳过桌子
                continue
            #if 'banana' not in mesh_name:
            #    continue
            for path in grasp_files_path:
                if mesh_name in path:
                    #读取模型之上的抓取
                    grasps_with_score = np.load(path) #


            #WTM  模型在场景中的位置姿态
            world_to_mesh_7d = self.mesh_poses_array[mesh_index]
            world_to_mesh_rot = RigidTransform.rotation_from_quaternion(world_to_mesh_7d[3:])
            world_to_mesh_trans = world_to_mesh_7d[0:3]
            world_to_mesh_T = RigidTransform(world_to_mesh_rot,world_to_mesh_trans).matrix
            
            #MTG 模型上的抓取相对于模型坐标系的位置姿态
            mesh_to_grasps_rot=self.get_rot_mat(grasps_with_score)
            mesh_to_grasps_trans =  grasps_with_score[:,0:3]   
            mesh_to_grasps_rot_trans = np.concatenate((mesh_to_grasps_rot,mesh_to_grasps_trans.reshape(-1,3,1)),axis=2) 
            temp = np.array([0,0,0,1]).reshape(1,1,4).repeat(mesh_to_grasps_rot.shape[0],axis=0) #补第四行
            mesh_to_grasps_T =np.concatenate((mesh_to_grasps_rot_trans,temp),axis=1) #再拼成标准4X4

            #计算CTG   抓取坐标系在相机坐标系下的位姿
            scaner_to_grasps_T =np.matmul(np.matmul(self.scaner_to_world_T,world_to_mesh_T),mesh_to_grasps_T)
            scaner_to_grasps_rot = scaner_to_grasps_T[:,0:3,0:3]
            scaner_to_grasps_trans = scaner_to_grasps_T[:,0:3,3].reshape(-1,3)
            
            #计算CTB  夹爪碰撞坐标系在相机坐标系下的位姿
            scaner_to_bottoms_T = np.matmul(scaner_to_grasps_T,grasp_to_bottom_T) #[-1,4,4]
            scaner_to_bottoms_rot = scaner_to_bottoms_T[:,0:3,0:3] #[-1,3,3]
            scaner_to_bottoms_trans = scaner_to_bottoms_T[:,0:3,3].reshape(-1,3) #[-1,3]

            #计算p
            Bp =np.swapaxes(np.expand_dims(local_hand_points,0),1,2)#.repeat(scaner_to_bottoms_rot.shape[0],axis=0)
            RCp = np.swapaxes(np.matmul(scaner_to_bottoms_rot,Bp),1,2) #[-1,21,3]

            Cp = RCp + np.expand_dims(scaner_to_bottoms_trans,1) #[-1,21,3]



            #将所有的模型的grasp  统一添加到场景的grasp 列表中
            grasp_centers = np.concatenate((grasp_centers,scaner_to_grasps_trans),axis=0)#()
            grasp_poses=np.concatenate((grasp_poses,scaner_to_grasps_rot),axis=0)#(-1,3,3)
            grasp_scores = np.concatenate((grasp_scores,grasps_with_score[:,[10]]),axis=0)  #(-1,1)
            hand_points = np.concatenate((hand_points,Cp),axis=0)#(-1,21,3)

        local_hand_points_extend = ags.get_hand_points_extend(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        bottom_centers = grasp_centers -ags.gripper.hand_depth * grasp_poses[:,:,0] 


        #计算所有夹爪的点
        '''
        hand_points = np.empty([0,21,3])
        for i in range(grasp_centers.shape[0]):
            points = ags.get_hand_points(bottom_centers[i],
                                                            grasp_poses[i,:,0], 
                                                            grasp_poses[i,:,1]).reshape(1,21,3)#(1,21,3)
            hand_points = np.concatenate((hand_points,points),axis=0)#(-1,21,3)
        '''
        
        return grasp_centers,bottom_centers,grasp_poses,grasp_scores,local_hand_points,local_hand_points_extend,hand_points



    def collision_check_pc_cuda(self):
        """对CTG抓取姿态和Cpc进行碰撞检测(使用显卡加速计算)
        """
        #对旋转后的抓取，逐个进行碰撞检测，并把没有碰撞的抓取保存下来
        #mask =np.zeros(centers.shape[0])
        if self.grasp_centers.shape[0]==0:
            return 0

        before = len(self.grasp_centers)
        poses_cuda=torch.from_numpy(self.grasp_poses).cuda()
        mask_cuda = torch.zeros(self.grasp_centers.shape[0]).cuda()
        local_hand_points= torch.from_numpy(self.local_hand_points).cuda()
        bottom_centers = torch.from_numpy(self.bottom_centers).cuda()
        pc = torch.from_numpy(self.used_pc).cuda()

        gripper_points_p = torch.tensor([local_hand_points[4][0],local_hand_points[2][1],local_hand_points[1][2],
                                                                    local_hand_points[12][0],local_hand_points[9][1],local_hand_points[10][2],
                                                                    local_hand_points[3][0],local_hand_points[13][1],local_hand_points[2][2],
                                                                    local_hand_points[12][0],local_hand_points[15][1],local_hand_points[11][2]]).reshape(4,1,-1).cuda()

        gripper_points_n = torch.tensor([local_hand_points[8][0],local_hand_points[1][1],local_hand_points[4][2],
                                                                    local_hand_points[10][0],local_hand_points[1][1],local_hand_points[9][2],
                                                                    local_hand_points[7][0],local_hand_points[2][1],local_hand_points[3][2],
                                                                    local_hand_points[20][0],local_hand_points[11][1],local_hand_points[12][2]]).reshape(4,1,-1).cuda()

        #对每个抓取进行碰撞检测
        for i in tqdm(range(len(bottom_centers)),desc='collision_check_pc'):

            #得到标准的旋转矩阵
            matrix = poses_cuda[i]
            #转置=求逆（酉矩阵）
            grasp_matrix = matrix.T  # same as cal the inverse
            #获取所有的点相对于夹爪底部中心点的向量
            points = pc - bottom_centers[i].reshape(1, 3)
            points_g = torch.mm(grasp_matrix, points.T).T
            #查找左侧夹爪碰撞检查
            points_p = points_g.repeat(4,1,1)
            points_n = points_g.repeat(4,1,1)

            points_p = points_p-gripper_points_p
            points_n = points_n-gripper_points_n
            check_op =torch.where(torch.sum((torch.mul(points_p,points_n)<0)[0],dim=1)==3)[0]

            #check_c = (torch.mul(points_p,points_n)<0)[1:]
            check_ = torch.where(torch.sum((torch.mul(points_p,points_n)<0)[1:],dim=2)==3)[0]

            points_in_close_area=points_g[check_op] #(-1,3)
            #if points_in_gripper_index.shape[0] == 0:#不存在夹爪点云碰撞
            if len(check_)==0:
                collision = False
                #检查夹爪内部点数是否够
                if len(points_in_close_area):#夹爪内部有点
                    deepist_point_x =  torch.min(points_in_close_area[:,0])
                    insert_dist = ags.gripper.hand_depth-deepist_point_x.cpu()
                    #设置夹爪内部点的最少点数,以及插入夹爪的最小深度
                    if  len(points_in_close_area)<self.minimum_points_num  or insert_dist<self.minimum_insert_dist:
                        mask_cuda[i]=1
                else:#夹爪内部根本没有点
                    mask_cuda[i]=1

            else:
                collision = True
                mask_cuda[i]=1
        
        mask = mask_cuda.cpu()
        self.bad_grasp_centers = self.grasp_centers[mask==1]
        self.bad_grasp_poses = self.grasp_centers[mask==1]
        self.bad_hand_points = self.hand_points[mask==1]
        self.bad_bottom_centers=self.bottom_centers[mask==1]


        self.grasp_centers =self.grasp_centers[mask==0]
        self.grasp_poses = self.grasp_poses[mask==0]
        self.grasp_scores = self.grasp_scores[mask==0]
        self.bottom_centers=self.bottom_centers[mask==0]
        self.hand_points = self.hand_points[mask==0]

        after = len(self.grasp_centers)
        print('Collision_check_pc done:  ',before,' to ',after)


    def collision_check_table_cuda(self):
        """对夹爪进行批量的碰桌子检测，夹爪上面的最低点不允许低于桌子高度
        """
        if self.grasp_centers.shape[0]==0:
            return 0

        before = len(self.grasp_centers)
        #先批量取出虚拟夹爪各点相对于相机坐标系C的坐标
        #批量获取bottom_centers 用于碰撞检测
        bottom_centers = torch.from_numpy(self.bottom_centers).cuda()

        Gp = self.local_hand_points#
        Gp = torch.transpose(torch.from_numpy(Gp).repeat(self.grasp_centers.shape[0],1,1),1,2).cuda()

        poses_cuda =torch.from_numpy(self.grasp_poses).cuda()
        Cp = torch.matmul(poses_cuda,Gp)+bottom_centers.reshape(-1,3,1)#Cgp

        world_to_scaner_rot_cuda = torch.from_numpy(self.world_to_scaner_rot).\
                                                                    repeat(self.grasp_centers.shape[0],1,1).cuda()
        world_to_scaner_trans_cuda = torch.from_numpy(self.world_to_scaner_trans).cuda()

        Wp =  torch.matmul(world_to_scaner_rot_cuda,Cp)+world_to_scaner_trans_cuda.reshape(1,3,1)#

        #判断夹爪各点是否低于桌面高度，设置容忍值
        lowest_points =torch.min(Wp[:,2,1:],dim = 1,keepdim = True)[0]#np.where()
        #最低点高于桌面的抓取为True
        mask = lowest_points.cpu().numpy()>(self.table_hight+self.safe_dis)
        mask = mask.flatten()#(-1,)

        self.bad_grasp_centers = self.grasp_centers[~mask] #(-1,3)
        self.bad_grasp_poses = self.grasp_poses[~mask]  #(-1,3,3)
        self.bad_bottom_centers=self.bottom_centers[~mask]
        self.bad_hand_points = self.hand_points[~mask]

        self.grasp_centers = self.grasp_centers[mask] #(-1,3)
        self.grasp_poses = self.grasp_poses[mask]  #(-1,3,3)
        self.grasp_scores = self.grasp_scores[mask]#(-1,)
        self.bottom_centers=self.bottom_centers[mask]
        self.hand_points = self.hand_points[mask]

        after =len(self.grasp_centers)
        print('Collision_check_table done:  ',before,' to ',after)

    def restrict_approach_angle(self):
        """仅仅保留抓取approach轴(世界坐标系W)与世界坐标系-z轴夹角 小于max_angle(度)的抓取
                防止抓取失败率过高
        """
        if self.grasp_centers.shape[0]==0:
            return 0

        before = len(self.grasp_centers)

        #抽取出各个抓取approach轴在世界坐标系W下的单位向量
        grasps_approach = self.grasp_poses[:,:,0] #(-1,3)
        #单位化
        grasps_approach = grasps_approach/np.linalg.norm(grasps_approach,axis=1,keepdims=True) #(-1,3)

        cos_angles = grasps_approach.dot(np.array([0,0,-1]).T)  #(-1,)
        #限制抓取approach轴与世界-z轴的角度范围
        mask = cos_angles>np.cos(self.max_angle/180*pi)
        #print(cos_angles[mask],np.cos(45/180*pi))
        self.bad_grasp_centers = self.grasp_centers[~mask]
        self.bad_grasp_poses = self.grasp_poses[~mask]
        self.bad_hand_points = self.hand_points[~mask]
        self.bad_bottom_centers=self.bottom_centers[~mask]

        self.grasp_centers = self.grasp_centers[mask]
        self.grasp_poses = self.grasp_poses[mask]
        self.grasp_scores = self.grasp_scores[mask]
        self.bottom_centers=self.bottom_centers[mask]
        self.hand_points = self.hand_points[mask]

        after =len(self.grasp_centers)
        print('Restrict_approach_angl done:  ',before,' to ',after)

    def collision_check_validate(self):
        """验证夹爪闭合区域是否有点
        """
        #mask =np.zeros(centers.shape[0])
        poses_cuda=torch.from_numpy(self.grasp_poses).cuda()
        local_hand_points= torch.from_numpy(self.local_hand_points).cuda()
        bottom_centers = torch.from_numpy(self.bottom_centers).cuda()
        pc = torch.from_numpy(self.pc).cuda()

        
        gripper_points_p = torch.tensor([local_hand_points[4][0],local_hand_points[2][1],local_hand_points[1][2]]).reshape(1,-1).cuda()
        gripper_points_n = torch.tensor([local_hand_points[8][0],local_hand_points[1][1],local_hand_points[4][2]]).reshape(1,-1).cuda()
        grasps_neighbor_points_index =[]

        #对每个抓取进行碰撞检测
        for i in tqdm(range(len(bottom_centers)),desc='collision_check_validate'):
            #得到标准的旋转矩阵
            matrix = poses_cuda[i]
            #转置=求逆（酉矩阵）
            grasp_matrix = matrix.T  # same as cal the inverse
            #获取所有的点相对于夹爪底部中心点的向量
            points = pc - bottom_centers[i].reshape(1, 3)
            points_g = torch.mm(grasp_matrix, points.T).T    #[len(pc),3]
            #
            points_p = points_g-gripper_points_p #[len(pc),3]
            points_n = points_g-gripper_points_n #[len(pc),3]
            points_index =torch.where(torch.sum((torch.mul(points_p,points_n)<0),dim=1)==3)[0]

            if len(points_index)==0:
                raise ValueError("Grasp has no neighbor points!")
                #print("Grasp has no neighbor points!")
            else:
                grasps_neighbor_points_index.append(points_index.cpu().numpy())

    def get_rot_mat(self,poses_vector):
        '''从抓取向量中计算出夹爪相对于mesh模型的姿态
        '''
        major_pc = poses_vector[:,3:6]  # (-1,3)
        angle = poses_vector[:,[7]]# (-1,1)


        # cal approach
        cos_t = np.cos(angle)   #(-1,1)
        sin_t = np.sin(angle)
        zeros= np.zeros(cos_t.shape)  #(-1,1)
        ones = np.ones(cos_t.shape)

        #绕抓取binormal轴的旋转矩阵
        R1 = np.c_[cos_t, zeros, -sin_t,zeros, ones, zeros,sin_t, zeros, cos_t].reshape(-1,3,3) #[len(grasps),3,3]
        #print(R1)
        axis_y = major_pc #(-1,3)

        #设定一个与抓取y轴垂直且与C:x-o-y平面平行的单位向量作为初始x轴
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


    def get_grasp_angle(self):
        #将抓取转变回8d（7+1）向量保存（相对于相机坐标系）
        binormals=self.grasp_poses[:,:,1] #抽出抓取的binormal轴 退化为(-1,3)
        approachs=self.grasp_poses[:,:,0] #抽出approach轴 (-1,3)
        #计算angles，找到与binormals垂直且平行于C:x-o-y平面的向量
        temp =np.concatenate((binormals[:,[1]],-binormals[:,[0]]),axis =1)  #(-1,2)
        project = np.zeros((temp.shape[0],temp.shape[1]+1))
        project[:,:-1]=temp  #(-1,3)
        #计算投影approach与binormal之间的角度
        cos_angles = np.sum(approachs*project,axis=1)#退化(-1,)
        #检测负号
        minus_mask =approachs[:,2]>0
        #求出angles
        angles = np.arccos(cos_angles)
        angles[minus_mask] = -angles[minus_mask]

    def show_grasps_pc(self,max_n=200,show_bad=False,show_mesh=False,title='scene'):
        """显示点云和抓取，仅限于单进程debug
        """
        mlab.figure(figure=title+'_good grasp',bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))#创建一个窗口
        _ = self.show_points(self.used_pc)
        print(title+'_good grasp:{}'.format(self.grasp_centers.shape[0]))

        #是否把原始的模型也显示出来
        if show_mesh:
            for mesh_index,path in  enumerate(self.mesh_list):
                #mesh = mlab.pipeline.open(path)
                #获取原始点云
                path = path.split('.obj')[0]+'.stl'
                if 'bg' in path:
                    continue

                mesh = o3d.io.read_triangle_mesh(path)
                #得到点云对象
                raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 10)
                #均匀降采样
                voxel_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.001)
                #将点云转换为np对象
                pc = np.asarray(voxel_pc.points).T
                #WTM  模型在场景中的位置姿态
                world_to_mesh_7d = self.mesh_poses_array[mesh_index]
                world_to_mesh_rot = RigidTransform.rotation_from_quaternion(world_to_mesh_7d[3:])
                world_to_mesh_trans = world_to_mesh_7d[0:3]
                world_to_mesh_T = RigidTransform(world_to_mesh_rot,world_to_mesh_trans).matrix

                CTM = np.matmul(self.scaner_to_world_T,world_to_mesh_T)
                ctm_rot = CTM[0:3,0:3]
                ctm_trans =  CTM[0:3,3].reshape(-1,3)

                pc = ctm_rot.dot(pc).T+ctm_trans



                _ = self.show_points(pc,color='b')



        #随机抽选出n个抓取进行显示
        if self.grasp_centers.shape[0]>max_n:
            print(title+'_good grasp:{} ,  show random {} grasps here'.format(self.grasp_centers.shape[0],max_n))
            mask = random.sample(range(self.grasp_centers.shape[0]),max_n)
            good_hand_points = self.hand_points[mask]
        else:
            print(title+'_good grasp:{}'.format(self.grasp_centers.shape[0]))
            good_hand_points = self.hand_points

        #显示good抓取出来
        for index,hand_points in enumerate(good_hand_points):
            ags.show_grasp_3d(hand_points)

            #grasp_bottom_center = -ags.gripper.hand_depth * good_poses[index][:,0] + center
            #显示抓取的编号
            #mlab.text3d(grasp_bottom_center[0],grasp_bottom_center[1],grasp_bottom_center[2],
            #                            str(index),scale = (0.01),color=(1,0,0))

        #显示bad抓取
        if not show_bad:
            pass
        else:
            mlab.figure(figure=title+'_bad grasp',bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))#创建一个窗口
            _ = self.show_points(self.used_pc)
            #如果错误抓取太多，就随机抽选一些,方便显示
            max_n =200 
            if self.bad_centers.shape[0]>max_n:
                print(title+'_bad grasp:{} ,  show random {} grasps here'.\
                    format(self.bad_grasp_centers.shape[0],max_n))
                mask = random.sample(range(self.bad_grasp_centers.shape[0]),max_n)
                bad_hand_points = self.bad_hand_points[mask]
            else:
                print(title+'_bad grasp:{}'.format(self.bad_grasp_centers.shape[0]))
                bad_hand_points = self.bad_hand_points


            for index,hand_points in enumerate(bad_hand_points):
                ags.show_grasp_3d(hand_points)
                #grasp_bottom_center = -ags.gripper.hand_depth * bad_poses[index][:,0] + center
                #显示抓取的编号
                #mlab.text3d(grasp_bottom_center[0],grasp_bottom_center[1],grasp_bottom_center[2],
                #                            str(index),scale = (0.01),color=(1,0,0))
        mlab.show()





    def show_points(self,point, name='pc',color='lb', scale_factor=.004):
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

    do_job = GetLegalGrasps(use_dense=False)

    print('All done')

