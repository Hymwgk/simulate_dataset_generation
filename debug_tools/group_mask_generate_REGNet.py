# -*- coding: utf-8 -*-
from logging import raiseExceptions
from math import pi
import math
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
from mayavi import mlab
from torch._C import set_autocast_enabled
from tqdm import tqdm
from numpy.core.fromnumeric import swapaxes, transpose
import torch
from tqdm import tqdm
import multiprocessing

from dexnet.grasping import GpgGraspSampler  
from autolab_core import RigidTransform
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper



#解析命令行参数
parser = argparse.ArgumentParser(description='Group and mask generation')
parser.add_argument('--gripper', type=str, default='baxter')   #
parser.add_argument('--load_npy',action='store_true')  #设置同时处理几个场景
parser.add_argument('--process_num', type=int, default=50)  #设置同时处理几个场景
parser.add_argument('--dis_min', type=float, default=0.01)  #距离聚类
parser.add_argument('--theta_min', type=float, default=0.707)  #角度聚类

args = parser.parse_args()
home_dir = os.environ['HOME']
#场景文件夹
scenes_dir = home_dir+"/dataset/REGnet/temp_training_data/"
data_path = glob.glob(scenes_dir+'*.p')#一共19000+ 帧场景
data_path.sort()
mask_files_dir  =  home_dir+"/dataset/REGnet/mask_files/"






yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")
gripper = RobotGripper.load(args.gripper, home_dir + "/code/dex-net/data/grippers")
ags = GpgGraspSampler(gripper, yaml_config)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def get_rot_mat(poses_vector):
    center_point = poses_vector[:,0:3]    #夹爪中心(指尖中心) 
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
    R3=np.matmul(R2,R1)  #(-1,3,3)
    '''
    approach_normal =R3[:, :,0]
    #print(np.linalg.norm(approach_normal,axis=1,keepdims=True))
    approach_normal = approach_normal / np.linalg.norm(approach_normal,axis=1,keepdims=True)
    #minor_pc=R3[:, :,2]  是一样的
    minor_pc = np.cross( approach_normal,major_pc)
    '''
    #然后把平移向量放在每个旋转矩阵的最右侧，当成一列
    return R3

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def axis_angle(quaternion):
    """:obj:`np.ndarray` of float: The axis-angle representation for the rotation.
    """
    qw, qx, qy, qz = quaternion
    theta = 2 * np.arccos(qw)
    return abs(theta)

def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()

    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return abs(angle)


def posesTransWithinErrorBounds(trans1,trans2,rot1,rot2,trans_diff_threshold=0.05,rot_diff_threshold=0.5):
    '''计算两个抓取中心、pose之间的偏差是否超过阈值
    '''
    trans_diff = np.linalg.norm(trans1 - trans2)#抓取中心偏差距离
    rot1_inverse = rot1.T
    rot_angle_diff = rotation_from_matrix(rot1_inverse.dot(rot2))
    if trans_diff<trans_diff_threshold   and  rot_angle_diff<rot_diff_threshold :
        return True
    else:
        return False

def posesTransWithinErrorBoundsCuda(index_i,index_j,centers_diff,theta_diff):
    '''计算两个抓取中心、pose之间的偏差是否超过阈值
    '''
    #centers_diff = centers_diff<dis_min
    #theta_diff = theta_diff<theta_min

    if index_i>index_j:
        centers_diff_ = centers_diff[index_j,index_i]
        theta_diff_=theta_diff[index_j,index_i]
    elif index_i<index_j:
        centers_diff_ = centers_diff[index_i,index_j]
        theta_diff_=theta_diff[index_i,index_j]
    else:
        raise TypeError("index wrong")

    if  centers_diff_  and  theta_diff_ :
        return True
    else:
        return False

# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]

def do_job(scene_index):

    #读取数据集数据
    data = np.load(data_path[scene_index], allow_pickle=True)#读取样本数据
    grasp_score = data['select_score'].astype(np.float32)#点云三通道坐标数据

    legal_grasps = data['select_frame'].astype(np.float32)#点云三通道坐标数据

    #print('Scene  {}'.format(scene_index)+ '  legal grasp:{}'.format(len(legal_grasps)))
    grasp_centers =  legal_grasps[:,:3,3]#抓取中心 [len(grasp),3]
    grasp_poses = legal_grasps[:,:3,:3]#抓取旋转矩阵 [len(grasp),3,3]
    scene_name = data_path[scene_index].split('/')[-1].split('.')[0]
    scene_dir = mask_files_dir+scene_name+'/'
    pickel_name = scene_dir+'clusterMatrix_{}_{}.pickle'.format(args.dis_min,args.theta_min)

    #读取场景grasps
    cluster_pickel_name = scene_dir+'graspClusterWithScores_{}_{}.pickle'.format(args.dis_min,args.theta_min)

    if  not os.path.exists(cluster_pickel_name):
        cluster_pickel_name_ = glob.glob(scene_dir+'graspClusterWithScores*.pickle')
        if len(cluster_pickel_name_):#如果存在生成好的文件
            #删除之前的文件
            for file in cluster_pickel_name_:
                os.remove(file)
                print('Delete' + file)


        #读取聚类矩阵
        with open(clusterMatrixPaths[scene_index], 'rb') as f:
            clusterMatrix= pickle.load(f)
        grasp_clusters = []

        #random_grasp_index=list(range(len(legal_grasps)))
        #random.shuffle(random_grasp_index)

        for j in range(len(legal_grasps)):#为总编号为j的抓取
            '''为总编号为j的抓取分配集合簇
            '''
            #print(j)
            found_cluster = False
            for cluser_i in range(len(grasp_clusters)):#
                '''查看当前抓取是否与每个簇内部的抓取相近
                '''
                for grasp_index  in  grasp_clusters[cluser_i][0]:
                    '''判断当前抓取与簇中的每个抓取的距离是否在误差范围内
                    '''
                    ptWithinError = posesTransWithinErrorBoundsCuda(j,grasp_index,clusterMatrix[0],clusterMatrix[1])
                    #ptWithinError = True
                    if ptWithinError:#如果簇cluser_i中的某个抓取与当前抓取相近
                        found_cluster = True
                        grasp_clusters[cluser_i][1]+= grasp_score[j]#更新group分数
                        grasp_clusters[cluser_i][0].append(j)
                        break

                if found_cluster:#如果找到归属簇，就不再比较下一个簇
                    break
                

            if not found_cluster:#
                grasp_group =[]#创建新的簇
                grasp_group_score = 0 #簇分数
                grasp_group.append(j)#把当前抓取index存在簇中
                grasp_group_score = grasp_score[j]
                grasp_clusters.append([grasp_group,grasp_group_score])#把当前簇保存在簇集合中
        

        #对簇中的抓取计算分数，并按照分数对簇进行降序
        grasp_clusters.sort(key=takeSecond,reverse = True)
        #截取排名前30的簇（如果有的话）
        #if len(grasp_clusters)>30:
        #    grasp_clusters = grasp_clusters[:30]

        #保存group with score
        pickel_name = scene_dir+'graspClusterWithScores.pickle'
        with open(cluster_pickel_name, 'wb') as f:
            pickle.dump(grasp_clusters, f)

        #当前场景聚类完毕    
        print('Scene  {}'.format(scene_index)+ '  legal grasp:{}'.format(len(legal_grasps))+' done')
    else:
        print(cluster_pickel_name+' Exist')  


       
def collision_check_pc_cuda(centers,poses,pc):
    """对CTG抓取姿态和Cpc进行碰撞检测(使用显卡加速计算)
    返回符合要求的，场景点云点的索引，每个抓取都有自己的内部点索引，是一个二维list
    """
    #对旋转后的抓取，逐个进行碰撞检测，并把没有碰撞的抓取保存下来
    bottom_centers = centers -ags.gripper.hand_depth * poses[:,:,0] 
    #获得扩大夹爪尺寸
    hand_points = ags.get_hand_points_extend(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))#

    #mask =np.zeros(centers.shape[0])
    poses_cuda=torch.from_numpy(poses).cuda()
    hand_points= torch.from_numpy(hand_points).cuda()
    bottom_centers = torch.from_numpy(bottom_centers).cuda()
    pc = torch.from_numpy(pc).cuda()

    
    #gripper_points_p = torch.tensor([hand_points[10][0],hand_points[13][1],hand_points[9][2]]).reshape(1,-1).cuda()
    #gripper_points_n = torch.tensor([hand_points[20][0],hand_points[9][1],hand_points[10][2]]).reshape(1,-1).cuda()
    gripper_points_p = torch.tensor([hand_points[4][0],hand_points[2][1],hand_points[1][2]]).reshape(1,-1).cuda()
    gripper_points_n = torch.tensor([hand_points[8][0],hand_points[1][1],hand_points[4][2]]).reshape(1,-1).cuda()
    grasps_neighbor_points_index =[]

    #对每个抓取进行碰撞检测
    for i in tqdm(range(len(bottom_centers)),desc='collision_check_pc'):
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



        #points_in_close_area=points_g[check_op] #(-1,3)
        #if points_in_gripper_index.shape[0] == 0:#不存在夹爪点云碰撞
        if len(points_index)==0:
            print(centers[i],i)
            #print(centers[i-1],poses[i-1],i-1)
            #print(centers[i+1],poses[i+1],i+1)
            raise ValueError("Grasp has no neighbor points!")
            #print("Grasp has no neighbor points!")
        else:
            grasps_neighbor_points_index.append(points_index.cpu().numpy())
    
    return grasps_neighbor_points_index

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

def show_points(pc, clusters_sets=[],centers=[],poses=[],name='Scene Mask', scale_factor=.004):
    '''显示生成结果
    cluster_0 = pc[list(clusters_sets[0])]
    cluster_1 = pc[list(clusters_sets[1])]
    cluster_2 = pc[list(clusters_sets[2])]
    cluster_3 = pc[list(clusters_sets[3])]
    cluster_4 = pc[list(clusters_sets[4])]    
    '''
    m=len(clusters_sets)
    color_ = [(0, 0, 1),(1, 0, 0),(0, 1, 0),(0.22, 1, 1),(0.8, 0.8, 0.2),
                        (0.5, 0, 1),(1, 0.62, 0),(0.53, 1, 0),(0.22, 0.63, 1),(0.8, 0.22, 0.2)]

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81),size=(1800,1800))
    mlab.points3d(pc[:, 0], pc[:, 1],
             pc[:, 2], color=(0.5,0.5,0.5), scale_factor=scale_factor)
    
    if m==0:
        mlab.show()
    else:
        if m>10:
            m=10
            
        m-=1
        for i in range(m):
            j=m-i
            mlab.points3d(pc[list(clusters_sets[j])][:, 0], pc[list(clusters_sets[j])][:, 1],
                pc[list(clusters_sets[j])][:, 2], color=color_[j], scale_factor=scale_factor)
    
    if len(centers)==0 or len(poses)==0:
        mlab.show()
    else:
        for index,center in enumerate(centers):
            display_grasps(center,poses[index],color="d")
        mlab.show()



if __name__ == '__main__':

    #先用显卡进行差异矩阵的计算
    for i in range(len(data_path)):#第i个场景
        scene_name = data_path[i].split('/')[-1].split('.')[0]
        scene_dir = mask_files_dir+scene_name+'/'
        pickel_name = scene_dir+'/clusterMatrix_{}_{}.pickle'.format(args.dis_min,args.theta_min)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)

        if not os.path.exists(pickel_name):
            pickel_name_ = glob.glob(mask_files_dir+scene_name+'/clusterMatrix*.pickle')
            if len(pickel_name_):#如果存在生成好的文件
                #删除之前的文件
                for file in pickel_name_:
                    os.remove(file)
                    print('Delete' + file)
            

            #读取数据集数据
            data = np.load(data_path[i], allow_pickle=True)#读取样本数据
            #pc = data['view_cloud'].astype(np.float32)#点云三通道坐标数据

            #grasp_score = data['select_score'].astype(np.float32)#点云三通道坐标数据

            legal_grasps = data['select_frame'].astype(np.float32)#点云三通道坐标数据

            print('Scene  {}'.format(i)+ '  legal grasp:{}'.format(len(legal_grasps)))
            grasp_centers =  legal_grasps[:,:3,3]#抓取中心 [len(grasp),3]
            grasp_poses = legal_grasps[:,:3,:3]#抓取旋转矩阵 [len(grasp),3,3]

            #计算距离差矩阵
            grasp_centers_cuda = torch.from_numpy(grasp_centers).cuda().unsqueeze(0)#[1,len(grasps),3]
            grasp_centers_cuda_T = grasp_centers_cuda.clone().permute(1,0,2)#[len(grasps),1,3]

            grasp_centers_cuda = grasp_centers_cuda.repeat(grasp_centers.shape[0],1,1)
            grasp_centers_cuda_T = grasp_centers_cuda_T.repeat(1,grasp_centers.shape[0],1)

            grasp_centers_diff =grasp_centers_cuda-grasp_centers_cuda_T
            grasp_centers_diff_dis = torch.norm(grasp_centers_diff,dim=2)<args.dis_min#[len(grasps),len(grasps)]
            #grasp_centers_diff_dis


            #
            grasp_poses_cuda = torch.from_numpy(grasp_poses).cuda()#[len(grasps),3,3]
            #姿态逆矩阵
            grasp_poses_inverse_cuda=grasp_poses_cuda.clone().permute(0,2,1)#.repeat(1,grasp_poses_inverse_cuda.shape[0],1,1) #转置求逆,增加维度[len(grasps),1,3,3]
            
            grasp_poses_inverse_cuda=grasp_poses_inverse_cuda.unsqueeze(1).\
                repeat(1,grasp_poses_inverse_cuda.shape[0],1,1)
            grasp_poses_cuda = grasp_poses_cuda.unsqueeze(0).\
                repeat(grasp_poses_inverse_cuda.shape[0],1,1,1)#[len(grasps),len(grasps),3,3]

            grasp_poses_diff =  grasp_poses_inverse_cuda.matmul(grasp_poses_cuda)#[len(grasps),len(grasps),3,3]

            #l = torch.trace(grasp_poses_diff[1,1,:3,:3])
            mask =  torch.zeros((grasp_poses_inverse_cuda.shape[0], grasp_poses_inverse_cuda.shape[0],3, 3)).cuda()
            mask[:, :,torch.arange(0,3), torch.arange(0,3) ] = 1.0

            grasp_poses_diff = grasp_poses_diff.matmul(mask.float())
            grasp_poses_diff_trace = torch.sum(grasp_poses_diff,dim=(2,3))#[len(grasp),len(grasp)]

            cosa = (grasp_poses_diff_trace - 1.0) / 2.0 #[len(grasp),len(grasp)]
            theta = torch.abs(torch.acos(cosa))<args.theta_min#[len(grasp),len(grasp)]
            


            #保存距离矩阵
            with open(pickel_name, 'wb') as f:
                pickle.dump((grasp_centers_diff_dis.cpu().numpy(),theta.cpu().numpy()), f)

            print('Scene {}'.format(i) + 'done')    
        else:
            print(pickel_name+' Exist')  




    #多进程进行场景抓取聚类，排序
    #读取列表
    clusterMatrixPaths =glob.glob(mask_files_dir+'*/clusterMatrix*.pickle')
    clusterMatrixPaths.sort()
    '''
    pool_size=args.process_num  #选择同时使用多少个进程处理
    cores = multiprocessing.cpu_count()
    if pool_size>cores:
        pool_size = cores
        print('Cut pool size down to cores num')
    if pool_size>len(data_path):
        pool_size = len(data_path)
    print('We got {} raw pc, and pool size is {}'.format(len(data_path),pool_size))
    scene_index = 0
    pool = []
    for i in range(pool_size):  
        pool.append(multiprocessing.Process(target=do_job,args=(scene_index,)))
        scene_index+=1
    [p.start() for p in pool]  #启动多线程

    while scene_index<len(data_path):    #如果有些没处理完
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                p = multiprocessing.Process(target=do_job, args=(scene_index,))
                scene_index+=1
                p.start()
                pool.append(p)
                break
    [p.join() for p in pool]  #启动多线程
    '''

    #聚类排序完毕，然后计算grasp_cluster的点云分割
    #对每个场景的每个合法抓取进行夹爪周边点云索引计算，并保存
    for scene_index in range(len(data_path)):

        #读取数据集数据
        data = np.load(data_path[i], allow_pickle=True)#读取样本数据
        grasp_score = data['select_score'].astype(np.float32)#点云三通道坐标数据
        legal_grasps = data['select_frame'].astype(np.float32)#点云三通道坐标数据
        scene_name = data_path[i].split('/')[-1].split('.')[0]
        scene_dir = mask_files_dir+scene_name+'/'


        #print('Scene  {}'.format(i)+ '  legal grasp:{}'.format(len(legal_grasps)))
        grasp_centers =  legal_grasps[:,:3,3]#抓取中心 [len(grasp),3]
        grasp_poses = legal_grasps[:,:3,:3]#抓取旋转矩阵 [len(grasp),3,3]
        pc = data['view_cloud'].astype(np.float32)#点云三通道坐标数据 
        show_points(pc,centers=grasp_centers,poses=grasp_poses)
        
        print('Scene {}: Looking for grasp neighbor points'.format(scene_index))

        grasps_neighbor_points_index=collision_check_pc_cuda(grasp_centers,grasp_poses,pc)

        pickel_path = scene_dir+'graspsNeighborPointsIndex.pickle'


        #保存每个抓取周边的point index  
        with open(pickel_path, 'wb') as f:
            pickle.dump(grasps_neighbor_points_index, f)


    #利用每个抓取的周边点，计算每个cluster的周边点，并记录下来
    #读取聚类文件
    clusterWithScorePaths =glob.glob(scene_dir+'*/graspClusterWithScores*.pickle')
    clusterWithScorePaths.sort()
    graspNeighborPointsIndexPaths = glob.glob(scene_dir+'*/graspsNeighborPointsIndex.pickle')
    graspNeighborPointsIndexPaths.sort()



    for scene_index,pc_path in enumerate(raw_pc_paths):
        #读取场景grasps
        legal_grasps = np.load(legal_grasp_file_paths[scene_index])
        print('Scene  {}'.format(i)+ '  legal grasp:{}'.format(len(legal_grasps)))
        grasp_centers =  legal_grasps[:,0:3]#抓取中心
        grasp_poses = get_rot_mat(legal_grasps)#抓取旋转矩阵


        pc = np.load(pc_path)#场景点云
        with open(clusterWithScorePaths[scene_index], 'rb') as f:
            clusterWithScore= pickle.load(f)#聚类结果（抓取index）
        with open(graspNeighborPointsIndexPaths[scene_index], 'rb') as f:
            graspNeighborPointsIndex= pickle.load(f)#每个grasp的周边点索引

        clusters_sets = []#所有簇对应的内部点索引集合

        #计算每个cluster的索引
        for cluster in clusterWithScore:
            cluster_grasps_index = cluster[0]  #当前簇内部的抓取index

            points_in_cluster = set()#当前簇中的点索引集合

            for grasp_index in cluster_grasps_index:
                points_in_grasp = set(graspNeighborPointsIndex[grasp_index])#grasp_index 对应的抓取内部的点index转化为集合
                #求与当前簇中点索引集合的并集
                points_in_cluster = points_in_cluster.union(points_in_grasp)
            
            clusters_sets.append(points_in_cluster)#所有簇对应的内部点索引集合
        #print('Scene {} got {} raw clusters',format(scene_index,len(clusters_sets)))
        raw_cluster_num = len(clusters_sets)
        #计算完毕每个簇内部的点索引集合，还需要对这些集合进行一个合法性筛选
        #主要是剔除掉交叉面积过多的簇，不能允许簇与簇之间的重合过多
        clusters_index = np.array(list(range(raw_cluster_num)))

        clusters_sets = np.array(clusters_sets)
        show_points(pc,clusters_sets)
        #mlab.close()


        ref_index = 0
        while ref_index < len(clusters_sets)-1:
            delete_set_index = []
            ref_set = clusters_sets[ref_index] #取出当前参考簇

            next_ref_index = ref_index+1
            current_ref_index = ref_index
            #从参考簇下一个开始与参考簇对比
            for target_set_index in range(next_ref_index,len(clusters_sets)):
                target_set = clusters_sets[target_set_index]#取出目标集
                inter_set = ref_set.intersection(target_set)#求 "参考集-目标集"之间的交集

                iou_t =  len(inter_set)/len(target_set)#交集 占 目标集的比例
                iou_r = len(inter_set)/len(ref_set)#交集 占 参考集的比例

                if iou_t>0.8 and iou_r>0.8:
                    delete_set_index.append(target_set_index)#交集占两者比例都大于80% 删除目标集
                    clusters_sets[ref_index] = clusters_sets[ref_index] | clusters_sets[target_set_index]
                    ref_index=next_ref_index#
                elif iou_t<0.8 and iou_r>0.8:
                    if current_ref_index in delete_set_index:
                        pass
                    else:
                        delete_set_index.append(current_ref_index)#交集占参考集比例大于80% 删除参考集
                elif iou_t>0.8 and iou_r<0.8:
                    delete_set_index.append(target_set_index)#交集占目标集比例大于80% 删除目标集
                    ref_index=next_ref_index
                else:
                    ref_index=next_ref_index#交集占目标集与参考集比例都不超过80%，不删除集合
                
            #在所有的簇比对完成之后，才进行最终的删除
            clusters_sets = np.delete(clusters_sets,delete_set_index)
            clusters_index = np.delete(clusters_index,delete_set_index)

            #print('scene: {},raw cluster num: {}, current cluster num {},Ref index: {} '.
            #    format(scene_index,raw_cluster_num,len(clusters_sets),ref_index))

        grasp_index = []
        for index in  clusters_index:
            grasp_index=grasp_index+clusterWithScore[index][0]



        grasp_centers = grasp_centers[grasp_index]
        grasp_poses=grasp_poses[grasp_index]
        grasp_centers = grasp_centers[np.random.choice(len(grasp_centers), size=100, replace=True)]
        grasp_poses = grasp_poses[np.random.choice(len(grasp_poses), size=100, replace=True)]
        #显示结果
        #show_points(pc,clusters_sets,grasp_centers,grasp_poses)
        #show_points(pc,clusters_sets)
        show_points(pc)
        #mlab.close()
        
        print('Generate scene {} mask done, from {} to {}'.format(scene_index, raw_cluster_num, len(clusters_sets)))






        
print('All job done')





                





    


