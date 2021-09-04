# -*- coding: utf-8 -*-
from math import pi
import os
import sys
import argparse
import time
import mayavi
import numpy as np
import pickle
import glob
from autolab_core import RigidTransform
from autolab_core import YamlConfig
from dexnet.grasping import GpgGraspSampler  
from mayavi import mlab

from dexnet.grasping import RobotGripper
from numpy.core.fromnumeric import swapaxes


#解析命令行参数
parser = argparse.ArgumentParser(description='Get legal grasps with score')
parser.add_argument('--gripper', type=str, default='panda')   #
parser.add_argument('--load_npy',action='store_true')  #设置同时处理几个场景

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
    major_pc = poses_vector[:,3:6]  # (-1,3)
    if poses_vector.shape[1]==11:
        angle = poses_vector[:,[7]]# (-1,1)
    else:
        angle = poses_vector[:,[6]]  #


    # cal approach
    cos_t = np.cos(angle)   #(-1,1)
    sin_t = np.sin(angle)
    zeros= np.zeros(cos_t.shape)  #(-1,1)
    ones = np.ones(cos_t.shape)

    #绕抓取binormal轴的旋转矩阵
    R1 = np.c_[cos_t, zeros, sin_t,zeros, ones, zeros,-sin_t, zeros, cos_t].reshape(-1,3,3) #(-1,3,3)
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

def get_inner_points(grasp_bottom_center, approach_normal, binormal,
                            minor_pc, graspable, p, way, vis=False):
    '''检测夹爪内部点的数量以及
    伸入夹爪内部点的最深的深度
    '''
    #单位化
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)

    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)

    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)

    #得到标准的旋转矩阵
    matrix = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
    #转置=求逆（酉矩阵）
    grasp_matrix = matrix.T  # same as cal the inverse

    points = graspable
    #获取所有的点相对于夹爪底部中心点的向量
    points = points - grasp_bottom_center.reshape(1, 3)
    #points_g = points @ grasp_matrix
    tmp = np.dot(grasp_matrix, points.T)
    points_g = tmp.T
    #p_open 代表，检查闭合区域内部有没有点，不包含夹爪自身碰撞，只是看夹爪内部有没有点云
    if way == "p_open": 
        s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
    #判断右侧夹爪本身，有没有与点云产生碰撞
    elif way == "p_left":
        s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
    #判断左侧夹爪本身，有没有与点云产生碰撞
    elif way == "p_right":
        s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
    #判断顶端夹爪本身，有没有与点云产生碰撞
    elif way == "p_bottom":
        s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
    else:
        raise ValueError('No way!')
    #查找points_g中所有y坐标大于p1点的y坐标
    a1 = s1[1] < points_g[:, 1]    #y      (-1,)   后面是True False
    a2 = s2[1] > points_g[:, 1]
    a3 = s1[2] > points_g[:, 2]    #z
    a4 = s4[2] < points_g[:, 2]
    a5 = s4[0] > points_g[:, 0]    #x
    a6 = s8[0] < points_g[:, 0]

    a = np.vstack([a1, a2, a3, a4, a5, a6])  #(6,-1)  每列都是是否符合判断条件的
    points_in_area_index = np.where(np.sum(a, axis=0) == len(a))[0] #找到符合每一个上述条件的点的索引
    points_in_area=points_g[np.sum(a, axis=0) == len(a)] #(-1,3)
    
    if len(points_in_area_index) == 0:
        #不存在点
        has_p = False
        insert_dist=0
    else:
        has_p = True
        #抽取出夹爪内部点的x轴坐标，并找到最深入的点的x坐标
        deepist_point_x =  np.min(points_in_area[:,0])
        insert_dist = ags.gripper.hand_depth-deepist_point_x


    if vis:
        print("points_in_area", way, len(points_in_area_index))
        mlab.clf()
        # self.show_one_point(np.array([0, 0, 0]))
        args.show_grasp_3d(p)
        #self.show_points(grasp_bottom_center, color='b', scale_factor=.008)
        #注意这一点，在检查的时候，参考系是相机坐标系，夹爪的相对相机的位姿并没有改变，
        # 他们反而是对点云进行了变换，搞不懂这样有什么好处
        args.show_points(points_g)
        # 画出抓取坐标系
        #self.show_grasp_norm_oneside(grasp_bottom_center,approach_normal, binormal,minor_pc, scale_factor=0.001)

        if len(points_in_area_index) != 0:
            args.show_points(points_g[points_in_area_index], color='r')
        mlab.show()

    # print("points_in_area", way, len(points_in_area))
    #返回是否有点has_p，以及，内部点的索引list
    return len(points_in_area_index),insert_dist

def collision_check_pc(centers,poses,scores,pc):
    """对CTG抓取姿态和Cpc进行碰撞检测
    """
    #对旋转后的抓取，逐个进行碰撞检测，并把没有碰撞的抓取保存下来
    bottom_centers = centers -ags.gripper.hand_depth * poses[:,:,0] 
    mask =np.zeros(centers.shape[0])
    #对每个抓取进行碰撞检测
    for i,bottom_center in enumerate(bottom_centers):
        approach_normal = poses[i,:,0] #approach轴
        binormal = poses[i,:,1]#
        minor_pc = poses[i,:,2]
        hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))#
        #检查与点云的碰撞情况
        collision = ags.check_collide(bottom_center,approach_normal,
                                binormal,minor_pc,pc,hand_points,vis=False)
        if  collision:
            mask[i]=1
    good_grasps_center = centers[mask==0]
    good_free_grasps_pose = poses[mask==0]
    good_scores = scores[mask==0]
    bad_grasps_center = centers[mask==1]
    bad_free_grasps_pose = poses[mask==1]
    return good_grasps_center,good_free_grasps_pose,good_scores,bad_grasps_center,bad_free_grasps_pose

def collision_check_table(centers,poses,scores,table_hight = 0.75,safe_dis = 0.01):
    """对夹爪进行批量的碰桌子检测，夹爪上面的最低点不允许低于桌子高度
    """
    #先批量取出虚拟夹爪各点相对于相机坐标系C的坐标
    grasps_hand_points = np.empty([0,21,3])
    for i in range(poses.shape[0]):
        hand_points = ags.get_hand_points(centers[i],
                                                        poses[i,:,0], poses[i,:,1]).reshape(1,-1,3)#
        grasps_hand_points = np.concatenate((grasps_hand_points,hand_points),axis=0)

    #求出虚拟夹爪各个点相对于世界坐标系W的坐标
    w_gp = np.swapaxes(np.matmul(world_to_scaner_rot,np.swapaxes(grasps_hand_points,1,2)),1,2)+\
                                                world_to_scaner_trans.reshape(1,3) #(-1,21,3)
    w_pc=np.matmul(world_to_scaner_rot,pc.T).T+\
                                                world_to_scaner_trans.reshape(1,3) #(-1,3)
    #判断夹爪各点是否低于桌面高度，设置容忍值
    lowest_points = np.min(w_gp[:,1:,2],axis = 1,keepdims = True)#np.where()
    #最低点高于桌面的抓取为True
    mask = lowest_points>table_hight+safe_dis
    mask = mask.flatten()#(-1,)
    temp_grasps_center = centers[mask] #(-1,3)
    temp_grasps_pose = poses[mask]  #(-1,3,3)
    temp_grasps_scores = scores[mask]#(-1,)
    bad_centers = centers[~mask] #(-1,3)
    bad_poses = poses[~mask]  #(-1,3,3)
    return temp_grasps_center,temp_grasps_pose,temp_grasps_scores,bad_centers,bad_poses

def restrict_approach_angle(centers,poses,scores,max_angle=50):
    """仅仅保留抓取approach轴(世界坐标系W)与世界坐标系-z轴夹角 小于max_angle(度)的抓取
            防止抓取失败率过高
    """
    #抽取出各个抓取approach轴在世界坐标系W下的单位向量
    grasps_approach = poses[:,:,0] #(-1,3)
    #单位化
    grasps_approach = grasps_approach/np.linalg.norm(grasps_approach,axis=1,keepdims=True) #(-1,3)

    cos_angles = grasps_approach.dot(np.array([0,0,-1]).T)  #(-1,)
    #限制抓取approach轴与世界-z轴的角度范围
    mask = cos_angles>np.cos(max_angle/180*pi)
    #print(cos_angles[mask],np.cos(45/180*pi))
    temp_grasps_center = centers[mask]
    temp_grasps_pose = poses[mask]
    temp_grasps_score = scores[mask]

    bad_centers = centers[~mask]
    bad_poses = poses[~mask]
    return temp_grasps_center,temp_grasps_pose,temp_grasps_score,bad_centers,bad_poses

def restrict_pc_num_deepth(centers,poses,scores,minimum_points_num=30,minimum_insert_dist=0.01):
    """限制夹爪内部点云点最小数量，以及限制夹爪内部点云的最小深度
    限制点云伸入夹爪内部的高度，忽略掉那些虽然有场景点，但是伸入夹爪内部的点的高度太低的抓取，
    这类抓取有可能是抓取的扁平物体，相机噪声干扰较大，另外，这类抓取很可能不稳定
    """
    # 获取
    bottom_centers = centers -ags.gripper.hand_depth * poses[:,:,0] 
    #对每一个抓取进行内部点数量检测
    mask =np.zeros(centers.shape[0])
    #对每个抓取进行碰撞检测
    for i,bottom_center in enumerate(bottom_centers):
        approach_normal = poses[i,:,0] #approach轴
        binormal = poses[i,:,1]#
        minor_pc = poses[i,:,2]
        hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))#
        #检查夹爪手内部点的数量，以及每个夹爪内部的“最深点”的伸入距离
        inner_points_num,insert_dist = get_inner_points(bottom_center,
                            approach_normal,binormal,minor_pc,pc,hand_points,"p_open",vis=False)
        #设置夹爪内部点的最少点数,以及插入夹爪的最小深度
        if  inner_points_num>minimum_points_num  and insert_dist>minimum_insert_dist:
            mask[i]=1
    #保留内部点数足够的采样抓取
    temp_grasps_center = centers[mask==1]
    temp_grasps_pose = poses[mask==1]
    temp_grasps_score = scores[mask==1]
    bad_centers = centers[mask==0]
    bad_poses = poses[mask==0]
    return temp_grasps_center,temp_grasps_pose,temp_grasps_score,bad_centers,bad_poses

def show_grasps_pc(pc,good_centers,good_poses,bad_centers=None,bad_poses=None,title=None):
    """显示点云和抓取，仅限于单进程debug
    """
    mlab.figure(figure=title,bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
    _ = show_points(pc)
    for index,center in enumerate(good_centers):
        display_grasps(center,good_poses[index],color="d")
        grasp_bottom_center = -ags.gripper.hand_depth * good_poses[index][:,0] + center
        #显示抓取的编号
        mlab.text3d(grasp_bottom_center[0],grasp_bottom_center[1],grasp_bottom_center[2],
                                    str(index),scale = (0.01),color=(1,0,0))

    if bad_centers is None or bad_poses is None:
        pass
    else:
        for index,center in enumerate(bad_centers):
            display_grasps(center,bad_poses[index],color=(1,0,0))
            grasp_bottom_center = -ags.gripper.hand_depth * bad_poses[index][:,0] + center
            #显示抓取的编号
            mlab.text3d(grasp_bottom_center[0],grasp_bottom_center[1],grasp_bottom_center[2],
                                        str(index),scale = (0.01),color=(1,0,0))
    mlab.show()
    



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
        grasps_score = np.empty(shape=(0,1))

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
                    grasps_with_score = np.load(path) #(-1,11)

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


            #将所有的模型的grasp  统一添加到场景的grasp 列表中
            #grasps_center = np.r_[grasps_center,scaner_to_grasps_trans]
            grasps_center = np.concatenate((grasps_center,scaner_to_grasps_trans),axis=0)
            #grasps_pose = np.r_[grasps_pose,scaner_to_grasps_rot]
            grasps_pose=np.concatenate((grasps_pose,scaner_to_grasps_rot),axis=0)
            #抽出对应分数
            grasps_score = np.concatenate((grasps_score,grasps_with_score[:,[10]]),axis=0)  #(-1,1)


        if args.load_npy:
            legal_grasps_vector=np.load(os.path.join(os.path.split(raw_pc_path)[0],'legal_grasps_with_score.npy'))
            #将拼接好的7d抓取向量，再次转化为旋转和平移两部分，显示出来,debug
            temp_good_grasps_center = legal_grasps_vector[:,0:3]
            temp_good_grasps_pose = get_rot_mat(legal_grasps_vector)
            #显示点云与抓取，仅限于debug
            show_grasps_pc(pc,temp_good_grasps_center,temp_good_grasps_pose,title='外部导入结果')

        else:
            #对旋转后的抓取，逐个进行碰撞检测，并把没有碰撞的抓取保存下来
            temp_good_grasps_center,temp_good_grasps_pose,temp_good_grasps_score,bad_centers,bad_poses\
                =collision_check_pc(grasps_center,grasps_pose,grasps_score,pc)

            #显示点云与抓取，仅限于debug
            #show_grasps_pc(pc,temp_good_grasps_center,temp_good_grasps_pose,
            #bad_centers,bad_poses,title='夹爪点云碰撞检测')

            #保存不与桌子碰撞的抓取
            temp_good_grasps_center,temp_good_grasps_pose,temp_good_grasps_score,bad_centers,bad_poses\
                =collision_check_table(temp_good_grasps_center,temp_good_grasps_pose,temp_good_grasps_score)

            #显示点云与抓取，仅限于debug
            #show_grasps_pc(pc,temp_good_grasps_center,temp_good_grasps_pose,
            #bad_centers,bad_poses,title='夹爪桌面碰撞检测')

            #限制抓取approach轴与桌面垂直方向的角度
            temp_good_grasps_center,temp_good_grasps_pose,temp_good_grasps_score,bad_centers,bad_poses\
                =restrict_approach_angle(temp_good_grasps_center,temp_good_grasps_pose,
                temp_good_grasps_score,max_angle=50)

            #显示点云与抓取，仅限于debug
            #show_grasps_pc(pc,temp_good_grasps_center,temp_good_grasps_pose,
            #bad_centers,bad_poses,title='限制抓取角度')

            #限制夹爪内部点云点最小数量，以及限制夹爪内部点云的最小深度
            temp_good_grasps_center,temp_good_grasps_pose,temp_good_grasps_score,bad_centers,bad_poses\
                =restrict_pc_num_deepth(temp_good_grasps_center,temp_good_grasps_pose,
                temp_good_grasps_score,30,0.01)

            #显示点云与抓取，仅限于debug
            show_grasps_pc(pc,temp_good_grasps_center,temp_good_grasps_pose,
            bad_centers,bad_poses,title='限制夹爪内部点数与深度')


            #将抓取转变回8d（7+1）向量保存（相对于相机坐标系）
            binormals=temp_good_grasps_pose[:,:,1] #抽出抓取的binormal轴 退化为(-1,3)
            approachs=temp_good_grasps_pose[:,:,0] #抽出approach轴 (-1,3)
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

            #先拼接位置姿态
            legal_grasps_vector = np.concatenate((np.concatenate((temp_good_grasps_center,binormals),axis=1),
                                                            angles.reshape(-1,1)),axis=1) #(-1,7)
            #再接上分数
            legal_grasps_vector = np.concatenate((legal_grasps_vector,temp_good_grasps_score),axis=1)

            print(legal_grasps_vector)
            #将拼接好的7d抓取向量，再次转化为旋转和平移两部分，显示出来,debug
            temp_good_grasps_center = legal_grasps_vector[:,0:3]
            temp_good_grasps_pose = get_rot_mat(legal_grasps_vector)
            show_grasps_pc(pc,temp_good_grasps_center,temp_good_grasps_pose,title='重显')
        


        #保存为'legal_grasp_with_score.npy'
        #save_path =os.path.join(os.path.split(raw_pc_path)[0],'legal_grasps_with_score.npy')
        #np.save(save_path,legal_grasps_vector)








