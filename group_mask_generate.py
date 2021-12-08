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
import copy
from dexnet.grasping import GpgGraspSampler  
from autolab_core import RigidTransform
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper



#解析命令行参数
parser = argparse.ArgumentParser(description='Group and mask generation')
parser.add_argument('--gripper', type=str, default='baxter')   #
parser.add_argument('--load_npy',action='store_true')  #设置同时处理几个场景
parser.add_argument('--process_num', type=int, default=70)  #设置同时处理几个场景
parser.add_argument('--dis_min', type=float, default=0.01)  #距离聚类
parser.add_argument('--theta_min', type=float, default=0.706)  #角度聚类
parser.add_argument('--redo', type=bool, default=False)  #是否重新计算聚类矩阵


parameters = parser.parse_args()
home_dir = os.environ['HOME']
#场景文件夹
scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(parameters.gripper)
legal_grasp_file_paths = glob.glob(scenes_dir+'*/legal_grasps_with_score.npy')
legal_grasp_file_paths.sort()

table_pc_list = glob.glob(scenes_dir+'*/table_pc.npy')
table_pc_list.sort()


yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")
gripper = RobotGripper.load(parameters.gripper, home_dir + "/code/dex-net/data/grippers")
graspSampler = GpgGraspSampler(gripper, yaml_config)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)






class Mask_generate:
    '''根据已有的抓取生成点云mask
    '''
    def __init__(self,grasp_file_paths,parameters,sampler,show_result='False',dense_pc=False) -> None:
        '''grasp_file_paths  抓取文件的路径
            parameters   命令行参数
            sampler   dex抓取采样器
        '''
        self.grasp_file_paths=grasp_file_paths
        #self.process_num=process_num
        self.scene_num = len(grasp_file_paths)
        self.parameters = parameters
        self.sampler = sampler
        self.dense_pc =dense_pc

        if show_result:
            #显示结果，测试用
            self.show_result()

        else:
            #首先计算每个抓取之间的差别
            self.compute_grasp_diff()
            #多进程进行场景抓取聚类，排序
            self.grasp_grouping()
            #对每个场景的每个合法抓取进行夹爪周边点云索引计算，并保存
            self.compute_neighbor_points()
            #计算mask
            self.mask_generate()


    
    def show_result(self):
        for scene_index,grasp_file in enumerate(self.grasp_file_paths):
            #读取场景grasps
            data = np.load(grasp_file, allow_pickle=True).item()
            grasp_num = len(data['grasp_centers'])

            print('Scene  {}'.format(scene_index)+ '  legal grasp:{}'.format(grasp_num))

            if self.dense_pc:
                pc = data['dense_pc']
            else:
                pc = data['pc'] #场景点云
            mask = data['grasp_mask']
            self.show_mask(pc,mask) 
            time.sleep(0.5)
            #mlab.close()     


    def temp(self,table_pc_list):
        for scene_index,grasp_file in enumerate(self.grasp_file_paths):
            #读取场景grasps
            data = np.load(grasp_file, allow_pickle=True).item()
            #ta = np.load(table_pc_list[scene_index])
            #data['dense_pc'] = ta
            #del data['table_pc']

            #先批量取出虚拟夹爪各点相对于相机坐标系C的坐标
            hand_points = np.empty([0,21,3])
            grasps_pose = data['grasp_poses']
            bottom_centers = data['bottom_centers']
            for i in range(grasps_pose.shape[0]):
                points = self.sampler.get_hand_points(bottom_centers[i],
                                                                grasps_pose[i,:,0], 
                                                                grasps_pose[i,:,1]).reshape(1,-1,3)#(1,21,3)
                hand_points = np.concatenate((hand_points,points),axis=0)#(-1,21,3)
            data['hand_points'] = hand_points

            np.save(grasp_file,data)
            print('Scene {}'.format(scene_index) + 'done')    



    def compute_grasp_diff(self):
        '''
        '''
        for scene_index,grasp_file in enumerate(self.grasp_file_paths):
            #读取场景grasps
            #读取场景grasps
            data = np.load(grasp_file, allow_pickle=True).item()

            print('Scene  {}'.format(scene_index)+ '  legal grasp:{}'.format(len(data['grasp_centers'])))
            grasp_centers = data['grasp_centers'].astype(np.float32)   #抓取中心
            grasp_poses =data['grasp_poses'].astype(np.float32)          #抓取旋转矩阵

            #计算距离差矩阵
            grasp_centers_cuda = torch.from_numpy(grasp_centers).cuda().unsqueeze(0)#[1,len(grasps),3]
            grasp_centers_cuda_T = grasp_centers_cuda.clone().permute(1,0,2)#[len(grasps),1,3]

            grasp_centers_cuda = grasp_centers_cuda.repeat(grasp_centers.shape[0],1,1)
            grasp_centers_cuda_T = grasp_centers_cuda_T.repeat(1,grasp_centers.shape[0],1)

            grasp_centers_diff =grasp_centers_cuda-grasp_centers_cuda_T
            grasp_centers_diff_dis = torch.norm(grasp_centers_diff,dim=2)< self.parameters.dis_min#[len(grasps),len(grasps)]
            #grasp_centers_diff_dis


            #计算
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
            theta = torch.abs(torch.acos(cosa))< self.parameters.theta_min#[len(grasp),len(grasp)]

            #保存距离矩阵
            data['grasp_dis_diff'] = grasp_centers_diff_dis.cpu().numpy()
            data['grasp_pose_diff'] = theta.cpu().numpy()

            #更新数据
            np.save(grasp_file,data)
            print('Scene {}'.format(scene_index) + 'done')    


    def grasp_grouping(self):
        '''多线程对每个抓取进行聚类
        '''
        pool_size=self.parameters.process_num  #选择同时使用多少个进程处理
        cores = multiprocessing.cpu_count()
        if pool_size>cores:
            pool_size = cores
            print('Cut pool size down to cores num')
        if pool_size>self.scene_num:
            pool_size = self.scene_num
        print('We got {} pc, and pool size is {}'.format(self.scene_num,pool_size))
        scene_index = 0
        pool = []
        for i in range(pool_size):  
            pool.append(multiprocessing.Process(target=self.do_job,args=(scene_index,)))
            scene_index+=1
        [p.start() for p in pool]  #启动多线程

        while scene_index<self.scene_num:    #如果有些没处理完
            for ind, p in enumerate(pool):
                if not p.is_alive():
                    pool.pop(ind)
                    p = multiprocessing.Process(target=self.do_job, args=(scene_index,))
                    scene_index+=1
                    p.start()
                    pool.append(p)
                    break
        [p.join() for p in pool]  #启动多线程


    def compute_neighbor_points(self):
        '''
        '''
        for scene_index,grasp_file in enumerate(self.grasp_file_paths):
            if scene_index==14:
                print('a')

            data = np.load(grasp_file, allow_pickle=True).item()
            print('Scene {}: Looking for grasp neighbor points'.format(scene_index))
            grasps_neighbor_points_index=self.get_gripper_neighor_points_cuda(data)

            #保存每个抓取周边的point index  
            data['grasp_neighbor_points'] = grasps_neighbor_points_index
            #更新数据
            np.save(grasp_file,data)


    def mask_generate(self):
        for scene_index,grasp_file in enumerate(self.grasp_file_paths):
            #读取场景grasps
            data = np.load(grasp_file, allow_pickle=True).item()
            grasp_num = len(data['grasp_centers'])

            print('Scene  {}'.format(scene_index)+ '  legal grasp:{}'.format(grasp_num))

            if self.dense_pc:
                pc = data['dense_pc']
            else:
                pc = data['pc'] #场景点云
            grasp_clusters = data['grasp_clusters']
            grasp_neighbor_points = data['grasp_neighbor_points']
            #hand_points = data['hand_points']

            clusters_sets = []#所有簇对应的内部点索引集合

            #计算每个cluster内各个点的索引
            for cluster in grasp_clusters:
                cluster_grasps_index = cluster[0]  #当前簇内部的抓取index
                points_in_cluster = set()#当前簇中的点索引集合

                for grasp_index in cluster_grasps_index:
                    points_in_grasp = set(grasp_neighbor_points[grasp_index])#grasp_index 对应的抓取内部的点index转化为集合
                    #求与当前簇中点索引集合的并集
                    points_in_cluster = points_in_cluster.union(points_in_grasp)
                clusters_sets.append(points_in_cluster)#所有簇对应的内部点索引集合


            raw_cluster_num = len(clusters_sets)
            #self.show_clusters(pc,clusters_sets)

            #检查iou
            clusters_sets = self.iou_check(pc,clusters_sets)
            #self.show_clusters(pc,clusters_sets)
            clusters_num = len(clusters_sets)
            #制作mask
            mask = np.zeros((pc.shape[0],10)).astype(int)

            for i in range(10):
                if i<clusters_num:
                    mask[list(clusters_sets[i]),i]=1

            #self.show_mask(pc,mask)

            #存起来
            data['clusters'] =clusters_sets
            #self.show_clusters(pc,clusters_sets)
            data['grasp_mask'] =mask

            #保存
            np.save(grasp_file,data)

            
            print('Generate scene {} mask done, from {} to {}'.format(scene_index, raw_cluster_num, clusters_num))



    def iou_check(self,pc,clusters_sets):
        ''' 根据不同聚类簇之间的重叠率，来对原始的cluster进行融合、删除等操作
        '''
        raw_cluster_num = len(clusters_sets)
        clusters_index = np.array(list(range(raw_cluster_num)))
        clusters_sets = np.array(clusters_sets)  #将列表转换为np.array，元素都是集合

        #新的
        new_clusters_sets =[]


        #融合聚类簇，进一步调整簇的范围，将大规模重叠的地方整合起来，主要是球体的地方会大规模重叠
        ref_index = 0 #参考簇index
        new_clusters_all=set()
        delete_set_index = [] #删除集合index
        while ref_index < len(clusters_sets)-1:
            if ref_index in delete_set_index:#如果当前簇已经被删除了
                ref_index+=1
                continue

            ref_set = copy.deepcopy(clusters_sets[ref_index]) #取出当前参考簇

            #new_clusters_sets.append(ref_set)#默认将参考簇加入结果中

            next_ref_index = ref_index+1 #下一个参考簇index
            #从参考簇下一个开始与参考簇对比,不断修改ref_set中的值
            for target_set_index in range(next_ref_index,len(clusters_sets)):
                if target_set_index in delete_set_index:
                    continue

                target_set = clusters_sets[target_set_index]#取出目标集
                inter_set = ref_set&target_set#求 "参考集-目标集"之间的交集

                iou_t =  len(inter_set)/len(target_set)#交集 占 目标集的比例
                iou_r = len(inter_set)/len(ref_set)#交集 占 参考集的比例

                if iou_t>0.8 or iou_r>0.8:#交集占两者比例都大于80% 删除目标集
                    delete_set_index.append(target_set_index)
                    #求参考簇和目标簇的并集
                    ref_set = ref_set | target_set

            #最后还要跟之前保存的所有的集合的并集比较一下
            inter_set = new_clusters_all&ref_set
            iou_r = len(inter_set)/len(ref_set)
            if iou_r >0.8:
                pass #说明以前的集合里面基本上是有了，不需要了
            else:
                new_clusters_sets.append(ref_set)
                #更新
                new_clusters_all = new_clusters_all | ref_set

            #clusters_sets[ref_index] = u_set
            ref_index+=1#交集占目标集与参考集比例都不超过80%，不删除集合
                
            #在参考簇对比完所有的簇之后，删除list中的目标簇
            #clusters_sets = np.delete(clusters_sets,delete_set_index)
            #self.show_clusters(pc,[clusters_sets[ref_index-1]])
            #clusters_index = np.delete(clusters_index,delete_set_index)
        
        return new_clusters_sets


    def iou_check_(self,clusters_sets):
        ''' 根据不同聚类簇之间的重叠率，来对原始的cluster进行融合、删除等操作
        '''
        raw_cluster_num = len(clusters_sets)
        clusters_index = np.array(list(range(raw_cluster_num)))
        clusters_sets = np.array(clusters_sets)

        #融合聚类簇，进一步调整簇的范围，将大规模重叠的地方整合起来，主要是球体的地方会大规模重叠
        ref_index = 0 #参考簇index
        while ref_index < len(clusters_sets)-1:
            delete_set_index = [] #删除集合index
            ref_set = clusters_sets[ref_index] #取出当前参考簇

            next_ref_index = ref_index+1 #下一个参考簇index
            current_ref_index = ref_index
            #从参考簇下一个开始与参考簇对比
            for target_set_index in range(next_ref_index,len(clusters_sets)):
                target_set = clusters_sets[target_set_index]#取出目标集
                inter_set = ref_set.intersection(target_set)#求 "参考集-目标集"之间的交集

                iou_t =  len(inter_set)/len(target_set)#交集 占 目标集的比例
                iou_r = len(inter_set)/len(ref_set)#交集 占 参考集的比例

                if iou_t>0.8 or iou_r>0.8:#交集占两者比例都大于80% 删除目标集
                    if target_set_index in delete_set_index:
                        pass
                    else:
                        delete_set_index.append(target_set_index)
                        #求参考簇和目标簇的并集
                        clusters_sets[ref_index] = clusters_sets[ref_index] | clusters_sets[target_set_index]
                        
                        ref_index=next_ref_index#移动参考index到next
                    '''
                    elif iou_t<0.8 and iou_r>0.8:#占参考簇大于0.8占目标簇小于0.8
                        if current_ref_index in delete_set_index:
                            pass
                        else:
                            delete_set_index.append(current_ref_index)#交集占参考集比例大于80% 删除参考集
                    elif iou_t>0.8 and iou_r<0.8:
                        delete_set_index.append(target_set_index)#交集占目标集比例大于80% 删除目标集
                        ref_index=next_ref_index
                    '''
                else:
                    ref_index=next_ref_index#交集占目标集与参考集比例都不超过80%，不删除集合
                
            #在参考簇对比完所有的簇之后，删除list中的目标簇
            clusters_sets = np.delete(clusters_sets,delete_set_index)
            clusters_index = np.delete(clusters_index,delete_set_index)
        return clusters_sets


    def posesTransWithinErrorBoundsCuda(self,index_i,index_j,centers_diff,theta_diff):
        '''计算两个抓取中心、pose之间的偏差是否超过阈值
        '''

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

    def do_job(self,scene_index):
        grasp_file =self.grasp_file_paths[scene_index]
        #多线程聚类
        data = np.load(grasp_file, allow_pickle=True).item()

        grasp_dis_diff = data['grasp_dis_diff']
        grasp_pose_diff = data['grasp_pose_diff']
        grasp_score = data['grasp_score']

        grasp_clusters = []

        for j in range(len(grasp_score)):#为总编号为j的抓取
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
                    ptWithinError = self.posesTransWithinErrorBoundsCuda(j,grasp_index,grasp_dis_diff,grasp_pose_diff)
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
        grasp_clusters.sort(key=self.takeSecond,reverse = True)
        #截取排名前30的簇（如果有的话）
        #if len(grasp_clusters)>30:
        #    grasp_clusters = grasp_clusters[:30]

        #保存group with score
        data['grasp_clusters']=grasp_clusters
        np.save(grasp_file,data)

        #当前场景聚类完毕    
        print('Scene  {}'.format(scene_index)+ '  legal grasp:{}'.format(len(data['grasp_centers']))+' done')


    # 获取列表的第二个元素
    def takeSecond(self,elem):
        return elem[1]


    def get_gripper_neighor_points_cuda(self,data):
        """对CTG抓取姿态和Cpc进行碰撞检测(使用显卡加速计算)
        返回符合要求的，场景点云点的索引，每个抓取都有自己的内部点索引，是一个二维list
        """

        grasp_poses = data['grasp_poses']#抓取旋转矩阵
        bottom_centers = data['bottom_centers']
        local_hand_points_extend = data['local_hand_points_extend']
        if self.dense_pc:
            pc = data['dense_pc']
        else:
            pc = data['pc']


        #mask =np.zeros(centers.shape[0])
        poses_cuda=torch.from_numpy(grasp_poses).cuda()
        hand_points= torch.from_numpy(local_hand_points_extend).cuda()
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
                raise ValueError("Grasp has no neighbor points!")
                #print("Grasp has no neighbor points!")
            else:
                grasps_neighbor_points_index.append(points_index.cpu().numpy())
        
        return grasps_neighbor_points_index

    def show_mask(self,pc,mask=[],hand_points=[],name='Scene Mask', scale_factor=.004):
        '''显示生成结果
        '''
        m=mask.shape[1] #cluster数量
        color_ = [(0, 0, 1),(1, 0, 0),(0, 1, 0),(0.22, 1, 1),(0.8, 0.8, 0.2),
                            (0.5, 0, 1),(1, 0.62, 0),(0.53, 1, 0),(0.22, 0.63, 1),(0.8, 0.22, 0.2)]

        #画出点云
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81),size=(1800,1800))
        mlab.points3d(pc[:, 0], pc[:, 1],
                pc[:, 2], color=(0.5,0.5,0.5), scale_factor=scale_factor)

        #
        if m==0:
            mlab.show()
            
        for i in range(m):
            index = np.where(mask[:,i]==1)[0]
            mlab.points3d(pc[index][:, 0], pc[index][:, 1],pc[index][:, 2], \
                color=color_[i], scale_factor=scale_factor)
    
        if len(hand_points)==0:
            mlab.show()
        else:
            index = list(range(hand_points.shape[0]))
            random.shuffle(index)
            max_n=500
            if len(index)<max_n:
                max_n=len(index)
            index = index[:max_n]

            for i in index:
                self.show_grasp_3d(hand_points[i])
            mlab.show()


    def show_clusters(self,pc, clusters_sets=[],hand_points=[],name='Scene Mask', scale_factor=.004):
        '''显示生成结果
        '''
        m=len(clusters_sets) #cluster数量
        color_ = [(0, 0, 1),(1, 0, 0),(0, 1, 0),(0.22, 1, 1),(0.8, 0.8, 0.2),
                            (0.5, 0, 1),(1, 0.62, 0),(0.53, 1, 0),(0.22, 0.63, 1),(0.8, 0.22, 0.2)]

        #画出点云
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81),size=(1800,1800))
        mlab.points3d(pc[:, 0], pc[:, 1],
                pc[:, 2], color=(0.5,0.5,0.5), scale_factor=scale_factor)

        #
        if m==0:
            mlab.show()
        else:
            if m>10:#仅仅显示排名前10的聚类结果
                m=10
                
            #m-=1
            #n=m-1
            for i in range(m):
                j=i #反序显示，将最大的簇最后显示
                mlab.points3d(pc[list(clusters_sets[j])][:, 0], pc[list(clusters_sets[j])][:, 1],pc[list(clusters_sets[j])][:, 2], color=color_[j], scale_factor=scale_factor)
        
        if len(hand_points)==0:
            mlab.show()
        else:
            index = list(range(hand_points.shape[0]))
            random.shuffle(index)
            max_n=500
            if len(index)<max_n:
                max_n=len(index)
            index = index[:max_n]

            for i in index:
                self.show_grasp_3d(hand_points[i])
            mlab.show()


    def show_grasp_3d(self, hand_points, color=(0.003, 0.50196, 0.50196)):
        # for i in range(1, 21):
        #     self.show_clusters(p[i])
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                        (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                        (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                        (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                        (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                                triangles, color=color, opacity=0.5)







if __name__ == '__main__':

    do_job = Mask_generate(
        legal_grasp_file_paths,
        parameters,
        graspSampler,
        show_result=True,
        dense_pc=False)





    print('All job done')





                





    


