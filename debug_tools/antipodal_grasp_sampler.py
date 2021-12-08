import numpy as np
import sys
import pickle
#from dexnet.grasping.quality import PointGraspMetrics3D
#from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler, grasp
#from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
#import dexnet
#from autolab_core import YamlConfig
#from meshpy.obj_file import ObjFile
#from meshpy.sdf_file import SdfFile
import os
import torch   
import math
import multiprocessing
#import matplotlib.pyplot as plt
import open3d as o3d
#from open3d import geometry
#from open3d.geometry import voxel_down_sample, sample_points_uniformly, orient_normals_towards_camera_location, estimate_normals


class AntipodalSampler:
    def __init__(self,ply_path,friction_coef):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.friction_coef  = friction_coef
        self.friction_cos_theta = torch.tensor(math.cos(math.atan(friction_coef)),dtype=torch.float64).cuda()




        #获取原始点云
        mesh = o3d.io.read_triangle_mesh(ply_path)
        #得到点云对象
        raw_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 10)
        #均匀降采样
        voxel_pc = o3d.geometry.PointCloud.voxel_down_sample(raw_pc, 0.0025)
        #估计点云的表面法向量
        o3d.geometry.PointCloud.estimate_normals(voxel_pc, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        #将点云转换为np对象
        self.pc = np.asarray(voxel_pc.points)
        self.normals = np.asarray(voxel_pc.normals)




    def farthest_point_sample(self,xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [N, C]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        N, C = xyz.shape
        centroids = torch.zeros(npoint, dtype=torch.long).to(device)
        distance = torch.ones(N,dtype=torch.float64).to(device) * 1e10
        farthest = torch.randint(0, N,(1,),dtype=torch.long).to(device)

        for i in range(npoint):
            # 更新第i个最远点
            centroids[i] = farthest
            # 取出这个最远点的xyz坐标
            centroid = xyz[farthest, :].view(1, 3)
            # 计算点集中的所有点到这个最远点的欧式距离
            dist = torch.sum((xyz - centroid) ** 2, -1)
            # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
            mask = dist < distance
            distance[mask] = dist[mask]
            # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
            farthest = torch.max(distance, -1)[1]
        return centroids

    def computeTangents(self,normals,max_samples=1000): 
        """计算每个输入点的切线空间坐标系，类似于Darboux frame，
        切线坐标系的z轴是表面法向量，y轴和x轴位于与点相切的平面内部
        """
        N,C = normals.shape
        inner_normals = -normals#[n,3]
        
        U,_,_ = torch.svd(inner_normals.view(-1,3,1),some=False)
        print(U.shape)
        x,y = U[:,:, 1], U[:,:, 2]# 退化[n,3]
        # make sure t1 and t2 obey right hand rule
        z_hat = torch.cross(x, y) #[n,3]
        mask  = torch.sum(z_hat.mul(inner_normals),dim=1)<0
        y[mask]=-y[mask]

        max_ip = torch.zeros([N,],dtype=torch.float64).cuda()
        max_theta = torch.zeros([N,],dtype=torch.float64).cuda()
        theta = torch.zeros([1,],dtype=torch.float64).cuda()
        target = torch.tensor([1, 0, 0]).unsqueeze(0).repeat(N,1).cuda()#[n,3]

        d_theta = torch.tensor([2 * math.pi / float(max_samples)]).cuda()#将2pi分解为1000份

        for i in range(max_samples):
            v = torch.cos(theta) * x + torch.sin(theta) * y#[n,3]
            mul = torch.sum(v.mul(target),dim=1)#[n,]  矩阵每行点乘值

            mask =mul>max_ip#[n,]

            max_ip[mask] = mul[mask]#更新
            max_theta[mask] = theta #
            theta += d_theta

        #新x轴 与 世界x轴配准
        v = torch.cos(max_theta).view(N,1) * x + torch.sin(max_theta).view(N,1) * y #[n,3]
        #叉乘出新的y轴
        w = torch.cross(inner_normals, v)

        return inner_normals, v, w


    def computeAxis(self,points,inner_noramls):
        #问题：如何处理夹爪对称的情况呢？
        N,C = points.shape #
        c1_points = points.clone()#[N,3]
        c2_points = points.clone()
        c1_to_c2 = c2_points.view(N,1,C) - c1_points  #[N,N,C]
        norm = torch.norm(c1_to_c2,dim=2,keepdim=True)
        #变成单位向量
        c1_to_c2_norm = c1_to_c2/torch.norm(c1_to_c2,dim=2,keepdim=True)

        #计算每个点的射线和inner_noramls之间的点乘，就是cos夹角
        angles = torch.sum(c1_to_c2_norm.mul(inner_noramls.view(N,1,3)),dim=2) #[N,N]

        mask = angles>self.friction_cos_theta  #找到位于
        pass










        
       
        



    def sample_grasps(self,):
        pc_cuda = torch.from_numpy(self.pc).cuda()#FIXME:这里的数据发生了截断
        normal_cuda = torch.from_numpy(self.normals).cuda()
        #抽取300个第一接触点
        c1_index = self.farthest_point_sample(pc_cuda,300)
        c1_points = pc_cuda[c1_index]
        c1_normals = normal_cuda[c1_index]
        self.inner_normals,self.tx,self.ty = self.computeTangents(c1_normals)
        self.computeAxis(c1_points,self.inner_normals)





        return True


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    file = home_dir+"/dataset/simulate_grasp_dataset/ycb/google_512k/002_master_chef_can_google_512k/002_master_chef_can/google_512k/nontextured.ply"
    #sf = SdfFile(file)
    #sdf = sf.read()
    #ply  = o3d.io.read_point_cloud(file)
    test = AntipodalSampler(file,2.0)
    test.sample_grasps()
