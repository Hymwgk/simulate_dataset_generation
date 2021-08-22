#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 30/05/2018 9:57 AM 
# File Name  : read_grasps_from_file.py
# 使用python3
import os
import sys
import re
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import GraspableObject3D
import numpy as np
from dexnet.visualization.visualizer3d import DexNetVisualizer3D as Vis
from dexnet.grasping import RobotGripper
from autolab_core import YamlConfig
from mayavi import mlab
from dexnet.grasping import GpgGraspSampler  # temporary way for show 3D gripper using mayavi
import glob

# global configurations:
home_dir = os.environ["HOME"] + "/code/MyGPD"
yaml_config = YamlConfig(home_dir + "/dex-net/test/config.yaml")
gripper_name = "robotiq_85"
gripper = RobotGripper.load(gripper_name, home_dir + "/dex-net/data/grippers")
ags = GpgGraspSampler(gripper, yaml_config)

save_fig = False  # save fig as png file
show_fig = True  # show the mayavi figure
generate_new_file = False  # whether generate new file for collision free grasps
check_pcd_grasp_points = False


def open_npy_and_obj(name_to_open_):
    """从路径指定的npy文件中读取并返回
    npy_m_:         抓取序列（只是针对某单个物体的抓取序列）
    obj_:                 待抓取物体的模型mesh（单个模型）
    ply_name_:  待抓取物体的路径
    object_name_:  待抓取物体的名称
    """
    npy_m_ = np.load(name_to_open_)
    #存放mesh模型的文件夹路径
    file_dir = home_dir + "/PointNetGPD/data/ycb_meshes_google/"
    #获取当前模型名称
    object_name_ = name_to_open_.split("/")[-1][:-4]
    print(object_name_)

    ply_name_ = file_dir + object_name_ + "/google_512k/nontextured.ply"
    if not check_pcd_grasp_points:
        #读取相关的obj以及sdf模型文件
        of = ObjFile(file_dir + object_name_ + "/google_512k/nontextured.obj")
        sf = SdfFile(file_dir + object_name_ + "/google_512k/nontextured.sdf")
        mesh = of.read()
        sdf = sf.read()
        #dex-net格式的可抓取对象
        obj_ = GraspableObject3D(sdf, mesh)
    else:
        cloud_path = home_dir + "/dataset/ycb_rgbd/" + object_name_ + "/clouds/"
        pcd_files = glob.glob(cloud_path + "*.pcd")
        obj_ = pcd_files
        obj_.sort()
    return npy_m_, obj_, ply_name_, object_name_


def display_object(obj_):
    """display object only using mayavi"""
    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.mesh(obj_.mesh.trimesh, color=(0.5, 0.5, 0.5), style="surface")
    Vis.show()


def display_gripper_on_object(obj_, grasp_):
    """display both object and gripper using mayavi"""
    # transfer wrong was fixed by the previews comment of meshpy modification.
    # gripper_name = "robotiq_85"
    # gripper = RobotGripper.load(gripper_name, home_dir + "/dex-net/data/grippers")
    # stable_pose = self.dataset.stable_pose(object.key, "pose_1")
    # T_obj_world = RigidTransform(from_frame="obj", to_frame="world")
    t_obj_gripper = grasp_.gripper_pose(gripper)

    stable_pose = t_obj_gripper
    grasp_ = grasp_.perpendicular_table(stable_pose)

    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.gripper_on_object(gripper, grasp_, obj_,
                          gripper_color=(0.25, 0.25, 0.25),
                          # stable_pose=stable_pose,  # .T_obj_world,
                          plot_table=False)
    Vis.show()


def display_grasps(grasp, graspable, color):
    """
    显示给定的抓取，做一下碰撞检测，和指定颜色
    grasp：单个抓取
    graspable：目标物体（仅做碰撞）
    color：夹爪的颜色
    """
    center_point = grasp[0:3]    #夹爪中心(指尖中心)
    major_pc = grasp[3:6]  # binormal
    width = grasp[6]
    angle = grasp[7]
    level_score, refine_score = grasp[-2:]
    # cal approach
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]

    axis_y = major_pc
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])

    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)

    axis_z = np.cross(axis_x, axis_y)
    
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    approach_normal = R2.dot(R1)[:, 0]
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    minor_pc = np.cross(major_pc, approach_normal)

    #计算夹爪bottom_center
    grasp_bottom_center = -ags.gripper.hand_depth * approach_normal + center_point
    #以世界坐标系为参考系时
    hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
    #固定夹爪作为参考系时
    local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    #检测一下夹爪与目标物体的碰撞（这里使用原始函数，不检测与桌面的碰撞）
    if_collide = ags.check_collide(grasp_bottom_center, approach_normal,
                                   major_pc, minor_pc, graspable, local_hand_points)

    #只显示夹爪，不显示点云
    if not if_collide and (show_fig or save_fig):
        ags.show_grasp_3d(hand_points, color=color)
        return True
    elif not if_collide:
        return True
    else:
        return False


def show_selected_grasps_with_color(m, ply_name_, title, obj_):
    """显示选择出的抓取，将抓取序列中good的抓取和bad的抓取区分开来，并分别显示在两个窗口中
    m：                    针对被抓物体的带有打分的grasp序列
    ply_name_ ： 被抓物体的mesh文件路径
    title：被抓物体的名称
    obj_：被抓物体的mesh文件
    """
    #筛选出优质抓取
    m_good = m[m[:, -2] <= 0.4]
    #从优质的抓取中随机抽取出25个（如果都要的话，就太多了，不易于显示）
    m_good = m_good[np.random.choice(len(m_good), size=25, replace=True)]

    #筛选出不合格抓取
    m_bad = m[m[:, -2] >= 1.8]
    #抽选显示
    m_bad = m_bad[np.random.choice(len(m_bad), size=25, replace=True)]

    collision_grasp_num = 0
    if save_fig or show_fig:
        # fig 1: good grasps
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
        #导入目标物体的mesh模型
        mlab.pipeline.surface(mlab.pipeline.open(ply_name_))
        
        for a in m_good:
            # display_gripper_on_object(obj, a)  # real gripper
            collision_free = display_grasps(a, obj_, color="d")  # simulated gripper
            if not collision_free:
                collision_grasp_num += 1

        if save_fig:
            mlab.savefig("good_"+title+".png")
            mlab.close()
        elif show_fig:
            mlab.title(title, size=0.5)

        # fig 2: bad grasps
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
        mlab.pipeline.surface(mlab.pipeline.open(ply_name_))

        for a in m_bad:
            # display_gripper_on_object(obj, a)  # real gripper
            collision_free = display_grasps(a, obj_, color=(1, 0, 0))
            if not collision_free:
                collision_grasp_num += 1

        if save_fig:
            mlab.savefig("bad_"+title+".png")
            mlab.close()
        elif show_fig:
            mlab.title(title, size=0.5)
            mlab.show()
    elif generate_new_file:
        # only to calculate collision:
        collision_grasp_num = 0
        ind_good_grasp_ = []
        for i_ in range(len(m)):
            collision_free = display_grasps(m[i_][0], obj_, color=(1, 0, 0))
            if not collision_free:
                collision_grasp_num += 1
            else:
                ind_good_grasp_.append(i_)
        collision_grasp_num = str(collision_grasp_num)
        collision_grasp_num = (4-len(collision_grasp_num))*" " + collision_grasp_num
        print("collision_grasp_num =", collision_grasp_num, "| object name:", title)
        return ind_good_grasp_


def get_grasp_points_num(m, obj_):
    """获取夹爪内部点云的数量
    """
    has_points_ = []
    ind_points_ = []
    for i_ in range(len(m)):
        grasps = m[i_][0]
        approach_normal = grasps.rotated_full_axis[:, 0]
        approach_normal = approach_normal / np.linalg.norm(approach_normal)
        major_pc = grasps.configuration[3:6]
        major_pc = major_pc / np.linalg.norm(major_pc)
        minor_pc = np.cross(approach_normal, major_pc)
        center_point = grasps.center

        #计算bottom_center
        grasp_bottom_center = -ags.gripper.hand_depth * approach_normal + center_point

        # hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        has_points_tmp, ind_points_tmp = ags.check_collision_square(grasp_bottom_center, approach_normal,
                                                                    major_pc, minor_pc, obj_, local_hand_points,
                                                                    "p_open")
        ind_points_tmp = len(ind_points_tmp)  # here we only want to know the number of in grasp points.
        has_points_.append(has_points_tmp)
        ind_points_.append(ind_points_tmp)
    return has_points_, ind_points_


if __name__ == "__main__":
    #获取路径下，所有的.npy文件路径
    npy_names = glob.glob(home_dir + "/PointNetGPD/data/ycb_grasp/train/*.npy")
    npy_names.sort()
    for i in range(len(npy_names)):
        #把夹爪姿态和打分，先从npy中读取出来
        grasps_with_score, obj, ply_name, obj_name = open_npy_and_obj(npy_names[i])
        print("load file {}".format(npy_names[i]))
        #显示抓取
        ind_good_grasp = show_selected_grasps_with_color(grasps_with_score, ply_name, obj_name, obj)

