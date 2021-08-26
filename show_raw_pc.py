# -*- coding: utf-8 -*-
#读取模拟出的虚拟点云
import  os
#复制文件
import shutil
import sys
import glob
import numpy as np
import threading
from mayavi import mlab
import tty
import termios

def readchar():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

def readkey(getchar_fn=None):
  getchar = getchar_fn or readchar
  c1 = getchar()
  if ord(c1) != 0x1b:
    return c1
  c2 = getchar()
  if ord(c2) != 0x5b:
    return c1
  c3 = getchar()
  return chr(0x10 + ord(c3) - 65)

def show_points(point, color='lb', scale_factor=.005):
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

def do_job():
    global n_pressed,quit
    while True:
        key=readkey()
        if key=='n': #下一张点云
            n_pressed=True
        if key=='q':
            quit = True  #退出程序
            print('已经按下q 键，关掉点云窗口即可完全退出')
            
            break




if __name__ == '__main__':
    if len(sys.argv) > 2:
        gripper_name = sys.argv[1]
        select = sys.argv[2] #直接选择查看某帧
    else:
        #默认panda夹爪
        gripper_name = "panda"
        #
        select = 0


    home_dir = os.environ['HOME']
    pcs_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(gripper_name)
    pcs_path_list = glob.glob(pcs_dir+"*/raw_pc.npy")

    n_pressed = False
    quit = False
    print("按下q退出查看")

    if select ==0:
        #开辟一个子线程做按键检测
        keyboardcheck = threading.Thread(target=do_job,)
        keyboardcheck.start()

        for pc_path in pcs_path_list:
            #读取
            pc_raw = np.load(pc_path)
            #剔除NAN值
            pc = pc_raw[~np.isnan(pc_raw).any(axis=1)]

            show_points(pc)
            mlab.show()

            if quit:
                print("用户中断点云查看")
                break
        keyboardcheck.join()

    else:#直接查看某帧
        if select>=len(pcs_path_list):
            raise NameError("Can not find point cloud ",select)
        show_points(pcs_path_list[select])
        mlab.show()







