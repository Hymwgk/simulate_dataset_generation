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
import inspect
import ctypes
import argparse
import time

#解析命令行参数
parser = argparse.ArgumentParser(description='Show raw point clouds')
parser.add_argument('--gripper', type=str, default='baxter')
parser.add_argument('--raw_pc',  action='store_true')#出现raw 就选择显示原生点云，否则显示降采样后的点云

parser.add_argument('--show', type=int, default=0)#可以选择直接显示某帧点云

args = parser.parse_args()


#用于结束键盘检测子线程
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
 
#用于结束键盘检测子线程
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


#按键检测
def readchar():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch
#按键检测
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


def show_points(point, name='raw_pc',color='lb', scale_factor=.004):
    mlab.figure(figure=name,bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81),size=(1800,1800))
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

def show_axis(scale_factor=.004):
        # un1 = grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width
        un2 = [0,0,0]
        # un3 = grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width
        # un4 = grasp_bottom_center
        # un5 = grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width
        # un6 = grasp_bottom_center
        #把当前选择的点画成一个球
        #show_points(np.array([0,0,0]), color='g', scale_factor=scale_factor * 4)
        # self.show_points(un1, scale_factor=scale_factor * 4)
        # self.show_points(un3, scale_factor=scale_factor * 4)
        # self.show_points(un5, scale_factor=scale_factor * 4)
        # self.show_line(un1, un2, color='g', scale_factor=scale_factor)  # binormal/ major pc
        # self.show_line(un3, un4, color='b', scale_factor=scale_factor)  # minor pc
        # self.show_line(un5, un6, color='r', scale_factor=scale_factor)  # approach normal
        #画箭头，起始点坐标xyz为un2[0], un2[1], un2[2]
        #终止点坐标xyz 为grasp_axis[0], grasp_axis[1], grasp_axis[2]
        mlab.quiver3d(un2[0], un2[1], un2[2], 0,1,0,
                      scale_factor=0.5, line_width=1, color=(0, 1, 0), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], 0,0,1,
                      scale_factor=0.5, line_width=1, color=(0, 0, 1), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], 1,0,0,
                      scale_factor=0.5, line_width=1, color=(1, 0, 0), mode='arrow')



#按键检测子线程
def do_job():
    global all_done,quit
    while not all_done:
        if  all_done:
            break
        #读取外部按键
        key=readkey()
        if key=='q':
            quit = True  #退出程序
            print('已经按下q 键，关掉点云窗口即可完全退出')
            break

def get_raw_pc_list(file_dir_,type='pc.npy'):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        for file in files:
            if file==type:
                file_list.append(os.path.join(root,file))
    #排序
    file_list.sort()
    return file_list



if __name__ == '__main__':

    home_dir = os.environ['HOME']
    pcs_dir = home_dir+"/dataset/simulate_grasp_dataset/{}/scenes/".format(args.gripper)
    #pcs_path_list = glob.glob(pcs_dir+"*/raw_pc.npy")
    if args.raw_pc:
        pcs_path_list = get_raw_pc_list(pcs_dir,'raw_pc.npy')  #该方法可以根据场景编号排序
        max_digits = len(pcs_path_list[0].split('/')[-2])
        selected_path = os.path.join(pcs_dir,str(args.show).zfill(max_digits),'raw_pc.npy')
    else:
        pcs_path_list = get_raw_pc_list(pcs_dir,'pc.npy')  #该方法可以根据场景编号排序
        max_digits = len(pcs_path_list[0].split('/')[-2])
        selected_path = os.path.join(pcs_dir,str(args.show).zfill(max_digits),'pc.npy')
        



    all_done = False
    quit = False
    print("按下q退出查看")
    lock = threading.Lock()

    if args.show ==0:
        #开辟一个子线程做按键检测
        keyboardcheck = threading.Thread(target=do_job,)
        keyboardcheck.start()

        for pc_path in pcs_path_list:
            #读取
            pc_raw = np.load(pc_path)
            #剔除NAN值
            pc = pc_raw[~np.isnan(pc_raw).any(axis=1)]

            show_points(pc,name='scene index '+pc_path.split('/')[-2])
            show_axis()
            print("Show:",pc_path)
            print("关闭点云窗口播放下一张")

            mlab.show()

            if quit:
                print("用户中断点云查看")
                break
        while not quit:
            time.sleep(0.8)
            print('所有点云播放完毕，按下q退出子线程')
        keyboardcheck.join()
    

    else:#直接查看某帧
        if args.show>=len(pcs_path_list):
            raise NameError("Can not find point cloud ",args.show)

        pc_raw = np.load(selected_path)
        pc = pc_raw[~np.isnan(pc_raw).any(axis=1)]
        show_points(pc,name='scene index '+selected_path.split('/')[-2])
        mlab.show()







