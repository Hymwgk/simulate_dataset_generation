import numpy as np
import sys
import multiprocessing
from autolab_core import YamlConfig
import argparse
import os
from mayavi import mlab
from tqdm import tqdm
import time
#解析命令行参数
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gripper', type=str, default='baxter')
args = parser.parse_args()

a =200
max_digits = 5
print(str(a).zfill(5))

a = np.arange(90).reshape(-1,3,3)
b = np.arange(90).reshape(-1,3,3)
print(a[0])
print(b[0])
print(a[0].dot(b[0]))
print("============================")

c =np.matmul(a,b)
d =np.c_[c,np.array([0,0,1]).reshape(-1,3,1).repeat(c.shape[0],axis=0)]
e = np.array([0,0,0,1]).reshape(1,1,4).repeat(c.shape[0],axis=0)
d=np.concatenate((d,e),axis = 1)
print(c[0])
print("============================")


a= np.arange(36).reshape(-1,3,3) #(-1,3,3)
print(a)
print("#")
b = a[:,:,0] #(-1,3)
b[:,2]=0#(-1,3)
print(b)
print("#")
print(a)     
print("#")
#minus_mask = (a[:,:,0][:,2]==0)
#print(minus_mask)

print("============================")
"""使用索引和fancy indexing对numpy进行切片索引，是否会返回一个新对象？
      参考 https://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
"""
arr = np.arange(10)
slice = arr[2:5]  #对arr进行切片，并不会返回新对象，而是返回一个reference/view 指向该部分的内容
slice[:] = 12     #使用该reference直接修改内容，会直接导致原arr对象被修改
print(arr)
slice[:]= slice[:]/2    #使用符号运算
print(slice)
print(arr)
slice = slice / 2
print(slice)
print(arr)
print("#")



index_b =[1,2,3] 
#在等号右侧使用fancy indexing，
slice_b = arr[index_b]  #返回的不是一个view，而是copy对应部分数据的新对象
print(slice_b)
slice_b[0]=100
print(slice_b)
print(arr)
print('#')
#在等号左侧直接使用fancy indexing, 并不会创建view也不会创建copy，就相当于直接操作
#因为没有必要做这些
arr[index_b] = 100
print(arr)
print('#')
#
arr[index_b][1] = 100
print(arr)
print('#')

#在右侧的简单索引
arr = np.arange(9).reshape(3,3)
slice_c = arr[0]  #退化(-1,)   返回view
slice_c[0]=50
print(arr)
print('#')

arr = np.arange(9)
slice_c = arr[0]  #退化(-1,)   返回的还是一个view，但是因为此时 arr[0]退化为一个np.int64的数字
#slice_c[0]=50  #是无法使用这种访问的形式来直接修改arr的值的
slice_c=50  #这样子实际上是把slice_c  分配到了一个新的python  int 类型对象上，并不会修改arr值
print(arr)
"""总结：
1.在右侧的简单索引或者切片返回的是view
2.在右侧的fancy indexing 返回的是copy
3.在左侧的简单索引切片或者fancy indexing，都相当于直接操作原数组，不创建view或者copy
"""



print("============================")

a = np.arange(2,7,2)
print(a)
b=np.arange(20)

b=list(set(b).difference(set(a)))
print(b)

print("============================")




def do_job(job_id):      #处理函数  处理index=i的模型
    grasps_with_score_ = range(5)
    print("Worker {} got {} grasps.".format(job_id,len(grasps_with_score_)))
    #虽然grasps_with_score是全局变量，但是已经在子线程中拷贝了一份，
    # 这里修改的是子进程中的数据，并不会影响主线程中的grasps_with_score
    global grasps_with_score
    #添加到外部的采集库中
    grasps_with_score+=grasps_with_score_
    print("Now we have {} grasps in total".format(len(grasps_with_score)))


def do_job1(job_id):      #测试多线程进度条
    #trange(i)是tqdm(range(i))的一种简单写法
    for i in tqdm(range(100), desc='Processing'+str(job_id)):
        time.sleep(0.5)

def do_job2(id):
    '''测试多线程下的显卡调用
    '''
    

from dexnet.grasping import RobotGripper
from dexnet.grasping import GpgGraspSampler  

if __name__ == '__main__':
    home_dir = os.environ['HOME']
    manager = multiprocessing.Manager()
    yaml_config = YamlConfig(home_dir + "/code/dex-net/test/config.yaml")
    gripper = RobotGripper.load(args.gripper, home_dir + "/code/dex-net/data/grippers")
    ags = GpgGraspSampler(gripper, yaml_config)

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
    grasp_bottom_center = np.array([0, 0, 0])
    approach_normal = np.array([1, 0, 0])
    binormal = np.array([0, 1, 0])
    minor_pc = np.array([0,0,1])
    center_point=grasp_bottom_center+ags.gripper.hand_depth * approach_normal
    ps = ags.get_hand_points(grasp_bottom_center, approach_normal, binormal)

    ags.show_grasp_3d(ps)
    ags.show_points(ps,scale_factor=0.005)
    for i,p in enumerate(ps):
        mlab.text3d(p[0],p[1],p[2],str(i),scale = (0.005),color=(0,0,1))
    

    #grasp_bottom_center = -ags.gripper.hand_depth * np.array([1,0,0]) + np.array([0,0,0])
    '''
    mlab.text3d(0,-0.008,-0.01,'bottom_center',scale = (0.002),color=(0,0,1))
    mlab.text3d(0.03,0,0,'approach_normal',scale = (0.004),color=(1,0,0))
    mlab.text3d(0,0.03,0,'binormal',scale = (0.004),color=(0,1,0))
    mlab.text3d(0,0,0.03,'minor_pc',scale = (0.004),color=(0,0,1))
    '''

    ags.show_grasp_norm_oneside(center_point,approach_normal,binormal,minor_pc,scale_factor=0.001)

    
    mlab.show()

    grasps_with_score=manager.list()
    pool =[]

    for i in range(10):
        pool.append(multiprocessing.Process(target=do_job1, args=(i,)))
    #启动多线程
    [p.start() for p in pool]                  
    #等待所有进程结束，返回主进程
    [p.join() for p in pool]              
    #grasps_with_score  = [x for x in grasps_with_score]
    #print(len(grasps_with_score))    

    #print(type(grasps_with_score))

