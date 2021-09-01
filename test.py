import numpy as np
import sys
import multiprocessing

a =200
max_digits = 5
print(str(a).zfill(5))

a = np.arange(90).reshape(-1,3,3)
b = np.arange(90).reshape(-1,3,3)
print(a[0])
print(b[0])
print(a[0].dot(b[0]))

c =np.matmul(a,b)
d =np.c_[c,np.array([0,0,1]).reshape(-1,3,1).repeat(c.shape[0],axis=0)]
e = np.array([0,0,0,1]).reshape(1,1,4).repeat(c.shape[0],axis=0)
d=np.concatenate((d,e),axis = 1)
print(c[0])


def do_job(job_id):      #处理函数  处理index=i的模型
    grasps_with_score_ = range(5)
    print("Worker {} got {} grasps.".format(job_id,len(grasps_with_score_)))
    #虽然grasps_with_score是全局变量，但是已经在子线程中拷贝了一份，
    # 这里修改的是子进程中的数据，并不会影响主线程中的grasps_with_score
    global grasps_with_score
    #添加到外部的采集库中
    grasps_with_score+=grasps_with_score_
    print("Now we have {} grasps in total".format(len(grasps_with_score)))


if __name__ == '__main__':

    manager = multiprocessing.Manager()

    grasps_with_score=manager.list()
    pool =[]

    for i in range(10):
        pool.append(multiprocessing.Process(target=do_job, args=(i,)))
    #启动多线程
    [p.start() for p in pool]                  
    #等待所有进程结束，返回主进程
    [p.join() for p in pool]              
    grasps_with_score  = [x for x in grasps_with_score]
    print(len(grasps_with_score))    

    print(type(grasps_with_score))

