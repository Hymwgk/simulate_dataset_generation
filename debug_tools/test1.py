from multiprocessing import Pool, cpu_count
import os, time, random
from tqdm import tqdm
import glob


class MyMultiprocess(object):
    def __init__(self, process_num):
        self.pool = Pool(processes=process_num)

    def work(self, func, args):
        for arg in args:
            self.pool.apply_async(func, (arg,))
        self.pool.close()
        self.pool.join()


def func(num):
    name = num
    for i in tqdm(range(5), ncols=80, desc='执行任务' + str(name) + ' pid:' + str(os.getpid())):
        # time.sleep(random.random() * 3)
        time.sleep(1)

def takeSceneIndex(elem):
    index = elem.split('/')[-2]
    return int(index)

if __name__ == "__main__":

    home_dir = os.environ['HOME']
    #场景文件夹
    scenes_dir = home_dir+"/dataset/simulate_grasp_dataset/baxter/scenes/"
    scenes_grasp_file_paths = glob.glob(scenes_dir+'*/legal_grasps_with_score.npy')
    print(type(scenes_grasp_file_paths))
    scenes_grasp_file_paths.sort()

    raw_pc_paths = glob.glob(scenes_dir+'*/raw_pc.npy')



    print('父进程 %s.' % os.getpid())
    mymultiprocess = MyMultiprocess(cpu_count())
    start = time.time()
    mymultiprocess.work(func=func, args=range(10))
    end = time.time()
    print("\n应用多进程耗时: %0.2f seconds" % (end - start))

    start = time.time()
    for i in range(10):
        func(i)
    end = time.time()
    print("\n不用多进程耗时: %0.2f seconds" % (end - start))