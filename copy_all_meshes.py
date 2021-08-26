# -*- coding: utf-8 -*-
import  os
#复制文件
import shutil
import sys
import glob

#输入文件夹地址，返回一个列表，其中保存的是文件夹中的文件名称
def get_file_name(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        #遍历出stl文件
        for file in files:
            if file=='nontextured.stl':
                file_list.append(os.path.join(root,file))
    #排序
    file_list.sort()
    return file_list

#把单一目标文件拷贝到指定的目录下
def copy_stls(srcfile,dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exit!" % (srcfile))
    else:
        #fpath,fname=os.path.split(dstfile)
        #先找到模型的名字
        stl_name = srcfile.split('/')[-3]
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
            #将源文件拷贝到指定文件夹中，同时修改为指定的名称
        shutil.copyfile(srcfile,dstpath+stl_name+".stl")

#把单一目标文件拷贝到指定的目录下
def copy_objs(srcfile,dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exit!" % (srcfile))
    else:
        #fpath,fname=os.path.split(dstfile)
        #先找到模型的名字
        obj_name = srcfile.split('/')[-3]
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
            #将源文件拷贝到指定文件夹中，同时修改为指定的名称
        shutil.copyfile(srcfile,dstpath+obj_name+".obj")


#把单一目标文件拷贝到指定的目录下
def copy_maps(srcfile,dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exit!" % (srcfile))
    else:
        #fpath,fname=os.path.split(dstfile)
        #先找到模型的名字
        png_name = srcfile.split('/')[-3]
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
            #将源文件拷贝到指定文件夹中，同时修改为指定的名称
        shutil.copyfile(srcfile,dstpath+png_name+".png")

#把单一目标文件拷贝到指定的目录下
def copy_mtls(srcfile,dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exit!" % (srcfile))
    else:
        #fpath,fname=os.path.split(dstfile)
        #先找到模型的名字
        mtl_name = srcfile.split('/')[-3]
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
            #将源文件拷贝到指定文件夹中，同时修改为指定的名称
        shutil.copyfile(srcfile,dstpath+mtl_name+".mtl")



if __name__ == '__main__':

    home_dir = os.environ['HOME']
    #注意是16k分辨率的文件夹
    google_16k = home_dir + "/dataset/simulate_grasp_dataset/ycb/google_16k/"   
    copy_to = home_dir + "/dataset/simulate_grasp_dataset/ycb/all_16k_meshes/" 
    google_16k_stls = get_file_name(google_16k)   #返回所有stl模型所处的文件夹的路径列表
    google_16k_objs=glob.glob(google_16k+'*/google_16k/*.obj')
    google_16k_texture_maps = glob.glob(google_16k+'*/google_16k/texture_map.png')
    google_16k_texture_mtls = glob.glob(google_16k+'*/google_16k/textured.mtl')

    #复制所有16k mesh模型到指定目录，并替换为规定的名字
    for google_16k_stl in google_16k_stls:
        copy_stls(google_16k_stl,copy_to)

    for google_16k_obj in google_16k_objs:
        copy_objs(google_16k_obj,copy_to)

    for google_16k_map in google_16k_texture_maps:
        copy_maps(google_16k_map,copy_to)
    for google_16k_mtl in google_16k_texture_mtls:
        copy_mtls(google_16k_mtl,copy_to)

    

