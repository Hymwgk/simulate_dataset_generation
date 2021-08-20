# -*- coding: utf-8 -*-
#实现多线程下载YCB数据集，使用python2
import os
import sys
import json
import urllib
import urllib2
import multiprocessing


# You can either set this to "all" or a list of the objects that you'd like to
# download.
#objects_to_download = "all"
#objects_to_download = ["030_fork", "031_spoon","032_knife","062_dice","063-b_marbles","070-b_colored_wood_blocks","072-d_toy_airplane","072-e_toy_airplane","076_timer"]


objects_to_download =['001_chips_can', '002_master_chef_can', 
'003_cracker_box', '004_sugar_box', '005_tomato_soup_can', 
'006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', 
'009_gelatin_box', '010_potted_meat_can', '011_banana', '012_strawberry', 
'013_apple', '014_lemon', '015_peach', '016_pear', '017_orange', '018_plum', 
'019_pitcher_base', '021_bleach_cleanser', '022_windex_bottle', '023_wine_glass', 
'024_bowl', '025_mug', '026_sponge', '027-skillet', '028_skillet_lid', '029_plate', '030_fork', 
'031_spoon', '032_knife', '033_spatula', '035_power_drill', '036_wood_block', '037_scissors', 
'038_padlock', '039_key', '040_large_marker', '041_small_marker', '042_adjustable_wrench', 
'043_phillips_screwdriver', '044_flat_screwdriver', '046_plastic_bolt', '047_plastic_nut', '048_hammer', 
'049_small_clamp', '050_medium_clamp', '051_large_clamp', '052_extra_large_clamp', '053_mini_soccer_ball', 
'054_softball', '055_baseball', '056_tennis_ball', '057_racquetball', '058_golf_ball', '059_chain', '061_foam_brick', 
'062_dice', '063-a_marbles', '063-b_marbles', '063-c_marbles', '063-d_marbles', '063-e_marbles', '063-f_marbles', 
'065-a_cups', '065-b_cups', '065-c_cups', '065-d_cups', '065-e_cups', '065-f_cups', '065-g_cups', '065-h_cups', 
'065-i_cups', '065-j_cups', '070-a_colored_wood_blocks', '070-b_colored_wood_blocks', '071_nine_hole_peg_test', 
'072-a_toy_airplane', '072-b_toy_airplane', '072-c_toy_airplane', '072-d_toy_airplane', '072-e_toy_airplane', 
'072-f_toy_airplane', '072-g_toy_airplane', '072-h_toy_airplane', '072-i_toy_airplane', '072-j_toy_airplane', 
'072-k_toy_airplane', '073-a_lego_duplo', '073-b_lego_duplo', '073-c_lego_duplo', '073-d_lego_duplo', 
'073-e_lego_duplo', '073-f_lego_duplo', '073-g_lego_duplo', '073-h_lego_duplo', '073-i_lego_duplo',
'073-j_lego_duplo', '073-k_lego_duplo', '073-l_lego_duplo', '073-m_lego_duplo', '076_timer', '077_rubiks_cube']


# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
#files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]
#files_to_download = ["berkeley_processed", "berkeley_rgbd"]
#修改分辨率
files_to_download = ["google_16k", "google_512k"]
#每种放一个单独文件夹
output_directory = ["./"+type  for type in files_to_download ]


# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True 

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

for _ in output_directory:
    if not os.path.exists(_):
        os.makedirs(_)

def fetch_objects(url):
    response = urllib2.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s (%s MB)" % (filename, file_size/1000000.0)

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    f.close()

def tgz_url(object, type):
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url,object):
    try:
        request = urllib2.Request(url)
        request.get_method = lambda : 'HEAD'
        response = urllib2.urlopen(request)
        return True
    except Exception as e:
        print("{} url check failed".format(object))
        return False


def do_job(object_i,files_to_download):
    for index, file_type in enumerate(files_to_download):
        url = tgz_url(object_i, file_type)
        if not check_url(url,object_i):
            continue
        filename = "{path}/{object}_{file_type}.tgz".format(path=output_directory[index],
                                                                    object=object_i,
                                                                    file_type=file_type)
        download_file(url, filename)
        if extract:
            extract_tgz(filename, output_directory[index])



if __name__ == "__main__":

    objects = objects_to_download#fetch_objects(objects_url)

    #开启多个线程去处理数据
    pool_size=10  #同时下载10个文件
    if pool_size>len(objects_to_download):
        pool_size = len(objects_to_download)
    count = 0
    pool = []
    for i in range(pool_size):  
        count = count+1
        pool.append(multiprocessing.Process(target=do_job,args=(objects[i],files_to_download,)))
    [p.start() for p in pool]  #启动多线程

    while count<len(objects_to_download):    #如果有些没处理完
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                p = multiprocessing.Process(target=do_job, args=(objects[count],files_to_download,))
                count=count+1
                p.start()
                pool.append(p)
                break
    print('All job done.')




