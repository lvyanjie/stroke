#coding:utf-8
'''
/*---author: yanjie.lv---*/
/*---用于脑卒中数据预处理以及初步病灶分割，生成mask---*/
/*---version: 20181105---*/
'''


import pydicom as dicom
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import math

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import data, img_as_float
from skimage.segmentation import morphological_chan_vese,morphological_geodesic_active_contour,inverse_gaussian_gradient,checkerboard_level_set
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes

# Load the scans in given folder path
def load_scan(path):
    '''
    加载文件序列
    :param path:扫描序列路径
    '''
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))#slice排序
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness#添加属性
        
    return slices

def get_pixels_hu(slices):
    '''
    将体素值转化为CT值
    :param slices:序列list
    '''
    image = np.stack([s.pixel_array for s in slices])#三维矩阵  zyx
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)#16个灰阶表示

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    #将zyx矩阵转化为xyz
    image = image.transpose([2,1,0])#xyz
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):# zyx
    '''
    对序列数据矩阵进行重采样， 使得xyz轴采样值一致
    :param image: 原始数据矩阵  xyz
    :param scan: 序列文件
    :param new_spacing: 要得到的体素间隔
    '''
    # Determine current pixel spacing
    scan_thickness = scan[0].SliceThickness#z
    scan_spacing = scan[0].PixelSpacing#xy
    #spacing = np.array([float(scan_spacing), float(scan_thickness[0]), float(scan_thickness[1])], dtype=np.float32)#zxy
    spacing = np.array([float(scan_spacing[0]), float(scan_spacing[1]), float(scan_thickness)], dtype=np.float32)#xyz spacing

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')#xyz
    
    return image, new_spacing, spacing

def trans_based_window(image_scale, scans):
    '''
    对原始数据基于窗值进行预处理
    :param image_scale: 转换为Hu值后的矩阵   xyz轴顺序
    :param scans: 用于获取sigmoid窗口参数
    '''

    window_center = scans[0].WindowCenter
    window_width = scans[0].WindowWidth

    #具体采用的窗值函数（包括0, 1, 2）
    window_center = window_center
    window_width = window_width

    ymin = np.min(image_scale)
    ymax = np.max(image_scale)

    thres1 = window_center - 0.5 - (window_width - 1) / 2
    thres2 = window_center - 0.5 + (window_width - 1) / 2

    index1 = np.where(image_scale <= thres1)
    index2 = np.where(image_scale > thres2)
    index3 = np.where((image_scale >thres1) & (image_scale<=thres2))

    image_scale[index1]=ymin
    image_scale[index2]=ymax
    image_scale[index3]=((image_scale[index3]-(window_center-0.5))/(window_width-1)+0.5)*(ymax-ymin)+ymin

    result_scale = (image_scale-ymin)/(ymax-ymin)*255
    return result_scale, ymin, ymax#这样保持不丢失信息

def process_to_mango(image_trans):
    '''
    image_trans为xyz格式
    坑点1：mango坐标y轴和原始图像呈现镜像翻转的
    坑点2：mango坐标z轴和原始图像呈现逆序
    
    '''
    image = image_trans.copy()#xyz
    image_flip_reverse = []
    w, h, c = image.shape#xyz
    for i in range(c):
        slice = image[:,:,c-1-i]#xy
        slice = cv2.flip(slice, 0)#进行垂直翻转
        image_flip_reverse.append(slice)

    image_flip_reverse = np.array(image_flip_reverse)#z, x, y
    #image_flip_reverse = image_flip_reverse.transpose([2,1,0])#轴转换 x,y,z
    image_flip_reverse = image_flip_reverse.transpose([1, 2, 0])
    return image_flip_reverse

def save_max_objects(img):
    '''
    ##输入：一张二值图像，无须指定面积阈值，
    ##输出：会返回保留了面积最大的连通域的图像
    '''
    labels = measure.label(img)  # 返回打上标签的img数组
    jj = measure.regionprops(labels)  # 找出连通域的各种属性。  注意，这里jj找出的连通域不包括背景连通域
    # is_del = False
    if len(jj) == 1:
        out = img
        # is_del = False
    else:
        # 通过与质心之间的距离进行判断
        num = labels.max()  #连通域的个数
        del_array = np.array([0] * (num + 1))#生成一个与连通域个数相同的空数组来记录需要删除的区域（从0开始，所以个数要加1）
        for k in range(num):#TODO：这里如果遇到全黑的图像的话会报错
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一个连通域
            else:
                k_area = jj[k].area  # 将元组转换成array

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
        # is_del = True
    return out

def bolt_segment(gray_image):
    '''
    头骨分割，去除脑部CT头骨部分
    :param gray_image: 单slice矩阵
    '''
    binary_except_bone = ((gray_image <255) & (gray_image>0))#阈值分割，去除bone
    
    #先腐蚀
    kernel = np.ones((3,3),np.uint8)
    binary_except_bone = np.array(binary_except_bone, np.uint8)
    erosion_mask = cv2.erode(binary_except_bone, kernel, iterations = 1)
    
    #保留最大连通域
    mask = np.array(erosion_mask, dtype = np.bool)
    out = save_max_objects(binary_except_bone)
    
    #进行空洞填充
    mask_fill_hold = binary_fill_holes(out)#进行孔洞填充
    
    return mask_fill_hold

def stroke_segment(image, bbox):
    '''
    基于GAC进行病灶分割
    '''
    
    # GAC segment
    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store
    def stroke_segment_inline(image, bbox):
        '''
        对输入图片进行病灶分割
        :param image:剔除头骨后的脑图
        :param bbox:病灶范围  
        '''
        [min_x, min_y, max_x, max_y] = bbox
        image = img_as_float(image)
        gimage = inverse_gaussian_gradient(image)

        # Initial level set
        init_ls = np.zeros(image.shape, dtype=np.int8)
        #根据医生标注的框进行分割
        init_ls[min_x:max_x, min_y:max_y] = 1

        #init_ls[10:-10, 10:-10] = 1
        #初值选定为中心点


        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_geodesic_active_contour(gimage, 100, init_ls,
                                                   smoothing=1, balloon=-1,
                                                   threshold=0.69,
                                                   iter_callback=callback)#获取的病灶
        return ls#暂时先不进行任何处理，直接保留所有的分割病灶
    
    return stroke_segment_inline(image, bbox)

def pos_transform(initial_pos, initial_spacing, new_spacing):
    '''
    将实际坐标转化scale坐标
    :param initial_pos: 原始三维坐标
    '''
    scale_x = initial_spacing[0] / new_spacing[0]
    scale_y = initial_spacing[1] / new_spacing[1]
    scale_z = initial_spacing[2] / new_spacing[2]
    
    scale_params = np.array([scale_x, scale_y, scale_z])
    transform_pos = initial_pos * scale_params
    return transform_pos

def main(slice_path, 
         center,
         diammeter):
    '''
    数据处理整体调用接口
    :param slice_path:CT序列路径
    :param center: 病灶中心
    :param diammeter: xyz三轴长度
    :return mask
    '''
    #加载序列
    slices = load_scan(slice_path)
    #xyz 数据矩阵
    image = get_pixels_hu(slices)
    #对数据矩阵进行重采样
    image_resample, new_spacing, spacing = resample(image, slices, new_spacing=[1,1,1])
    #对重采样的数据进行窗值预处理
    result_scale, ymin, ymax = trans_based_window(image_resample, slices)#image_resample为xyz轴顺序
    #对result_scale转换成mango显示格式
    image_flip_reverse = process_to_mango(result_scale)#image_flip_reverse为xyz轴顺序
    
    #center和diammeter基于spacing进行转换
    center_trans = pos_transform(center, spacing, new_spacing)
    diammeter_trans = pos_transform(diammeter, spacing, new_spacing)
    
    #获取min_x, min_y, min_z, max_x, max_y, max_z
    [min_x, min_y, min_z] = center_trans - diammeter_trans / 2.
    [max_x, max_y, max_z] = center_trans + diammeter_trans / 2.

    min_x = math.floor(min_x)
    min_y = math.floor(min_y)
    min_z = math.floor(min_z)

    max_x = math.ceil(max_x)
    max_y = math.ceil(max_y)
    max_z = math.ceil(max_z)
    
    save_path = "./result-check/"+slice_path.split("/")[-1]+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #基于center和diammeter获取相应slice，并进行病灶分割
    for i in range(min_z, max_z):
        stroke_slice = image_flip_reverse[:,:,i]#获取相应slice， xy
        brain_mask = bolt_segment(stroke_slice)#头骨分割
        stroke_slice_except_bone = stroke_slice * brain_mask
        mask = stroke_segment(stroke_slice_except_bone, [min_x, min_y, max_x, max_y])
        
        #opencv画图
        image, contours, hierarchy = cv2.findContours(np.array(mask, dtype=np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(stroke_slice, contours, -1, (255, 0,0), 2)
        cv2.imwrite(save_path + str(i).rjust(3,'0')+".png", stroke_slice)

if __name__=="__main__":
    main(slice_path="/SkyDiscovery/cephfs/user/yanjie.lv/Stroke/PA0/ST0/SE1", 
         center=np.array([355, 259, 8]),
         diammeter=np.array([77, 95, 8]))