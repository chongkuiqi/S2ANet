"""
-------------
This is the multi-process version
"""
import os
import codecs
import numpy as np
import math
from dota_utils import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo
import dota_utils as util
import copy

# 进程池
from multiprocessing import Pool
from functools import partial
import time

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    # 根据起点的不同，poly1可以有四种点的排列方法
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)

    # 选择和poly2的点距离最小的排列顺序
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))



class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code = 'utf-8',
                 gap=512,
                 subsize=1024,
                #  thresh=0.7,
                 thresh=0.5,
                 choosebestpoint=True,
                 ext = '.png',
                 padding=True,
                 num_process=8
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """
        self.basepath = basepath
        self.outpath = outpath
        # 编码格式，'utf-8'
        self.code = code
        self.gap = gap
        # 图像切片的尺寸
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        # 原始图像的路径和切图后的保存路径
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
       
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        
    


        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding

        # Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，
        # 就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，
        # 直到池中有进程结束，才会创建新的进程来执行这些请求
        self.pool = Pool(num_process)
        print('padding:', padding)

        # pdb.set_trace()
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            # pdb.set_trace()
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
        # pdb.set_trace()
    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []

        # 切图在大图上的坐标，也可看做一个旋转框
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        # self.code，编码格式，'utf-8'
        with codecs.open(outdir, 'w', self.code) as f_out:
            # 该大图上的目标，与切图进行计算
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 0):
                    continue
                # inter_poly为目标框与切图的交集部分构成的多边形，half_iou为交集面积/旋转框面积
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                # print('writing...')
                # 如果整个目标均在切图内部
                if (half_iou == 1):
                    # 目标框坐标转化为在切图上的坐标
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                elif (half_iou > 0):
                #elif (half_iou > self.thresh):
                  ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    # inter_poly.exterior.coords表示形成多边形外部边界的点
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    
                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])
                    
                    # 如果多边形的点少于四个，说明旋转框去掉了1个角，则把这个目标框去掉
                    if len(out_poly) < 4:
                        continue
                    # 如果刚好有4个点，说明旋转框去掉了2个角，不需要对多边形做进一步的处理，该目标框保留
                    elif len(out_poly) == 4:
                        pass
                    # 如果多边形有5个点，说明旋转框去掉了一个角，这样的框保留，并且作进一步处理
                    elif (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    # 如果多边形大于5个点，这种情况不太可能发生，总之去掉该目标框
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue

                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        # outline = outline + ' ' + obj['name'] + ' ' + '2'
                        continue
                    f_out.write(outline + '\n')
                #else:
                 #   mask_poly.append(inter_poly)
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        try:
            img = cv2.imread(os.path.join(self.imagepath, name + extent))
            print('img name:', name)
        except:
            print(f'can not read img name:{name}')
            # 这里我们改一下，如果找不到图像，干脆就退出程序
            exit()
        if np.shape(img) == ():
            print("图像数据为空")
            exit()
            # return
        
        # 获得对应的标注文件
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
            #obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

        # 多尺度切图
        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        resizeimg_w = np.shape(resizeimg)[1]
        resizeimg_h = np.shape(resizeimg)[0]

        # if (max(weight, height) < self.subsize):
        #     return

        # 切图的左上角点
        left, up = 0, 0
        # 下面的while循环，是从左向右遍历进行切图
        while (left < resizeimg_w):
            # 从左向右，如果超出了图像右边界，则从最右边界向左切出subsize，如果这样超出了左边界，则从0开始
            if (left + self.subsize >= resizeimg_w):
                left = max(resizeimg_w - self.subsize, 0)
            up = 0
            
            # 下面的while，是从上向下遍历进行切图
            while (up < resizeimg_h):
                # 
                if (up + self.subsize >= resizeimg_h):
                    up = max(resizeimg_h - self.subsize, 0)
                
                # 右端点最大，也不能超出整张图像的右边界
                right = min(left + self.subsize, resizeimg_w - 1)
                down = min(up + self.subsize, resizeimg_h - 1)
                # 切片子图的名字
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                
                # 保存图像，以及对应的标签，此时objects仍然是大图上的坐标，而不是切图上的坐标
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                # 跳出从上往下遍历的切图
                if (up + self.subsize >= resizeimg_h):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= resizeimg_w):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """

        # 获取图像列表
        imagelist = GetFileFromThisRootDir(self.imagepath)
        # 图像文件名字，不包括拓展名
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]

        # partial是python自带的函数，就是把一个函数的参数固定住，然后返回一个新的函数
        # 这里就是把self.SplitSingle函数的rate和extent固定住，然后返回给worker函数
        worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
        #
        # for name in imagenames:
        #     self.SplitSingle(name, rate, self.ext)
        # 

        # Pool类中的map方法，与内置的map函数用法行为基本一致，它会使进程阻塞直到结果返回
        # 注意：虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程
        self.pool.map(worker, imagenames)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
if __name__ == '__main__':
    # example usage of ImgSplit
    # start = time.clock()
    # split = splitbase(r'/data/dj/dota/val',
    #                    r'/data/dj/dota/val_1024_debugmulti-process_refactor') # time cost 19s
    # # split.splitdata(1)
    # # split.splitdata(2)
    # split.splitdata(0.4)
    #
    # elapsed = (time.clock() - start)
    # print("Time used:", elapsed)

    # split = splitbase(r'/data/dota2/train',
    #                    r'/data/dota2/train1024',
    #                   gap=200,
    #                   subsize=1024,
    #                   num_process=32
    #                   )
    # split_train = splitbase(r'/data0/data_dj/dota2/train',
    #                   r'/data0/data_dj/dota2/trainval1024_ms',
    #                     gap=200,
    #                     subsize=1024,
    #                     num_process=30)
    # split_train.splitdata(1.5)
    # split_train.splitdata(0.5)
    #
    # split_val = splitbase(r'/data0/data_dj/dota2/val',
    #                       r'/data0/data_dj/dota2/trainval1024_ms',
    #                       gap=200,
    #                       subsize=1024,
    #                       num_process=30)
    # split_val.splitdata(1.5)
    # split_val.splitdata(0.5)


    # split = splitbase(r'/home/dingjian/project/dota2/test-c1',
    #                   r'/home/dingjian/project/dota2/test-c1-1024',
    #                   gap=512,
    #                   subsize=1024,
    #                   num_process=16)
    # split.splitdata(1)


    # split_train = splitbase(r'/data/mmlab-dota1.5/train',
    #                   r'/data/mmlab-dota1.5/split-1024/trainval1024_ms',
    #                     gap=200,
    #                     subsize=1024,
    #                     num_process=40)
    # split_train.splitdata(1.5)
    # split_train.splitdata(0.5)
    #
    # split_val = splitbase(r'/data/mmlab-dota1.5/val',
    #                       r'/data/mmlab-dota1.5/split-1024/trainval1024_ms',
    #                       gap=200,
    #                       subsize=1024,
    #                       num_process=40)
    # split_val.splitdata(1.5)
    # split_val.splitdata(0.5)
    #
    # split_train_single = splitbase('/data/mmlab-dota1.5/train',
    #                                '/data/mmlab-dota1.5/split-1024/trainval1024',
    #                                gap=200,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_single.splitdata(1)
    #
    # split_val_single = splitbase('/data/mmlab-dota1.5/val',
    #                              '/data/mmlab-dota1.5/split-1024/trainval1024',
    #                              gap=200,
    #                              subsize=1024,
    #                              num_process=40)
    # split_val_single.splitdata(1)

    # dota-1.5 1024 split new
    # split_train_single = splitbase('/data/mmlab-dota1.5/train',
    #                                '/data/mmlab-dota1.5/split-1024_v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_single.splitdata(1)
    #
    # split_train_ms = splitbase('/data/mmlab-dota1.5/train',
    #                                '/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_ms.splitdata(0.5)
    # split_train_ms.splitdata(1.5)
    #
    # # val
    # split_val_single = splitbase('/data/mmlab-dota1.5/val',
    #                                '/data/mmlab-dota1.5/split-1024_v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_val_single.splitdata(1)
    #
    # split_val_ms = splitbase('/data/mmlab-dota1.5/val',
    #                            '/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_val_ms.splitdata(0.5)
    # split_val_ms.splitdata(1.5)
    #
    # # test
    # split_test_single = splitbase('/data/mmlab-dota1.5/test',
    #                                '/data/mmlab-dota1.5/split-1024_v2/test1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_test_single.splitdata(1)
    #
    # split_test_ms = splitbase('/data/mmlab-dota1.5/test',
    #                            '/data/mmlab-dota1.5/split-1024_v2/test1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_test_ms.splitdata(0.5)
    # split_test_ms.splitdata(1.5)

    # split_train_single = splitbase(r'/data/data_dj/dota2/train',
    #                                r'/data/data_dj/dota2/split-1024-v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_train_single.splitdata(1)
    #
    # split_train_ms = splitbase(r'/data/data_dj/dota2/train',
    #                            r'/data/data_dj/dota2/split-1024-v2/trainval1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_train_ms.splitdata(0.5)
    # split_train_ms.splitdata(1.5)
    #
    #
    # split_val_single = splitbase(r'/data/data_dj/dota2/val',
    #                                r'/data/data_dj/dota2/split-1024-v2/trainval1024',
    #                                gap=512,
    #                                subsize=1024,
    #                                num_process=40)
    # split_val_single.splitdata(1)
    #
    # split_val_ms = splitbase(r'/data/data_dj/dota2/val',
    #                            r'/data/data_dj/dota2/split-1024-v2/trainval1024_ms',
    #                            gap=512,
    #                            subsize=1024,
    #                            num_process=40)
    # split_val_ms.splitdata(0.5)
    # split_val_ms.splitdata(1.5)

    split_test_single = splitbase(r'/home/dingjian/project/dota2/test-dev',
                                  r'/home/dingjian/workfs/dota2_v2/split-1024-v2/test-dev1024',
                                  gap=512,
                                  subsize=1024,
                                  num_process=16)
    split_test_single.splitdata(1)

    split_test_ms = splitbase(r'/home/dingjian/project/dota2/test-dev',
                                  r'/home/dingjian/workfs/dota2_v2/split-1024-v2/test-dev1024_ms',
                                  gap=512,
                                  subsize=1024,
                                  num_process=16)
    # split_test_ms.splitdata(1)
    split_test_ms.splitdata(0.5)
    split_test_ms.splitdata(1.5)
