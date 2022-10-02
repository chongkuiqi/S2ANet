
import numpy as np
import math
import cv2

AR_thr = 5.0

LAR1024_classes_names = ("vehicle",)

# 输入角度范围为(0,pi]，归一化后角度范围为[-0.25pi, 0.75pi)之间
def norm_angle(angle):
    range = [-np.pi / 4, np.pi]
    # 对pi取余数，获得超过pi的部分，然后在
    ang = (angle - range[0]) % range[1] + range[0]
    return ang



def rotated_box_to_poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly


def rotated_box_to_poly_np(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        poly = rotated_box_to_poly_single(rrect)
        polys.append(poly)
    
    polys = np.array(polys).reshape(-1,8)
    return polys


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)


def poly_to_rotated_box_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_box:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    width = max(edge1, edge2)
    height = min(edge1, edge2)


    # np.arctan()输入的是正切值，输出的是弧度，角度范围[-0.5pi, 0.5pi]
    # np.arctan(x1, x2)输入的是坐标值，表示x1/x2是正切值，输出的是弧度，角度范围[-pi, pi]，因为可以根据x1和x2判断落在那个象限
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

    angle = norm_angle(angle)

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    rotated_box = np.array([x_ctr, y_ctr, width, height, angle])
    return rotated_box


def poly_to_rotated_box_np(polys):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_boxes:[x_ctr,y_ctr,w,h,angle]
    """
    rotated_boxes = []
    # print(polys.shape)
    for poly in polys:
        rotated_box = poly_to_rotated_box_single(poly)
        rotated_boxes.append(rotated_box)
    
    # print(rotated_boxes)
    return np.array(rotated_boxes).reshape(-1,5)



def load_DOTA_label(txt_pathname):
    with open(txt_pathname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 空图像
    if len(lines) == 0:
        return None, None, None

    if "imagesource" in lines[0]:
        # assert "imagesource" in lines[0]
        # assert "gsd" in lines[1]
        lines = lines[2:]
    
    lines = [line.strip() for line in lines]

    lines = [line.split(' ') for line in lines]

    boxes_points = [line[:8] for line in lines]
    boxes_points = [list(map(float, box_points)) for box_points in boxes_points]
    boxes_points = np.array(boxes_points).reshape(-1,8)

    boxes_classname = [line[8] for line in lines]

    boxes_difficult = np.array([float(line[9]) for line in lines]).reshape(-1)
    return boxes_points, boxes_classname, boxes_difficult


# 根据旋转框，获得外接水平框
def get_hboxes(boxes_points):
    '''
    boxes_points: shape (N,8)
    '''

    np.min(boxes_points)

    boxes_x = boxes_points[:, [0,2,4,6]]
    boxes_y = boxes_points[:, [1,3,5,7]]

    xmin = boxes_x.min(axis=1).reshape(-1,1)
    xmax = boxes_x.max(axis=1).reshape(-1,1)
    ymin = boxes_y.min(axis=1).reshape(-1,1)
    ymax = boxes_y.max(axis=1).reshape(-1,1)

    hboxes = np.concatenate((xmin,ymin,xmax,ymax), axis=1).reshape(-1,4)


    return hboxes

    

def plot_hbboxes(img, hboxes, cls_fall_point=0, color=(0, 0, 255), thickness=3):
    '''
        画水平框，以及目标类别
    inputs:
        img:
        box_points: shape:[N,9] or [N,8], np.float64类型
        classes_name:
        cls_fall_point:表示，目标类别文本画在旋转框四个角点的哪个角上
    return:
        img
    '''
    num_cols = hboxes.shape[1]
    # 逐个画box
    for hbox in hboxes:

        # 画4个点组成的轮廓，也就是旋转框
        # cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
        # contours:轮廓列表，每个元素都是一个轮廓, 元素可以是一个numpy数组，
        # contourIdx：表示画列表中的哪个轮廓，-1表示画所有轮廓
        # thickness：如果是-1（cv2.FILLED），则为填充模式。
        # cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift )
        pt1 = hbox[:2].astype(np.int32)
        pt2 = hbox[2:].astype(np.int32)
        cv2.rectangle(img, pt1=pt1,pt2=pt2, color=color, thickness=thickness)
        
        # if cls_fall_point <0 or cls_fall_point > 3:
        #     print(f"cls_fall_pont must be 0-3 !!!")
        #     exit()
        # pt1 = tuple(box_points[cls_fall_point])
        # # cv2.putText，按照顺序，各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        # cv2.putText(img, 
        #     text = classes_name[cls_id], 
        #     org = pt1, 
        #     fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
        #     fontScale = 1, 
        #     color = color, 
        #     thickness = thickness
        # )

    return img



def plot_rotate_boxes(img, boxes_points, cls_fall_point=0, color=(0, 0, 255), thickness=3):
    '''
        画旋转框，以及目标类别
    inputs:
        img:
        box_points: shape:[N,9] or [N,8], np.float64类型
        classes_name:
        cls_fall_point:表示，目标类别文本画在旋转框四个角点的哪个角上
    return:
        img
    '''
    num_cols = boxes_points.shape[1]
    # 逐个画box
    for box_points in boxes_points:
        if num_cols == 9:
            # 目标类别
            cls_id = int(box_points[0])
            box_points = box_points[1:].reshape(4,2).astype(np.int32)
        elif num_cols == 8:
            box_points = box_points.reshape(4,2).astype(np.int32)
        # 画4个点组成的轮廓，也就是旋转框
        # cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
        # contours:轮廓列表，每个元素都是一个轮廓, 元素可以是一个numpy数组，
        # contourIdx：表示画列表中的哪个轮廓，-1表示画所有轮廓
        # thickness：如果是-1（cv2.FILLED），则为填充模式。
        cv2.drawContours(img, [box_points], contourIdx=-1, color=color, thickness=thickness)
        
        # if cls_fall_point <0 or cls_fall_point > 3:
        #     print(f"cls_fall_pont must be 0-3 !!!")
        #     exit()
        # pt1 = tuple(box_points[cls_fall_point])
        # # cv2.putText，按照顺序，各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        # cv2.putText(img, 
        #     text = classes_name[cls_id], 
        #     org = pt1, 
        #     fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
        #     fontScale = 1, 
        #     color = color, 
        #     thickness = thickness
        # )

    return img


def plot_rotate_boxes_and_ids(img, boxes_points, boxes_ids, cls_fall_point=0, color=(0, 0, 255), thickness=3):
    '''
        画旋转框，以及目标类别
    inputs:
        img:
        box_points: shape:[N,9] or [N,8], np.float64类型
        classes_name:
        cls_fall_point:表示，目标类别文本画在旋转框四个角点的哪个角上
    return:
        img
    '''
    num_cols = boxes_points.shape[1]
    # 逐个画box
    for idx, box_points in enumerate(boxes_points):
        if num_cols == 9:
            # 目标类别
            cls_id = int(box_points[0])
            box_points = box_points[1:].reshape(4,2).astype(np.int32)
        elif num_cols == 8:
            box_points = box_points.reshape(4,2).astype(np.int32)
        # 画4个点组成的轮廓，也就是旋转框
        # cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
        # contours:轮廓列表，每个元素都是一个轮廓, 元素可以是一个numpy数组，
        # contourIdx：表示画列表中的哪个轮廓，-1表示画所有轮廓
        # thickness：如果是-1（cv2.FILLED），则为填充模式。
        cv2.drawContours(img, [box_points], contourIdx=-1, color=color, thickness=thickness)
        
        # if cls_fall_point <0 or cls_fall_point > 3:
        #     print(f"cls_fall_pont must be 0-3 !!!")
        #     exit()
        pt1 = tuple(box_points[0])
        box_id = str(boxes_ids[idx].item())
        # # cv2.putText，按照顺序，各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(img, 
            text = box_id, 
            org = pt1, 
            fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 1, 
            color = (255,0,0), 
            thickness = 1
        )

    return img


def get_boxes_AR(boxes_points):
    
    boxes_AR = []

    for box_points in boxes_points:
        box_points = box_points.reshape(4,2).astype(np.int32)
        # 找到最小外接矩形
        # edge1, edge2不确定哪个长度更长，要自己确定
        rotate_box = cv2.minAreaRect(box_points)
        (x, y), (edge1, edge2), theta = rotate_box
        if edge1 > edge2:
            box_AR = edge1 / edge2
        else:
            box_AR = edge2 / edge1

        boxes_AR.append(box_AR)
    
    boxes_AR = np.array(boxes_AR).reshape(-1)

    return boxes_AR