import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def replaceZeroes(data):
    # 假设这里的data图像的大小是（213， 320），data[np.nonzero(data)]为一维的非0数组，数组总共有68160个元素
    # 而 min_nonzero 的意思是：获取data中非0值的最小值
    min_nonzero = min(data[np.nonzero(data)])
    # 把data中像素值为0的元素换成最小值
    data[data == 0] = min_nonzero
    return data


def MSR(src_img, size, ci):
    w = 1/3.0
    length = len(ci)
    log_R = []
    src_img = replaceZeroes(src_img)
    img_double = np.double(src_img)
    img_log = np.log(img_double / 255.0)

    for i in range(length):
        i_gauss =  cv2.GaussianBlur(src_img, (size, size), ci[i])
        # i_gauss =  gu.conv_2d(gu.gauss(size, ci[i]), src_img)
        mul = cv2.multiply(img_log, np.log(np.double(i_gauss)/255.0))
        test_img = cv2.subtract(img_log, mul)
        log_R.append(test_img)

    R_test = 0
    for i in range(length):
        R_test += w * log_R[i]
    test_img = np.exp(R_test)
    dst_R = cv2.normalize(test_img, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return  log_uint8



def MSR_t(img):

    size = 3
    ci = [15, 75, 101, 301]
    if len(img.shape) == 3 :
        b, g, r = cv2.split(img)
        b_MSR = MSR(b, size, ci)
        g_MSR = MSR(g, size, ci)
        r_MSR = MSR(r, size, ci)
        img_MSR = cv2.merge((b_MSR, g_MSR, r_MSR))
    else:
        img_MSR = MSR(img, size, ci)
    return img_MSR
