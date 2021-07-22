import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import MSR


def show(name, *args):
    img = np.hstack(args)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cal_show(img, img_test):
    fig, axex = plt.subplots(nrows=2, ncols=2, figsize=[10, 8], dpi=100)
    cul_img = cv2.calcHist([img], [0], None, [256], [0, 255])
    cul_img_test = cv2.calcHist([img_test], [0], None, [256], [0, 255])
    axex[0][0].imshow(img[:, :, ::-1])
    axex[0][0].set_title("original_image")
    axex[1][0].plot(cul_img)
    axex[1][0].grid()
    axex[0][1].imshow(img_test[:, :, ::-1])
    axex[0][1].set_title("new_img")
    axex[1][1].plot(cul_img_test)
    axex[1][1].grid()
    plt.show()


def an_img(img, R):
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        small_img = np.where(b > g, g, b)
        image = np.where(small_img < r, small_img, r)
        image = cv2.erode(image, np.ones((2 * R + 1, 2 * R + 1)))
    else:
        image = cv2.erode(img, np.ones((2 * R + 1, 2 * R + 1)))

    # show("1", image)
    return image





def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def t_Dark(an_img, w, R, path_01):
    r_value = 4*R
    t = 1 - w * an_img
    t = np.where(t > 0.1, t, 0.1)
    t_thresh = guidedfilter(cv2.imread(path_01, 0) /255.0, t,  5000, 0.0000000001)
    pic_thresh = cv2.normalize(t_thresh, None, 0, 255, cv2.NORM_MINMAX)
    pic_thresh = cv2.convertScaleAbs(pic_thresh)

    t = guidedfilter(cv2.imread(path_01, 0) / 255.0, t,  r_value, 0.001)
    pic_thresh = cv2.normalize(t_thresh, None, 0, 255, cv2.NORM_MINMAX)
    tc = cv2.convertScaleAbs(pic_thresh)
    show('t', tc)
    cal_show(cv2.merge((tc, tc, tc)), cv2.merge((tc, tc, tc)))

    return pic_thresh, t


def div(t, num):
    sky_arr = []
    Not_sky_arr = []
    data = np.int(math.ceil(t.shape[0] / num))
    for i in range(num):
        img1 = t[data * i:data * (i + 1), :]
        if np.mean(img1) <= 150:
            sky_arr.append(img1)
        else:
            Not_sky_arr.append(img1)
    sky_img = np.vstack(tuple(sky_arr))
    Not_sky_img = np.vstack(tuple(Not_sky_arr))
    return sky_img, Not_sky_img


def splitImg(img, path):
    anImg = an_img(img, 7)
    bin_value, t = t_Dark(anImg, 0.95, 7, path)
    sky_img, Not_sky_img = div(bin_value, 100)
    img = MSR.MSR_t(sky_img)
    cal_show(cv2.merge((img, img, img)), cv2.merge((img, img, img)))
    sky_img = np.where(img > 0, 255, 0)
    Not_sky_img = np.where(Not_sky_img > 0, 255, 255)
    sky_img = sky_img.astype('uint8')
    Not_sky_img = Not_sky_img.astype('uint8')
    image = np.vstack((sky_img, Not_sky_img))
    show('im', img)
    show('img', image)
    return image


def BrightAir_A(img1, thresh):
    A = 0
    img_of_not_air = np.where(thresh == 255)
    img_of_is_air = np.where(thresh == 0)
    (b, g, r) = cv2.split(img1)
    b_of_air = b[img_of_is_air]
    g_of_air = g[img_of_is_air]
    r_of_air = r[img_of_is_air]

    b = np.array(b_of_air)
    g = np.array(g_of_air)
    r = np.array(r_of_air)

    small_img = np.where(b > g, b, g)
    image = np.where(small_img < r, r, small_img)

    cul_img = cv2.calcHist([image], [0], None, [256], [0, 256])
    sum = 0
    cul = 0
    for i in range(250, 150, -1):
        sum += cul_img[i][0]
    for i in range(250, 150, -1):
        cul += cul_img[i][0]
        value = (cul/ sum) * 100

        print(i, value)
        if value > 0 and value < 10:
            A = i
            break
    return A


# 输入黑白图像和雾图
def air_area_Img(BinImg, fog_img):
    b, g, r = cv2.split(fog_img)
    b_area = np.where(BinImg != 0, b, 0)
    g_area = np.where(BinImg != 0, g, 0)
    r_area = np.where(BinImg != 0, r, 0)
    area = cv2.merge((b_area, g_area, r_area))

    b_air = np.where(BinImg != 255, b, 0)
    g_air = np.where(BinImg != 255, g, 0)
    r_air = np.where(BinImg != 255, r, 0)
    air = cv2.merge((b_air, g_air, r_air))

    show('area', area)
    show("air", air)
    return air, area


# 计算透射率
def t_clear(img, A, path):
    b, g, r = cv2.split(img)
    b = np.double(b) / A
    g = np.double(g) / A
    r = np.double(r) / A
    mer_img = cv2.merge((b, g, r))
    J_an_img = an_img(mer_img, 7)
    # print(J_an_img.shape)
    t = t_Dark(J_an_img, 0.95, 7, path)[1]
    return t

def clear_fog(A, t, air_img, area_img):
    # b, g, r = cv2.split(fog_img)
    # img_b = ((np.double(b) - A) / t) + A
    # img_g = ((np.double(g) - A) / t) + A
    # img_r = ((np.double(r) - A) / t) + A
    # img = cv2.merge((img_b, img_g, img_r))
    # pic_tou = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # pic_tou = cv2.convertScaleAbs(pic_tou)

    # 对非天空区域处理
    b, g, r = cv2.split(area_img)
    area_b = np.where(b != 0, ((np.double(b) - A) / t) + A, 0)
    area_g = np.where(g != 0, ((np.double(g) - A) / t) + A, 0)
    area_r = np.where(r != 0, ((np.double(r) - A) / t) + A, 0)
    img = cv2.merge((area_b, area_g, area_r))


    # 对天空区域处理
    b, g, r = cv2.split(air_img)
    air_b = np.where(b != 0, b, 0)
    air_g = np.where(g != 0, g, 0)
    air_r = np.where(r != 0, r, 0)

    # 两幅图像融合相加
    clear_b = air_b + area_b
    clear_g = air_g + area_g
    clear_r = air_r + area_r
    img = cv2.merge((clear_b, clear_g, clear_r))


    pic_tou = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    pic_tou = cv2.convertScaleAbs(pic_tou)

    show("int", pic_tou)
    return pic_tou


path_01 = './images/nong10.png'
img = cv2.imread(path_01)
BinImg = splitImg(img, path_01)
anImg = an_img(img, 7)
A = BrightAir_A(img, BinImg)
print(A)
t = t_clear(img, A, path_01)
air, area = air_area_Img(BinImg, img)
clear = clear_fog(A, t, air, area)
print(clear)
show('clear', clear)


