
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
    return tc, t


def drawPoint(x, y):
    plt.plot(x, y)
    plt.show()



def div(t, num):
    sky_arr = []
    Not_sky_arr = []
    data = np.int(math.ceil(t.shape[0] / num))
    for i in range(num):
        img1 = t[data * i:data * (i + 1), :]
        if np.mean(img1) <= 120:
            sky_arr.append(img1)
        else:
            Not_sky_arr.append(img1)
    sky_img = np.vstack(tuple(sky_arr))
    Not_sky_img = np.vstack(tuple(Not_sky_arr))
    show('div', sky_img)
    return sky_img







def mean_filter(size, image):
    count = 0
    x = []
    y = []
    start = 0
    data_x = np.int(math.ceil(image.shape[0] / size))
    data_y = np.int(math.ceil(image.shape[1] / size))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img = image[j:j + data_x, i:i + data_y]
            show('image', img)
            value = np.sum(image[i:i + size, j:j + size])
            val = np.double(value) / (image.shape[0] * image.shape[1])
            print(val)
            x.append(count)
            y.append(val)
            i = i + size
    drawPoint(x, y)
    return image





path_01 = './images/tiananmen_input.png'
img = cv2.imread(path_01)
anImg = an_img(img, 7)
img = t_Dark(anImg, 0.95, 7 ,path_01)[0]
# img_msr = MSR.MSR_t(img)
mean_filter(10, img)
