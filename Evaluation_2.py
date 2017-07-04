import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2

'''Оценка параметров деформации изображения при помощи псевдоградиентного алгоритма.'''

deform_img = cv2.imread("2.bmp")
deform_img = cv2.cvtColor(deform_img, cv2.COLOR_BGR2GRAY)

deform_img = deform_img.astype('uint8')
cv2.imshow("test_4", deform_img)
cv2.waitKey(0)

img = cv2.imread("2.tiff")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img.astype('uint8')
cv2.imshow("test_4", img)
cv2.waitKey(0)

'''Размеры изображения'''

height = np.size(img, 0)
width = np.size(deform_img, 1)

x0 = (width - 1)/2
y0 = (height - 1)/2

def Inter(x, y):
    inter = img[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')] + \
            int(x - np.floor(x)) * \
            (img[np.ceil(x).astype('uint16')][np.floor(y).astype('uint16')] - img[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')]) + \
            int(y - np.floor(y)) * (img[np.floor(x).astype('uint16')][np.ceil(y).astype('uint16')] - img[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')]) + \
            int(x - np.floor(x)) * int(y - np.floor(y)) * (img[np.ceil(x).astype('uint16')][np.ceil(y).astype('uint16')] + img[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')] - img[np.ceil(x).astype('uint16')][np.floor(y).astype('uint16')] - img[np.floor(x).astype('uint16')][np.ceil(y).astype('uint16')])
    return(inter)

def Inter_2(x, y):
    #.astype('uint8')
    floor_x = int(np.floor(x))
    floor_y = int(np.floor(y))
    ceil_x = int(np.ceil(x))
    ceil_y = int(np.ceil(y))
    img_fx_fy = img[floor_x][floor_y]
    img_cx_fy = img[ceil_x][floor_y]
    img_fx_cy = img[floor_x][ceil_y]
    dif_x_fx = (x - floor_x)
    dif_y_fy = (y - floor_y)
    inter = img_fx_fy + dif_x_fx * \
            (img_cx_fy - img_fx_fy) + \
            dif_y_fy * (img_fx_cy - img_fx_fy) + \
            dif_x_fx * dif_y_fy * \
            (img[ceil_x][ceil_y] + img_fx_fy - img_cx_fy - img_fx_cy)
    return(inter)

def F(f):
    rotate_matrix = np.array([[np.cos(f), -np.sin(f)], [np.sin(f), np.cos(f)]])
    return rotate_matrix

def K(k):
    k_matrix = np.array([[k, 0],[0, k]])
    return k_matrix

def h(hx, hy):
    h_matrix = np.array = ([hx, hy])
    return h_matrix

def dxif(x, y, f, k):
    dxif_matrix = k * (-np.sin(f) * (x - x0) - np.cos(f) * (y - y0))
    return dxif_matrix

def dyif(x, y, f, k):
    dyif_matrix = k * (np.cos(f) * (x - x0) - np.sin(f) * (y - y0))
    return dyif_matrix

def dxik(x, y, f):
    dxik_matrix = np.cos(f) * (x - x0) - np.sin(f) * (y - y0)
    return dxik_matrix

def dyik(x, y, f):
    dyik_matrix = np.sin(f) * (x - x0) - np.cos(f) * (y - y0)
    return dyik_matrix

def dot(p, o):
    res_1 = np.dot(p, o)
    return res_1

def matrix(x, x0, y, y0):
    res_2 = np.array([x - x0, y - y0])
    return res_2

def h_function(hx, hy):
    res_3 = np.array([hx, hy])
    return res_3

def koord(x0, y0):
    res_4 = np.array([x0, y0])
    return res_4

dxihx = 1
dxihy = 0
dyihx = 0
dyihy = 1

T = int(input())

y_axis_0 = []
y_axis_1 = []
y_axis_2 = []
y_axis_3 = []

x_values = []

f = np.zeros((T))
k = np.zeros((T))
hx = np.zeros((T))
hy = np.zeros((T))
M = np.zeros((4))

f[0] = 0
k[0] = 1
hx[0] = 0
hy[0] = 0

for t in range(1, T):
    dQf = 0
    dQk = 0
    dQhx = 0
    dQhy = 0
    m = 0
    while m < 10:
        x = np.ceil(np.random.uniform(width -1)).astype('uint8')
        y = np.ceil(np.random.uniform(height -1)).astype('uint8')
        xiyi = dot(dot(F(f[t-1]), K(k[t-1])), matrix(x, x0, y, y0)) + h_function(hx[t-1], hy[t-1]) + koord(x0, y0)
        xi = xiyi[0]
        yi = xiyi[1]
        if xi >= 1 and xi <= (width - 2) and yi >= 1 and yi <= (height - 2):
            dQf = dQf + 2 * (Inter(xi, yi) - deform_img[x][y]) * (((Inter(xi+1, yi) - Inter(xi-1, yi))/2) *  dxif(x, y, f[t-1], k[t-1]) + ((Inter(xi, yi+1) - Inter(xi, yi-1))/2) * dyif(x, y, f[t-1], k[t-1]))
            dQk = dQk + 2 * (Inter(xi, yi) - deform_img[x][y]) * (((Inter(xi+1, yi) - Inter(xi-1, yi))/2) *  dxik(x, y, f[t-1]) + ((Inter(xi, yi+1) - Inter(xi, yi-1))/2) * dyik(x, y, f[t-1]))
            dQhx = dQhx + 2 * (Inter(xi, yi) - deform_img[x][y]) * (((Inter(xi+1, yi) - Inter(xi-1, yi))/2) * dxihx + ((Inter(xi, yi+1) - Inter(xi, yi-1))/2) * dyihx)
            dQhy = dQhy + 2 * (Inter(xi, yi) - deform_img[x][y]) * (((Inter(xi+1, yi) - Inter(xi-1, yi))/2) * dxihy + ((Inter(xi, yi + 1) - Inter(xi, yi - 1)) / 2) * dyihy)
            m = m + 1
    f[t] = f[t-1] - np.sign(dQf/m) * 0.003
    k[t] = k[t-1] - np.sign(dQk/m) * 0.01
    hx[t] = hx[t-1] - np.sign(dQhx/m) * 0.04
    hy[t] = hy[t - 1] - np.sign(dQhy/m) * 0.04


M[0] = -f[T - 1] * (180 / np.pi)
M[1] = (1 / k[T - 1])
M[2] = (-hx[T - 1])
M[3] = (-hy[T - 1])
print(M)
