import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2

'''Выполнение геометрических преобразований опорного изображения с заданными
параметрами последовательно (сдвиг, поворот, изменение масштаба, зеркальное
отражение) используя для каждого очередного параметра в качестве опорного
изображения кадр преобразованного изображения с предыдущим параметром.
При этом указанные геометрические преобразования заданного кадра изображения
производятся относительно его центра.'''

img = cv2.imread("7.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('test', gray)
cv2.waitKey(0)

'''Размеры изображения'''

height = np.size(gray, 0)
width = np.size(gray, 1)


x0 = (height - 1)/2
y0 = (width - 1)/2


'''Сдвиг'''

print('Введите параметр hx вектора сдвига(целое число, не больше показателя размерности изображения по соответствующей оси)')
hx = int(input())
print('Введите параметр hy вектора сдвига(целое число, не больше показателя размерности изображения по соответствующей оси)')
hy = int(input())

'''Нахождение значений яркостей в узлах преобразованного кадра изображения для
линейного сдвига при использовании билинейной интерполяции'''

def Inter(x, y):
    inter = gray[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')] + \
            int(x - np.floor(x)) * \
            (gray[np.ceil(x).astype('uint16')][np.floor(y).astype('uint16')] - gray[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')]) + \
            int(y - np.floor(y)) * (gray[np.floor(x).astype('uint16')][np.ceil(y).astype('uint16')] - gray[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')]) + \
            int(x - np.floor(x)) * int(y - np.floor(y)) * (gray[np.ceil(x).astype('uint16')][np.ceil(y).astype('uint16')] + gray[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')] - gray[np.ceil(x).astype('uint16')][np.floor(y).astype('uint16')] - gray[np.floor(x).astype('uint16')][np.ceil(y).astype('uint16')])
    return(inter)


def Inter_for_heap(image, x, y):
    inter = image[np.floor(x).astype('uint16')][np.floor(y).astype('uint16')] + \
            int(x - np.floor(x)) * \
            (image[np.ceil(x).astype('uint16')][np.floor(y).astype('uint16')] - image[np.floor(x).astype('uint16')][
                np.floor(y).astype('uint16')]) + \
            int(y - np.floor(y)) * (
                image[np.floor(x).astype('uint16')][np.ceil(y).astype('uint16')] - image[np.floor(x).astype('uint16')][
                np.floor(y).astype('uint16')]) + \
            int(x - np.floor(x)) * int(y - np.floor(y)) * (
            image[np.ceil(x).astype('uint16')][np.ceil(y).astype('uint16')] + image[np.floor(x).astype('uint16')][
                np.floor(y).astype('uint16')] - image[np.ceil(x).astype('uint16')][np.floor(y).astype('uint16')] -
            image[np.floor(x).astype('uint16')][np.ceil(y).astype('uint16')])
    return (inter)

shift = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        xiyi = np.array([x - x0, y - y0]) + np.array([-hx, -hy]) + np.array([x0, y0])
        xi = xiyi[0]
        yi = xiyi[1]
        if 0 < xi < width and 0 < yi < height:
            shift[x][y] = Inter(xi, yi)

shift=shift.astype('uint8')
cv2.imshow("test",shift)
cv2.waitKey(0)


'''Поворот'''

print('Введите угол поворота в градусах:')
f = float(input())
f = (f * (3.14/180))

rotate_matrix = np.array([[np.cos(f), np.sin(f)], [-np.sin(f), np.cos(f)]])

one = np.zeros((width, height))
turn = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        xiyi = np.dot(rotate_matrix,np.array([x - x0, y - y0])) + np.array([x0, y0])
        xi = xiyi[0]
        yi = xiyi[1]
        if 0 < xi < width-1 and 0 < yi < height-1:
            turn[x][y] = Inter(xi, yi)
            one[x][y] = Inter_for_heap(shift, xi, yi)

turn = turn.astype('uint8')
cv2.imshow("test_2", turn)
cv2.waitKey(0)


'''Изменение масштаба'''

print('Введите коэффициент искажения по х(целое или дробное число)')
kx = float(input())
print('Введите коэффициент искажения по у(целое или дробное число)')
ky = float(input())

rotate_matrix_1 = np.array([[1/kx, 0], [0, 1/ky]])

two = np.zeros((width, height))
extension = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        xiyi = np.dot(rotate_matrix_1,np.array([x - x0, y - y0])) + np.array([x0, y0])
        xi = xiyi[0]
        yi = xiyi[1]
        if 0 < xi < width-1 and 0 < yi < height-1:
            extension[x][y] = Inter(xi, yi)
            two[x][y] = Inter_for_heap(one, xi, yi)

extension = extension.astype('uint8')
cv2.imshow("", extension)
cv2.waitKey(0)


'''Зеркальное отражение'''


rotate_matrix_2 = np.array([[1, 0], [0, -1]])

three = np.zeros((width, height))
mirror = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        xiyi = np.dot(rotate_matrix_2,np.array([x - x0, y - y0])) + np.array([x0, y0])
        xi = xiyi[0]
        yi = xiyi[1]
        if 0 < xi < width-1 and 0 < yi < height-1:
            mirror[x][y] = Inter(xi, yi)
            three[x][y] = Inter_for_heap(two, xi, yi)

mirror = mirror.astype('uint8')
cv2.imshow("test_4", mirror)
cv2.waitKey(0)

heap = np.zeros((width, height))


'''Heap'''

three = three.astype('uint8')
cv2.imshow("test_5", three)
cv2.waitKey(0)
