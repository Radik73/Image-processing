import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2

'''Наложение искажений типа: смаз и расфокусировка на изображения и добавление в них шума'''

img = cv2.imread("10.bmp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''Размеры изображения'''

height = np.size(gray, 0)
width = np.size(gray, 1)
cv2.imshow('Input image', gray)


'''Получение центрированного спектра исходного изображения'''

spectrum = np.fft.fft2(gray)
spectrum =  - spectrum / (np.sqrt(height * width))
centered_spectrum = np.fft.fftshift(spectrum) #Центрирование спектра
magnitude_spectrum = 20*np.log(np.abs(centered_spectrum)) #Амплитуды составляющих спектра
plt.plot(),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


'''Формирование ФРТ для смаза'''

print('Введите знчение параметра искажения(к примеру, от 10 до 20):')
r = int(input())

smooth = np.zeros((r))
for y in range(r):
        smooth[y] = 1/r


'''Формирование ФРТ для расфокусировки'''

re_focus = np.zeros((2*r+1, 2*r+1))
for x in range(2*r):
    for y in range(2*r):
        if ((x - r)**2) + ((y - r)**2) <= r**2:
            re_focus[x][y] = (1/((3.14)*r**2))


'''Свёртка ФРТ и исходного изображения'''

smoothy = cv2.filter2D(gray, -2, smooth)
cv2.imshow('Convolution', smoothy)
cv2.waitKey(0)

'''Дополнение ФРТ нулями до размеров исходного изображения'''

added_frp_smooth = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        if y < r and x == 0:
            added_frp_smooth[x][y] = 1/r
        else:
            added_frp_smooth[x][y] = 0


added_frp_re_focus = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        if ((x - r)**2) + ((y - r)**2) <= r**2:
            added_frp_re_focus[x][y] = 1/(3.14*(r**2))


'''Получение передаточной функции искажающей системы'''

spectrum_deformation_system = np.fft.fft2(added_frp_smooth) #Двумерное преобразование Фурье (added_frp_smooth / added_frp_re_focus)
spectrum_deformation_system = spectrum_deformation_system / (np.sqrt(height * width))
tr_function = np.zeros((width, height), dtype=np.complex)
tr_function = spectrum_deformation_system * width
#print(tr_function)

'''Получение центрированного амплитудного спектра ФРТ'''

frp_spectrum = np.fft.fft2(added_frp_smooth)
frp_spectrum = frp_spectrum * (np.sqrt(height * width))
centered_spectrum_frp = np.fft.fftshift(frp_spectrum) # (added_frp_smooth / added_frp_re_focus)
frp_spectrum_center = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        frp_spectrum_center[x][y] = abs(centered_spectrum_frp[x][y])


'''Применение линейного контрастирования'''

max_frp_spectrum_center = frp_spectrum_center.max()
min_frp_spectrum_center = frp_spectrum_center.min()

max_value = 255
min_value = 0

frp_spectrum_center_contrast = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        frp_spectrum_center_contrast[x][y] = (((frp_spectrum_center[x][y] - min_frp_spectrum_center)/(max_frp_spectrum_center - min_frp_spectrum_center))*(max_value - min_value) + min_value)

plt.plot(),plt.imshow(frp_spectrum_center_contrast, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

'''Cпектр искаженного изображения'''

print('Введите значение математическое ожидание')
m = int(input())
print('Введите значение дисперсии шума')
sig = int(input())

spectrum_deform_img = np.zeros((width, height), dtype=np.complex)
for x in range(width):
    for y in range(height):
        spectrum_deform_img[x][y] =  tr_function[x][y] * spectrum[x][y]

pre_deformation_img = np.fft.ifft2(spectrum_deform_img)#Обратное преобразование Фурье
pre_deformation_img = pre_deformation_img * (np.sqrt(height * width))

'''Иcкажённое изображение'''

deformation_img = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        deformation_img[x][y] = np.abs(pre_deformation_img[x][y])


'''Обрезка и ограничение диапазона яркостей искажённого изображения'''

cutting = np.zeros((width - 2*r, height - 2*r))
for x in range(width - 2*r):
    for y in range(height - 2 * r):
        cutting[x][y] = deformation_img[x + r][y + r]
        if cutting[x][y] < 0:
            cutting[x][y] = 0
        elif cutting[x][y] > 255:
            cutting[x][y] = 255
cv2.imshow('Smoothy image', cutting.astype('uint8'))
cv2.waitKey(0)


'''Получение матрицы шума и зашумление изображения'''

gauss = np.random.normal(m, sig, (width - 2*r, height - 2*r))
noisy = cutting + gauss

for x in range(width - 2*r):
    for y in range(height - 2*r):
        if noisy[x][y] < 0:
            noisy[x][y] = 0
        elif noisy[x][y] > 255:
            noisy[x][y] = 255
noisy = noisy.astype('uint8')
cv2.imshow('Noisy image', noisy)
cv2.waitKey(0)


'''Получение спектра искажённого изображения'''

spectrum_2 = np.fft.fft2(noisy)
spectrum_2 = spectrum_2 / (np.sqrt((height - 2*r) * (width - 2*r)))
spectrum_2 = -spectrum_2
centered_spectrum_2 = np.fft.fftshift(spectrum_2)
magnitude_spectrum_2 = 20*np.log(np.abs(centered_spectrum_2))

plt.plot(),plt.imshow(magnitude_spectrum_2, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

