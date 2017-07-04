import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Distortion_and_Noise_3


'''Восстановление изображений при помощи инверсного фильтра, фильтра Винера и фильтра Тихонова'''

height = np.size(Distortion_and_Noise_3.cutting, 0)
width = np.size(Distortion_and_Noise_3.cutting, 1)

frp_for_filter = np.zeros((width, height))
for i in range(width):
    for j in range(height):
        frp_for_filter[i][j] =  Distortion_and_Noise_3.added_frp_smooth[i][j]

spectrum_3 = np.fft.fft2(frp_for_filter) #Двумерное преобразование Фурье
spectrum_3 = spectrum_3 / (np.sqrt(height * width))
spectrum_3 = np.conj(spectrum_3)

H = np.zeros((width, height), dtype=np.complex)
H =  spectrum_3 * width

'''Получим функцию окна'''

#b = int(10)
#y = int(850)
#windows_function = np.zeros(width)
#for i in range(width):
#    windows_function[i] = 0.5 * (np.tanh((i + (y/2))/b) - np.tanh((i - (y/2))/b))


'''Умножение искажённого изображения на функцию окна'''

#для горизонтально смаза
#img_with_window = np.zeros((width, height))
#for i in range(width):
#    for j in range(height):
#        img_with_window[i][j] = thirt.noisy[i][j] * windows_function[j] * windows_function[height - j]


'''Спектр искажённого изображения'''

spectrum_4 = np.fft.fft2(Distortion_and_Noise_3.cutting) #Двумерное преобразование Фурье


'''Реализуем инверсный фильтр'''

inversion_filter = np.zeros((width, height), dtype = np.complex)
for i in range(width):
    for j in range(height):
        if H[i][j] == 0:
            inversion_filter[i][j] = (1/(H[i][j] + 10**(-30)))
        elif H[i][j] != 0:
            inversion_filter[i][j] = (1/H[i][j])



'''Восстановление изображения инверсным фильтром'''

reconstructed_image = np.zeros((width, height), dtype= np.complex)
for i in range(width):
    for j in range(height):
        reconstructed_image[i][j] = inversion_filter[i][j] * spectrum_4[i][j]

spectrum_reconstructed_image = np.fft.ifft2(reconstructed_image)

complete_reconstructed_image_inversion = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        complete_reconstructed_image_inversion[x][y] = np.abs(spectrum_reconstructed_image[x][y])
        if spectrum_reconstructed_image[x][y] < 0:
            complete_reconstructed_image_inversion[x][y] = 0
        elif spectrum_reconstructed_image[x][y] > 255:
            complete_reconstructed_image_inversion[x][y] = 255
cv2.imshow('Inversion filter', complete_reconstructed_image_inversion.astype('uint8'))
cv2.waitKey(0)



'''Фильтр Винера'''

K = float(0.00005)
viner_filter = np.zeros((width, height), dtype = np.complex)
for i in range(width):
    for j in range(height):
        viner_filter[i][j] = (np.conj(H[i][j]) / ((np.abs(H[i][j]) ** 2) + K))



'''Восстановление изоюбражения фильтром Винера'''
reconstructed_image_vinner = np.zeros((width, height), dtype= np.complex)
for i in range(width):
    for j in range(height):
        reconstructed_image_vinner[i][j] = viner_filter[i][j] * spectrum_4[i][j]


spectrum_reconstructed_image_viner = np.fft.ifft2(reconstructed_image_vinner)



complete_reconstructed_image_viner = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        complete_reconstructed_image_viner[x][y] = np.abs(spectrum_reconstructed_image_viner[x][y])
        if spectrum_reconstructed_image_viner[x][y] < 0:
            complete_reconstructed_image_viner[x][y] = 0
        elif spectrum_reconstructed_image_viner[x][y] > 255:
            complete_reconstructed_image_viner[x][y] = 255
cv2.imshow('Viner`s filter', abs(complete_reconstructed_image_viner).astype('uint8'))
cv2.waitKey(0)


'''Фильтр Тихонова'''

P = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

for_p_function = np.zeros((width, height))
for i in range(3):
    for j in range(3):
        for_p_function[i][j] = P[i][j]

for i in range(4, width):
    for j in range(4, height):
        for_p_function[i][j] = 0

spectrum = np.fft.fft2(for_p_function) #Двумерное преобразование Фурье

p_function = np.zeros((width, height), dtype = np.complex)
for i in range(width):
    for j in range(height):
        p_function[i][j] = spectrum[i][j] * width


'''Реализация фильтра(Тихонова)'''

v = float(0.00005)
tihonov_filter = np.zeros((width, height), dtype = np.complex)
for i in  range(width):
    for j in range(height):
        tihonov_filter[i][j] = (np.conj(H[i][j])/((np.conj(H[i][j]) * H[i][j]) + v * (np.conj(p_function[i][j]) * p_function[i][j])))

reconstructed_image_tihonov = np.zeros((width, height), dtype = np.complex)
for x in range(width):
    for y in range(height):
        reconstructed_image_tihonov[x][y] = tihonov_filter[x][y] * spectrum_4[x][y]

spectrum_reconstructed_image_tihonov = np.fft.ifft2(reconstructed_image_tihonov)

complete_reconstructed_image_tihoniv = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        complete_reconstructed_image_tihoniv[x][y] = np.abs(spectrum_reconstructed_image_tihonov[x][y])
        if spectrum_reconstructed_image_tihonov[x][y] < 0:
            complete_reconstructed_image_tihoniv[x][y] = 0
        elif spectrum_reconstructed_image_tihonov[x][y] > 255:
            complete_reconstructed_image_tihoniv[x][y] = 255
cv2.imshow('Tihonov`s filter', abs(complete_reconstructed_image_tihoniv).astype('uint8'))
cv2.waitKey(0)