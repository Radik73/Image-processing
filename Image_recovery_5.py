import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Distortion_and_Noise_3


'''Восстановление изображений при помощи итерационного алгоритма)'''

height = np.size(Distortion_and_Noise_3.cutting, 0)
width = np.size(Distortion_and_Noise_3.cutting, 1)

frp_for_filter = np.zeros((width, height))
for i in range(width):
    for j in range(height):
        frp_for_filter[i][j] =  Distortion_and_Noise_3.added_frp_smooth[i][j]

spectrum_3 = np.fft.fft2(frp_for_filter) #Двумерное преобразование Фурье
spectrum_3 = spectrum_3 / (np.sqrt(height * width))


H = np.zeros((width, height), dtype=np.complex)
H =  spectrum_3 * width


'''Спектр искажённого изображения'''

spectrum_4 = np.fft.fft2(Distortion_and_Noise_3.cutting) #Двумерное преобразование Фурье


'''Реализация итерационного алгоритма'''

HSS = np.zeros((width, height), dtype = np.complex)
for i in range(width):
    for j in range(height):
        HSS[i][j] = np.conj(H[i][j]) * spectrum_4[i][j]

H1H = np.zeros((width, height))
for i in range(width):
    for j in range(height):
        H1H[i][j] = 1 - H[i][j] * np.conj(H[i][j])

print('Введите колличество итераций')
N1 = int(input())
spectrum_reconstructed_iteration_algorithm = np.zeros((width, height), dtype = np.complex)
B = HSS.copy()
for iter in range(N1 + 1):
    for i in range(width):
        for j in range(height):
            B[i][j] = HSS[i][j] + H1H[i][j] * B[i][j]
reconstructed_iteration_algorithm = np.fft.ifft2(B)


complete_reconstructed_iteration_algorithm = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        complete_reconstructed_iteration_algorithm[x][y] = np.abs(reconstructed_iteration_algorithm[x][y])
        if reconstructed_iteration_algorithm[x][y] < 0:
            complete_reconstructed_iteration_algorithm[x][y] = 0
        elif reconstructed_iteration_algorithm[x][y] > 255:
            complete_reconstructed_iteration_algorithm[x][y] = 255
cv2.imshow('Iteration algorithm filter', abs(complete_reconstructed_iteration_algorithm).astype('uint8'))
cv2.waitKey(0)