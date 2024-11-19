import tkinter as tk
from tkinter import Label, Button, Scale, filedialog, HORIZONTAL
import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
from cv2 import imread, cvtColor, COLOR_BGR2RGB

# Funciones para compresión de imagen
from numpy import zeros, array, cos, pi, sqrt

cos_backup = array([])

def cos_values(N):
    ret = zeros((N, N))
    for n in range(len(ret)):
        for k in range(len(ret[n])):
            ret[k, n] = cos(((pi * k) * (2 * n + 1)) / (2 * N))
    global cos_backup
    cos_backup = ret

def direct_dct(vector):
    N = len(vector)
    if len(cos_backup) != N:
        cos_values(N)
    vector = cos_backup.dot(vector)
    vector[0] = vector[0] * sqrt(1 / 2)
    vector = vector * sqrt(2 / N)
    return vector

def inverse_dct(vector):
    N = len(vector)
    if len(cos_backup) != N:
        cos_values(N)
    vector[0] = vector[0] * sqrt(1 / 2)
    vector = vector * sqrt(2 / N)
    return cos_backup.T.dot(vector)

def direct_dct_2d(matrix):
    Nx, Ny = matrix.shape
    for line in range(Nx):
        matrix[line] = direct_dct(matrix[line])
    for column in range(Ny):
        matrix[:, column] = direct_dct(matrix[:, column])
    return matrix

def inverse_dct_2d(matrix):
    Nx, Ny = matrix.shape
    for column in range(Ny):
        matrix[:, column] = inverse_dct(matrix[:, column])
    for line in range(Nx):
        matrix[line] = inverse_dct(matrix[line])
    return matrix

def direct_dct_image(img):
    if img.shape[2] == 3:
        for i in range(3):
            img[:, :, i] = direct_dct_2d(img[:, :, i])
    else:
        img[:, :, 0] = direct_dct_2d(img[:, :, 0])
    return img

def inverse_dct_image(img):
    if img.shape[2] == 3:
        for i in range(3):
            img[:, :, i] = inverse_dct_2d(img[:, :, i])
    else:
        img[:, :, 0] = inverse_dct_2d(img[:, :, 0])
    return img.clip(0, 255)

def remove_coeficients_from_image(img, keep):
    img_new = np.zeros(img.shape)
    for i in range(keep * 3):  # * 3, because 3 color channels
        index = np.unravel_index(np.absolute(img).argmax(), img.shape)
        img_new[index] = img[index]  # copy it over to new image
        img[index] = 0  # remove from original so we don't count it again
    return img_new

def compress_image():
    global img, percentage
    total_coefficients = img.shape[0] * img.shape[1]
    num_coefficients = int(total_coefficients - ((percentage / 100) * total_coefficients))
    print(f"Manteniendo {num_coefficients} coeficientes de {total_coefficients} ({percentage}%).")

    rows = 1  
    columns = 4

    fig = plt.figure(figsize=(20, 10))

    fig.add_subplot(rows, columns, 1)
    plt.imshow(cvtColor(img.astype('uint8'), COLOR_BGR2RGB))

    fig.add_subplot(rows, columns, 2)
    x = direct_dct_image(img.copy())
    plt.imshow(cvtColor(x.astype('uint8'), COLOR_BGR2RGB))

    fig.add_subplot(rows, columns, 3)
    y = remove_coeficients_from_image(x.copy(), num_coefficients)
    plt.imshow(cvtColor(y.astype('uint8'), COLOR_BGR2RGB))

    fig.add_subplot(rows, columns, 4)
    plt.imshow(cvtColor(inverse_dct_image(y).astype('uint8'), COLOR_BGR2RGB))

    plt.show()

def open_image():
    global img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = imread(file_path)
        img = img.astype('float64')
        print(f"Imagen {file_path} cargada con éxito.")

def update_percentage(val):
    global percentage
    percentage = int(val)

def volver(root, main):
    root.destroy()
    main.deiconify()

def Comprimir(main):
    global img, percentage
    percentage = 50  # Valor por defecto

    main.withdraw()

    root = tk.Tk()
    root.title("Compresor de Imágenes usando DCT")

    label = Label(root, text="Seleccione una imagen y ajuste el porcentaje de compresión:")
    label.pack()

    open_button = Button(root, text="Abrir Imagen", command=open_image)
    open_button.pack()

    scale = Scale(root, from_=0, to=100, orient=HORIZONTAL, command=update_percentage, label="Porcentaje de Compresión")
    scale.set(percentage)
    scale.pack()

    compress_button = Button(root, text="Comprimir Imagen", command=compress_image)
    compress_button.pack()

    boton_volver = tk.Button(root, text="Volver", command=lambda: volver(root, main))
    boton_volver.pack(pady=10)

    root.mainloop()