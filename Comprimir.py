from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, 
    QLabel, QLineEdit, QPushButton, QGridLayout, QMessageBox,QFileDialog,QFrame
)

import cv2
import numpy as np

from scipy.fftpack import dct, idct

import os


def comprimir_imagen(self):
     """
    Comprime la imagen cargada con el porcentaje especificado y la muestra en la interfaz.
    """
     if self.imagen is not None:
        try:
            # Obtener el porcentaje de compresión
            porcentaje_compresion = [int(self.compression_input_img.text()),80,50,25]
            porcentaje_compresion[0]=self.intvalue(porcentaje_compresion[0])
            porcentaje_compresion[1]=self.intvalue(porcentaje_compresion[1])
            porcentaje_compresion[2]=self.intvalue(porcentaje_compresion[2])
            porcentaje_compresion[3]=self.intvalue(porcentaje_compresion[3])
            if porcentaje_compresion[0] < 0 or porcentaje_compresion[0] > 100:
                raise ValueError("El porcentaje debe estar entre 0 y 100.")
            
            # Aplicar DCT tipo 2
            dct_tipo2 = dct(self.imagen, type=2, norm='ortho')
            
            # Comprimir la imagen según el porcentaje
            _, dct_comprimidainput = self.calcular_porcentaje(dct_tipo2, porcentaje_compresion[0])
            _, dct_comprimida80 = self.calcular_porcentaje(dct_tipo2, porcentaje_compresion[1])
            _, dct_comprimida50 = self.calcular_porcentaje(dct_tipo2, porcentaje_compresion[2])
            _, dct_comprimida25 = self.calcular_porcentaje(dct_tipo2, porcentaje_compresion[3])
            
            # Aplicar IDCT para descomprimir la imagen
            idct_tipo2_input = idct(dct_comprimidainput, type=2, norm='ortho')
            idct_tipo2_input = np.clip(idct_tipo2_input, 0, 255).astype(np.uint8) 
            idct_tipo2_80 = idct(dct_comprimida80, type=2, norm='ortho')
            idct_tipo2_80 = np.clip(idct_tipo2_80, 0, 255).astype(np.uint8)  
            idct_tipo2_50 = idct(dct_comprimida50, type=2, norm='ortho')
            idct_tipo2_50 = np.clip(idct_tipo2_50, 0, 255).astype(np.uint8)  
            idct_tipo2_25 = idct(dct_comprimida25, type=2, norm='ortho')
            idct_tipo2_25 = np.clip(idct_tipo2_25, 0, 255).astype(np.uint8)   
            idcts=[idct_tipo2_input,idct_tipo2_80,idct_tipo2_50,idct_tipo2_25]
            # Mostrar la imagen descomprimida en la interfaz

            idcts = [idct_tipo2_input, idct_tipo2_80, idct_tipo2_50, idct_tipo2_25]
            nombres = [
                "imagen_comprimida_input.jpg",
                "imagen_comprimida_80.jpg",
                "imagen_comprimida_50.jpg",
                "imagen_comprimida_25.jpg"
            ]
            
            # Guardar imágenes comprimidas en almacenamiento local
            for idx, imagen in enumerate(idcts):
                cv2.imwrite(nombres[idx], imagen)

            # Mostrar la imagen descomprimida en la interfaz
            self.mostrar_imagenes(idcts)

            # Mostrar pesos de las imágenes guardadas
            pesos = [os.path.getsize(nombre) for nombre in nombres]
            pesos_texto = "\n".join(f"{nombres[i]}: {pesos[i]} bytes" for i in range(len(nombres)))
            QMessageBox.information(self, "Información", f"Imágenes comprimidas y guardadas correctamente.\n\n{pesos_texto}")

            QMessageBox.information(self, "Información", "Imagen comprimida y descomprimida correctamente.")
        
        except ValueError as ve:
            QMessageBox.critical(self, "Error", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al comprimir la imagen: {str(e)}")
     else:
        QMessageBox.warning(self, "Advertencia", "Por favor, carga una imagen antes de comprimirla.")