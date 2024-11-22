import pyaudio
import wave


# DEFINIMOS LOS PARAMETROS

formato = pyaudio.paInt16
canales = 1
rate = 44100
chunk = 1024
duracion = 2
archivo = "Left.wav"


def recordchetao():
    a = 21
    b = a + 40
    for k in range(a, b):

        archivo = f"./audios/Segmentar/Segmentar{k}.wav"
        print(k)
        # SE INICIA PYAUDIO
        audio = pyaudio.PyAudio()
        print("\nSeleccione su dispositivo de entrada: \n")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ",
                      audio.get_device_info_by_host_api_device_index(0, i).get('name'))

        # num_entrada = int(input("\nNumero de entrada: "))
        num_entrada = 1

        stream = audio.open(format=formato, channels=canales, input_device_index=num_entrada,
                            rate=rate, input=True, frames_per_buffer=chunk)

        # SE INICIA LA GRABACION DE LA SEÑAL

        print("\nEscuchando...")
        frames = []
        for i in range(0, int(rate/chunk*duracion)):
            data = stream.read(chunk)
            frames.append(data)

        print("\nGrabación terminada.")

        # SE DETIENE LA GRABACION

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # SE CREA Y GUARDA EL ARCHIVO

        waveFile = wave.open(archivo, 'wb')
        waveFile.setnchannels(canales)
        waveFile.setsampwidth(audio.get_sample_size(formato))
        waveFile.setframerate(rate)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()


def record():
    # SE INICIA PYAUDIO
    audio = pyaudio.PyAudio()
    print("\nSeleccione su dispositivo de entrada: \n")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ",
                  audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    num_entrada = int(input("\nNumero de entrada: "))

    stream = audio.open(format=formato, channels=canales, input_device_index=num_entrada,
                        rate=rate, input=True, frames_per_buffer=chunk)

    # SE INICIA LA GRABACION DE LA SEÑAL

    print("\nEscuchando...")
    frames = []
    for i in range(0, int(rate/chunk*duracion)):
        data = stream.read(chunk)
        frames.append(data)

    print("\nGrabación terminada.")

    # SE DETIENE LA GRABACION

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # SE CREA Y GUARDA EL ARCHIVO

    waveFile = wave.open(archivo, 'wb')
    waveFile.setnchannels(canales)
    waveFile.setsampwidth(audio.get_sample_size(formato))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


recordchetao()
