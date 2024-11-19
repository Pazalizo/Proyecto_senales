import pyaudio

# Inicializa PyAudio
audio = pyaudio.PyAudio()

# Obtén y muestra todos los dispositivos
print("\nDispositivos de entrada disponibles:")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    device_info = audio.get_device_info_by_host_api_device_index(0, i)
    if device_info.get('maxInputChannels') > 0:
        print(f"ID: {i} - Nombre: {device_info.get('name')} - Canales: {device_info.get('maxInputChannels')}")

audio.terminate()
