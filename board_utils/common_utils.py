import cv2
import configparser
import numpy as np
import torch

# Constantes compartilhadas
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

# Função para configurar o dispositivo (CPU/GPU)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Função para carregar o modelo YOLO
def load_yolo_model(path, device):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=path, device=device)

# Configuração inicial para trackbars
DEFAULT_SETTINGS = {
    'Confidence': 5,
    'Gaussian Blur': 0, 
    'Contrast': 100,
    'Brightness': 0,
    'Sharpen': 0,
    'Unsharp Masking': 0
}

# Função para criar trackbars
def create_trackbars(window_name, settings):
    for key, (min_val, max_val) in {
        'Confidence': (0, 10),
        'Gaussian Blur': (0, 50),
        'Contrast': (0, 200),
        'Brightness': (-255, 255),
        'Sharpen': (0, 1),
        'Unsharp Masking': (0, 1)
    }.items():
        cv2.createTrackbar(key, window_name, settings.get(key, 0), max_val, lambda _: None)

# Função para carregar configurações de um arquivo
def load_config(config_file, settings):
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        for key in settings:
            settings[key] = config["Settings"].getint(key)
        print("Config loaded")
    except Exception as e:
        print(f"Config not found: {e}")
    return settings

# Função para salvar configurações em um arquivo
def save_config(config_file, settings):
    config = configparser.ConfigParser()
    config["Settings"] = {k: str(v) for k, v in settings.items()}
    try:
        with open(config_file, "w") as configfile:
            config.write(configfile)
        print("Config saved")
    except Exception as e:
        print(f"Error saving config: {e}")
        
# Função para ajustar contraste e brilho
def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

# Função para aplicar filtros no frame
def apply_filters(frame, settings):
    # Reduz o ruído
    if settings['Gaussian Blur'] > 0:
        if settings['Gaussian Blur'] % 2 == 0:
            settings['Gaussian Blur'] += 1  # Apenas números ímpares
        frame = cv2.GaussianBlur(frame, (settings['Gaussian Blur'], settings['Gaussian Blur']), 0)

    # Ajusta o contraste e o brilho
    frame = adjust_contrast_brightness(frame, contrast=settings['Contrast']/100.0, brightness=settings['Brightness'])
    
    if settings['Sharpen']:
        # Kernel de nitidez
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        frame = cv2.filter2D(frame, -1, kernel)

    if settings['Unsharp Masking']:
        # Kernel de Unsharp Masking
        kernel = (-1) * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
        frame = cv2.filter2D(frame, -1, kernel)

    return frame
