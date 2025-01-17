import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import numpy as np
import cv2
import torch
import configparser
from utils_lousa.cam_utils import adjust_contrast_brightness


# Escolhe entre CPU e GPU automaticamente
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Carregar o modelo YOLO
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/lousa-virtual-v8.pt', device=device)

# Nome e dimensões da janela
WINDOW_NAME = "Calibration Camera"
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480


# Contador de capturas
img_counter = 0


# Função vazia para os trackbars
def emtpy_callback(_):
    pass


# Inicializa a captura de vídeo da webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME)


# Dicionário com as configurações dos trackbars (facilita na na leitura dos valores e possibilita recuperá-los com a janela fechada)
settings = { # Valores padrão
    'Confidence': 5,
    'Gaussian Blur': 0, 
    'Contrast': 100,
    'Brightness': 0,
    'Sharpen': 0,
    'Unsharp Masking': 0
}


# Cria os trackbars para os filtros
cv2.createTrackbar('Confidence', WINDOW_NAME, 0, 10, emtpy_callback)
cv2.createTrackbar('Gaussian Blur', WINDOW_NAME, 0, 50, emtpy_callback)
cv2.createTrackbar('Contrast', WINDOW_NAME, 0, 200, emtpy_callback)
cv2.createTrackbar('Brightness', WINDOW_NAME, -255, 255, emtpy_callback)
cv2.createTrackbar('Sharpen', WINDOW_NAME, 0, 1, emtpy_callback)
cv2.createTrackbar('Unsharp Masking', WINDOW_NAME, 0, 1, emtpy_callback)

# Carrega as configurações do arquivo config.ini
config = configparser.ConfigParser()
try:
    config.read("config.ini")
    for key in settings:
        settings[key] = config["Settings"].getint(key)
    print("Config loaded")
except:
    print("Config not found")

# Define os valores iniciais dos trackbars
for key in settings:
    cv2.setTrackbarPos(key, WINDOW_NAME, settings[key])


# Loop principal
while True:
    # Lê o frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT)), 1)  # Espelha o frame horizontalmente
    
    # Pega as configurações dos trackbars
    for key in settings:
        settings[key] = cv2.getTrackbarPos(key, WINDOW_NAME)
    
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
        frame = cv2.filter2D(frame, -1, kernel)  # Aplica o filtro de nitidez
    
    if settings['Unsharp Masking']:
        # Kernel de Unsharp Masking
        kernel = (-1) * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
        frame = cv2.filter2D(frame, -1, kernel)  # Aplica o filtro de Unsharp Masking

    # Faz a detecção com YOLO
    original_frame = frame.copy()
    model.conf = settings['Confidence']/10.0 # Ajusta a Confidence
    results = model(frame)

    # Pega as detecções
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    # Itera sobre as detecções e desenha as caixas delimitadoras e os rótulos
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Coordenadas inteiras
        label = results.names[int(cls)]  # Nome da classe detectada

        # Desenhar a caixa delimitadora e o rótulo no frame
        color = (0, 255, 0)  # Verde para as caixas
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar o frame com as detecções
    cv2.imshow(WINDOW_NAME, frame)

    # Lê a tecla pressionada
    k = cv2.waitKey(1)
    if k%256 == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        # ESC pressionado ou janela fechada
        print("Exiting...")
        break
    elif k%256 == 32:
        # ESPAÇO pressionado
        img_name = f"opencv_frame_{img_counter}.png"
        # Salva para a pasta "captures"
        cv2.imwrite(f"{img_name}", original_frame)
        print(f"{img_name} saved.")
        img_counter += 1

# Salva as configurações no arquivo config.ini
config["Settings"] = settings
try:
    with open("config.ini", "w") as configfile:
        config.write(configfile)
    print("Config saved")
except:
    print("Error saving config")
    
    
# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()