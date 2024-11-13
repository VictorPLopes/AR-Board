import cv2
import numpy as np
import configparser

# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# https://www.bluetin.io/opencv/opencv-color-detection-filtering-python/
# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html


# Função vazia para os trackbars
def emtpy_callback(_):
    pass


# Função para calcular a média das coordenadas dos keypoints
def coord_detect(keypoints):
    x = 0
    y = 0
    for keypoint in keypoints:
        x += keypoint.pt[0]
        y += keypoint.pt[1]
    if len(keypoints) > 0:
        x /= len(keypoints)
        y /= len(keypoints)
        # Retorna a média das coordenadas
        # print(int(x), int(y))
        return (int(x), int(y))
    return None


cap = cv2.VideoCapture(0)
cv2.namedWindow("Lousa")

# Dicionário com as configurações dos trackbars (facilita na na leitura dos valores e possibilita recuperá-los com a janela fechada)
settings = { # Valores padrão
    "LH_RED": 169,
    "LS_RED": 108,
    "UH_RED": 0,
    "LH_GREEN": 20,
    "LS_GREEN": 45,
    "UH_GREEN": 100,
    "EROSION": 4,
    "MAX_POINTS": 60
}

# Cria os trackbars para a cor vermelha
cv2.createTrackbar("LH_RED", "Lousa", 0, 179, emtpy_callback)
cv2.createTrackbar("LS_RED", "Lousa", 0, 255, emtpy_callback)
cv2.createTrackbar("UH_RED", "Lousa", 179, 179, emtpy_callback)

# Cria os trackbars para a cor verde
cv2.createTrackbar("LH_GREEN", "Lousa", 0, 179, emtpy_callback)
cv2.createTrackbar("LS_GREEN", "Lousa", 0, 255, emtpy_callback)
cv2.createTrackbar("UH_GREEN", "Lousa", 179, 179, emtpy_callback)

# Cria os trackbars para a erosão e o número máximo de pontos
cv2.createTrackbar("EROSION", "Lousa", 0, 151, emtpy_callback)
cv2.createTrackbar("MAX_POINTS", "Lousa", 10, 150, emtpy_callback)


# Carrega as configurações do arquivo config.ini
config = configparser.ConfigParser()
try:
    config.read("config.ini")
    for key in settings:
        settings[key] = config["Settings"].getint(key)
    print("Config Carregado")
except:
    print("Config não encontrado")

# Define os valores iniciais dos trackbars
for key in settings:
    cv2.setTrackbarPos(key, "Lousa", settings[key])

# Parâmetros para o detector de blobs
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1000
params.maxArea = 1000000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)

# Lista de coordenadas para desenhar círculos
red_draw = []
green_draw = []

while True:    
    # Lê o frame da câmera
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Pega as configurações dos trackbars
    for key in settings:
        settings[key] = cv2.getTrackbarPos(key, "Lousa")

    # Filtro para a cor vermelha
    lower_red = np.array([settings["LH_RED"], settings["LS_RED"], 0])
    upper_red = np.array([settings["UH_RED"], 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # Detecta keypoints na máscara da cor vermelha
    red_res = cv2.erode(
        cv2.bitwise_and(frame, frame, mask=red_mask), None, iterations=settings["EROSION"]
    )
    inverted_red_mask = cv2.bitwise_not(red_mask)
    red_keypoints = detector.detect(inverted_red_mask)

    # Filtro para a cor verde
    lower_green = np.array([settings["LH_GREEN"], settings["LS_GREEN"], 0])
    upper_green = np.array([settings["UH_GREEN"], 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Detecta keypoints na máscara da cor verde
    res_green = cv2.erode(
        cv2.bitwise_and(frame, frame, mask=green_mask), None, iterations=settings["EROSION"]
    )
    inverted_green_mask = cv2.bitwise_not(green_mask)
    green_keypoints = detector.detect(inverted_green_mask)

    # Adiciona a média das coordenadas dos red_keypoints
    coord = coord_detect(red_keypoints)
    if coord is not None:
        red_draw.append(coord)
    # Se houver mais de 30 coordenadas, remove a primeira
    if len(red_draw) > settings["MAX_POINTS"]:
        red_draw.pop(0)

    # Adiciona a média das coordenadas dos red_keypoints
    coord = coord_detect(green_keypoints)
    if coord is not None:
        green_draw.append(coord)
    # Se houver mais de 30 coordenadas, remove a primeira
    if len(green_draw) > settings["MAX_POINTS"]:
        green_draw.pop(0)

    # Desenha círculos nas coordenadas da lista
    for coord in red_draw:
        cv2.circle(red_res, coord, 10, (0, 0, 255), -1)
    for coord in green_draw:
        cv2.circle(res_green, coord, 10, (0, 255, 0), -1)

    combined_res = cv2.bitwise_or(red_res, res_green)
    cv2.imshow("Lousa", cv2.flip(combined_res, 1))


    # Lê a tecla pressionada
    key = cv2.waitKey(1)
    
    # Limpa as listas de coordenadas ao pressionar a tecla "c"
    if key == ord("c"):
        red_draw.clear()
        green_draw.clear()

    # Sai do loop ao pressionar a tecla "q"
    elif key == ord("q") or cv2.getWindowProperty('Lousa',cv2.WND_PROP_VISIBLE) < 1:
        break

# Salva as configurações no arquivo config.ini
config["Settings"] = {
    "LH_RED": settings["LH_RED"],
    "LS_RED": settings["LS_RED"],
    "UH_RED": settings["UH_RED"],
    "LH_GREEN": settings["LH_GREEN"],
    "LS_GREEN": settings["LS_GREEN"],
    "UH_GREEN": settings["UH_GREEN"],
    "EROSION": settings["EROSION"],
    "MAX_POINTS": settings["MAX_POINTS"]
}
try:
    with open("config.ini", "w") as configfile:
        config.write(configfile)
    print("Config Salvo")
except:
    print("Erro ao salvar config")

cap.release()
cv2.destroyAllWindows()
