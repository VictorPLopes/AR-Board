import cv2
import numpy as np

# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# https://www.bluetin.io/opencv/opencv-color-detection-filtering-python/
# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html


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

# Cria os trackbars para a cor vermelha
cv2.createTrackbar("LH_RED", "Lousa", 0, 179, emtpy_callback)
cv2.createTrackbar("LS_RED", "Lousa", 0, 255, emtpy_callback)
cv2.createTrackbar("UH_RED", "Lousa", 179, 179, emtpy_callback)
cv2.setTrackbarPos("LH_RED", "Lousa", 169)
cv2.setTrackbarPos("LS_RED", "Lousa", 108)

# Cria os trackbars para a cor verde
cv2.createTrackbar("LH_GREEN", "Lousa", 0, 179, emtpy_callback)
cv2.createTrackbar("LS_GREEN", "Lousa", 0, 255, emtpy_callback)
cv2.createTrackbar("UH_GREEN", "Lousa", 179, 179, emtpy_callback)
cv2.setTrackbarPos("LH_GREEN", "Lousa", 20)
cv2.setTrackbarPos("LS_GREEN", "Lousa", 45)
cv2.setTrackbarPos("UH_GREEN", "Lousa", 100)

# Cria os trackbars para a erosão e o número máximo de pontos
cv2.createTrackbar("EROSION", "Lousa", 0, 151, emtpy_callback)
cv2.setTrackbarPos("EROSION", "Lousa", 4)
cv2.createTrackbar("MAX_POINTS", "Lousa", 10, 150, emtpy_callback)
cv2.setTrackbarPos("MAX_POINTS", "Lousa", 60)


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
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Pega o valor da erosão
    erosion = cv2.getTrackbarPos("EROSION", "Lousa")

    # Pega o valor máximo de pontos
    max_points = cv2.getTrackbarPos("MAX_POINTS", "Lousa")

    # Filtro para a cor vermelha
    lh_red = cv2.getTrackbarPos("LH_RED", "Lousa")
    ls_red = cv2.getTrackbarPos("LS_RED", "Lousa")
    uh_red = cv2.getTrackbarPos("UH_RED", "Lousa")
    lower_red = np.array([lh_red, ls_red, 0])
    upper_red = np.array([uh_red, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # Detecta keypoints na máscara da cor vermelha
    red_res = cv2.erode(cv2.bitwise_and(frame, frame, mask=red_mask), None, iterations=erosion)
    inverted_red_mask = cv2.bitwise_not(red_mask)
    red_keypoints = detector.detect(inverted_red_mask)

    # Filtro para a cor verde
    lh_green = cv2.getTrackbarPos("LH_GREEN", "Lousa")
    ls_green = cv2.getTrackbarPos("LS_GREEN", "Lousa")
    uh_green = cv2.getTrackbarPos("UH_GREEN", "Lousa")
    lower_green = np.array([lh_green, ls_green, 0])
    upper_green = np.array([uh_green, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Detecta keypoints na máscara da cor verde
    res_green = cv2.erode(cv2.bitwise_and(frame, frame, mask=green_mask), None, iterations=erosion)
    inverted_green_mask = cv2.bitwise_not(green_mask)
    green_keypoints = detector.detect(inverted_green_mask)

    # Adiciona a média das coordenadas dos red_keypoints
    coord = coord_detect(red_keypoints)
    if coord is not None:
        red_draw.append(coord)
    # Se houver mais de 30 coordenadas, remove a primeira
    if len(red_draw) > max_points:
        red_draw.pop(0)

    # Adiciona a média das coordenadas dos red_keypoints
    coord = coord_detect(green_keypoints)
    if coord is not None:
        green_draw.append(coord)
    # Se houver mais de 30 coordenadas, remove a primeira
    if len(green_draw) > max_points:
        green_draw.pop(0)

    # Desenha círculos nas coordenadas da lista
    for coord in red_draw:
        cv2.circle(red_res, coord, 10, (0, 0, 255), -1)
    for coord in green_draw:
        cv2.circle(res_green, coord, 10, (0, 255, 0), -1)

    combined_res = cv2.bitwise_or(red_res, res_green)
    cv2.imshow("Lousa", cv2.flip(combined_res, 1))

    # Limpa as listas de coordenadas ao pressionar a tecla "c"
    if cv2.waitKey(1) & 0xFF == ord("c"):
        red_draw.clear()
        green_draw.clear()
    
    # Sai do loop ao pressionar a tecla "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
