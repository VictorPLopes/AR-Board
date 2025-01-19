import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
from board_utils.common_utils import *

device = get_device()
print(f"Using device: {device}")

model = load_yolo_model('models/lousa-virtual-v10.pt', device)
WINDOW_NAME = "AR Board"

# Inicializa configurações e trackbars
settings = load_config("config.ini", DEFAULT_SETTINGS)
cv2.namedWindow(WINDOW_NAME)
create_trackbars(WINDOW_NAME, settings)

board = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
last_marker, last_pencil = None, None

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT)), 1)

    # Atualiza configurações dos trackbars
    for key in settings:
        settings[key] = cv2.getTrackbarPos(key, WINDOW_NAME)

    frame = apply_filters(frame, settings)
    model.conf = settings['Confidence'] / 10.0
    results = model(frame)

    # Manipula as detecções
    detections = results.xyxy[0].cpu().numpy()
    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        coords = ((x1 + x2) // 2, (y1 + y2) // 2)
        label = results.names[int(cls)]
        if label == 'pencil':
            if last_pencil and np.linalg.norm(np.array(coords) - np.array(last_pencil)) < 50:
                cv2.line(board, last_pencil, coords, (128, 128, 128), 5)
            else:
                cv2.circle(board, coords, 5, (128, 128, 128), -1)
            last_pencil = coords
        elif label == 'marker':
            if last_marker and np.linalg.norm(np.array(coords) - np.array(last_marker)) < 50:
                cv2.line(board, last_marker, coords, (255, 0, 0), 5)
            else:
                cv2.circle(board, coords, 5, (255, 0, 0), -1)
            last_marker = coords
        elif label == 'eraser':
            cv2.rectangle(board, (x1, y1), (x2, y2), (0, 0, 0), -1)

    combined_frame = cv2.addWeighted(frame, 0.5, board, 0.5, 0)
    cv2.imshow(WINDOW_NAME, combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

save_config("config.ini", settings)
cap.release()
cv2.destroyAllWindows()
