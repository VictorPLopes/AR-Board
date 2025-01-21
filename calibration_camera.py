import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import cv2
from board_utils.common_utils import *

device = get_device()
print(f"Using device: {device}")

model = load_yolo_model('models/lousa-virtual-v12.pt', device)
WINDOW_NAME = "Calibration Camera"

settings = load_config("config.ini", DEFAULT_SETTINGS)
cv2.namedWindow(WINDOW_NAME)
create_trackbars(WINDOW_NAME, settings)

cap = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT)), 1)

    for key in settings:
        settings[key] = cv2.getTrackbarPos(key, WINDOW_NAME)

    frame = apply_filters(frame, settings)
    og_frame = frame.copy()
    model.conf = settings['Confidence'] / 10.0
    results = model(frame)

    detections = results.xyxy[0].cpu().numpy()
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = results.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)
    k = cv2.waitKey(1)
    if k % 256 == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:  # ESC
        break
    elif k % 256 == 32:  # SPACE
        cv2.imwrite(f"calibration_{img_counter}.png", og_frame)
        img_counter += 1

save_config("config.ini", settings)
cap.release()
cv2.destroyAllWindows()
