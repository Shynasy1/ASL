import numpy as np
import torch
import cv2
from ultralytics import YOLO

# Загрузите модель YOLO
model_path = '/Users/perfection/LASTCHANCE/Model/best.pt'
try:
    model = YOLO(model_path)
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit()

# Попробуйте открыть веб-камеру
cap = None
for i in range(3):  # Попробуйте индексы 0, 1, 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Веб-камера успешно открыта с индексом {i}.")
        break
    else:
        cap.release()
        cap = None

if not cap or not cap.isOpened():
    print("Не удалось открыть веб-камеру.")
    exit()

# Настройки фрейма веб-камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():

    # Захват кадра, если все прошло хорошо, то 'ret==True'
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр.")
        break

    # Выполните детекции
    try:
        results = model(frame)
        print("Детекции выполнены.")
    except Exception as e:
        print(f"Ошибка при выполнении детекций: {e}")
        break

    # Отобразите детекции
    try:
        annotated_frame = results[0].plot()
        print("Кадр аннотирован.")
    except Exception as e:
        print(f"Ошибка при аннотировании кадра: {e}")
        break

    # Покажите детекции
    cv2.imshow('YOLO', annotated_frame)

    # Если мы нажмем кнопку выхода 'q', завершится захват с веб-камеры
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закройте все в конце
cap.release()
cv2.destroyAllWindows()
print("Программа завершена.")
