import cv2
import numpy as np
import os

# Створення папки для результатів
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Завантаження зображення
input_image_path = os.path.expanduser("~/Desktop/1_binary.jpg")  # Шлях до зображення
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Не вдалося завантажити зображення. Перевірте шлях до файлу.")
    exit()

# Бінаризація зображення, якщо необхідно
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Морфологічні операції
kernel = np.ones((3, 3), np.uint8)  # Ядро 3x3 для операції

# 1. Ерозія
erosion = cv2.erode(binary_image, kernel, iterations=1)
cv2.imwrite(os.path.join(results_dir, "erosion.jpg"), erosion)

# 2. Нарощування
dilation = cv2.dilate(binary_image, kernel, iterations=1)
cv2.imwrite(os.path.join(results_dir, "dilation.jpg"), dilation)

# 3. Розмикання
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
cv2.imwrite(os.path.join(results_dir, "opening.jpg"), opening)

# 4. Замикання
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
cv2.imwrite(os.path.join(results_dir, "closing.jpg"), closing)

# 5. Границя (опціонально)
boundary = cv2.subtract(dilation, erosion)
cv2.imwrite(os.path.join(results_dir, "boundary.jpg"), boundary)

print(f"Результати збережено в папку '{results_dir}'.")
