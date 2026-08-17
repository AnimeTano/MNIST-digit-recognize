import sys
import torch
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap
import torch.nn as nn
import torch.nn.functional as F


# ---------- Модель (такая же, как при обучении) ----------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------- Виджет для рисования ----------
class DrawWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setMaximumSize(280, 280)
        self.setStyleSheet("background-color: black;")
        self.pixmap = QPixmap(280, 280)
        self.pixmap.fill(Qt.GlobalColor.black)
        self.last_point = QPoint()
        self.drawing = False
        self.pen_width = 12
        self.pen_color = Qt.GlobalColor.white

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.draw_point(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.draw_line(self.last_point, event.pos())
            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def draw_point(self, point):
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPoint(point)
        self.update()

    def draw_line(self, start, end):
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)
        self.update()

    def clear(self):
        self.pixmap.fill(Qt.GlobalColor.black)
        self.update()

    def get_image(self):
        # Конвертируем QPixmap в QImage, затем в numpy
        qimage = self.pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        gray = arr[:, :, 0]  # берём красный канал (все каналы одинаковы)
        return gray


# ---------- Основное окно ----------
class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Распознавание рукописных цифр")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Виджет рисования
        self.draw_widget = DrawWidget()
        layout.addWidget(self.draw_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Кнопки
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.clear_canvas)
        predict_btn = QPushButton("Предсказать")
        predict_btn.clicked.connect(self.predict_digit)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(predict_btn)
        layout.addLayout(btn_layout)

        # Метка для результата
        self.result_label = QLabel("Нарисуйте цифру и нажмите 'Предсказать'")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.result_label)

        self.setFixedSize(320, 400)

    def clear_canvas(self):
        self.draw_widget.clear()
        self.result_label.setText("Нарисуйте цифру и нажмите 'Предсказать'")

    def predict_digit(self):
        # Получаем изображение с холста (280x280)
        gray = self.draw_widget.get_image()
        # Преобразуем в PIL Image, ресайзим до 28x28
        img_pil = Image.fromarray(gray.astype(np.uint8))
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)

        # Инвертируем, если необходимо (MNIST: белая цифра на чёрном)
        # На холсте рисуем белым на чёрном – ничего не меняем.
        # Если вы рисуете чёрным на белом, раскомментируйте строку ниже:
        # img_pil = Image.eval(img_pil, lambda x: 255 - x)

        # Преобразуем в numpy и нормализуем
        arr = np.array(img_pil, dtype=np.float32) / 255.0
        # Добавляем размерности: (1, 1, 28, 28)
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

        # Предсказание
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probabilities[0][pred].item() * 100

        self.result_label.setText(f"Предсказано: {pred}  (уверенность: {confidence:.2f}%)")


# ---------- Загрузка модели ----------
def load_model(path='notebooks/mnist_cnn.pth'):
    model = CNN()
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print("Модель загружена успешно.")
    except FileNotFoundError:
        print(f"Файл {path} не найден. Убедитесь, что модель сохранена.")
        sys.exit(1)
    return model


# ---------- Запуск приложения ----------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = load_model()  # если файл называется иначе, измените имя
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())