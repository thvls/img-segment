import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QColorDialog, QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen
from PyQt5.QtCore import Qt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import torch

# Placeholder for the segmentation model (initialize with actual SAM2 setup later)
class SAM2Model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        self.checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        self.config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.model = build_sam2(self.config, self.checkpoint, device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def auto_segment(self, image):
        # Use SAM2 to perform auto-segmentation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask["segmentation"].astype(np.uint8))
        return combined_mask

    def segment_with_boxes(self, image, boxes):
        # Use SAM2 to segment with multiple bounding boxes
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        formatted_boxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        predictor = SAM2ImagePredictor(self.model)
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(box=formatted_boxes, multimask_output=False)
        
        return masks

class ImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation Tool")
        self.setGeometry(100, 100, 800, 600)

        self.model = SAM2Model()
        self.image = cv2.imread("data/federer.jpg")
        self.masks = []
        self.boxes = []
        self.drawing = False
        self.current_box = None

        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Image display
        self.image = cv2.imread("data/federer.jpg")
        self.image_label = QLabel("Load an image to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.start_drawing
        self.image_label.mouseMoveEvent = self.update_drawing
        self.image_label.mouseReleaseEvent = self.finish_drawing
        main_layout.addWidget(self.image_label)

        # Buttons
        button_layout = QHBoxLayout()

        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        button_layout.addWidget(load_button)

        auto_segment_button = QPushButton("Auto-Segment")
        auto_segment_button.clicked.connect(self.auto_segment)
        button_layout.addWidget(auto_segment_button)

        draw_box_button = QPushButton("Draw Boxes")
        draw_box_button.clicked.connect(self.enable_box_drawing)
        button_layout.addWidget(draw_box_button)

        segment_boxes_button = QPushButton("Segment Boxes")
        segment_boxes_button.clicked.connect(self.segment_boxes)
        button_layout.addWidget(segment_boxes_button)

        fill_color_button = QPushButton("Fill Segment with Color")
        fill_color_button.clicked.connect(self.fill_with_color)
        button_layout.addWidget(fill_color_button)

        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        button_layout.addWidget(save_button)

        main_layout.addLayout(button_layout)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)

        if self.boxes:
            painter = QPainter(pixmap)
            pen = QPen(Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)
            for box in self.boxes:
                x, y, w, h = box
                painter.drawRect(x, y, w, h)
            painter.end()

        self.image_label.setPixmap(pixmap)

    def auto_segment(self):
        if self.image is not None:
            combined_mask = self.model.auto_segment(self.image)
            self.masks = [combined_mask]
            self.show_masks()

    def enable_box_drawing(self):
        self.boxes = []
        self.update()

    def start_drawing(self, event):
        if self.image is not None:
            self.drawing = True
            self.current_box = (event.pos().x(), event.pos().y(), 0, 0)

    def update_drawing(self, event):
        if self.drawing and self.image is not None:
            x, y, _, _ = self.current_box
            w = event.pos().x() - x
            h = event.pos().y() - y
            self.current_box = (x, y, w, h)
            self.display_image(self.image)

    def finish_drawing(self, event):
        if self.drawing and self.image is not None:
            self.drawing = False
            self.boxes.append(self.current_box)
            self.current_box = None
            self.display_image(self.image)

    def segment_boxes(self):
        if self.image is not None and self.boxes:
            print("Segmenting with boxes:", self.boxes)
            self.masks = self.model.segment_with_boxes(self.image, self.boxes)
            print("Generated masks:", len(self.masks))
            self.show_masks()

    def fill_with_color(self):
        if self.masks is not None:
            color = QColorDialog.getColor()
            if color.isValid():
                r, g, b, _ = color.getRgb()
                overlay = np.zeros_like(self.image)
                for mask in self.masks:
                    overlay[mask > 0] = (b, g, r)
                self.image = cv2.addWeighted(self.image, 0.5, overlay, 0.5, 0)
                self.display_image(self.image)

    def show_masks(self):
        if self.masks is not None:
            scene = QGraphicsScene()

            # Display original image
            original_image_pixmap = QPixmap(self.image_label.pixmap())
            original_item = QGraphicsPixmapItem(original_image_pixmap)
            scene.addItem(original_item)

            # Overlay segmentation masks
            for idx, mask in enumerate(self.masks):
                if len(mask.shape) == 3:
                    _, height, width = mask.shape # First dimension carried over from 'masks', ignore it.
                else:
                    height, width = mask.shape
                mask_image = QImage(mask.data, width, height, QImage.Format_Grayscale8)
                mask_pixmap = QPixmap.fromImage(mask_image)
                x_offset = idx * width
                mask_item = QGraphicsPixmapItem(mask_pixmap)
                mask_item.setOffset(x_offset, 0)
                scene.addItem(mask_item)

            view = QGraphicsView(scene)
            view.show()

    def save_image(self):
        if self.image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, self.image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ImageSegmentationApp()
    main_window.show()
    sys.exit(app.exec_())
