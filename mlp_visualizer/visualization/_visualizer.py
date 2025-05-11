from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
                             QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
                             QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPen, QColor, QBrush, QPainter, QFont, QImage, QPixmap
import sys
import json
import numpy as np


class ZoomableGraphicsView(QGraphicsView):
    """Custom graphics view that supports mouse wheel zooming."""

    def wheelEvent(self, event):
        zoom_factor = 1.15

        # Zoom in or out
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)


class MLPVisualizer(QMainWindow):
    def __init__(self, json_data_path=None):
        super().__init__()
        self.setWindowTitle("MLP Architecture Visualization")
        self.resize(1200, 800)

        self.data = None
        self.current_pass = "1"
        self.max_pass = 1
        self._initial_fit_done = False
        self.setup_ui()

        # Load data if path is provided
        if json_data_path:
            self.load_data(json_data_path)
            self.visualize_network(preserve_current_view=False)

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Controls layout
        self.controls_layout = QHBoxLayout()
        self.main_layout.addLayout(self.controls_layout)

        # Set up pass slider
        self.controls_layout.addWidget(QLabel("Pass:"))
        self.pass_slider = QSlider(Qt.Orientation.Horizontal)
        self.pass_slider.setMinimum(1)
        self.pass_slider.setMaximum(1)  # Will be updated when data is loaded
        self.pass_slider.setValue(1)
        self.pass_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.pass_slider.setTickInterval(1)
        self.pass_slider.valueChanged.connect(self.change_pass)
        self.controls_layout.addWidget(self.pass_slider)

        # Zoom buttons
        self.controls_layout.addWidget(QPushButton("Zoom Out", clicked=self.zoom_out))
        self.controls_layout.addWidget(QPushButton("Zoom In", clicked=self.zoom_in))
        self.controls_layout.addStretch(1)

        # Content layout )
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # Image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.image_label)

        # Scene and view setup
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(Qt.GlobalColor.white))
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setViewport(QOpenGLWidget())
        self.content_layout.addWidget(self.view)

    def load_data(self, json_path):
        """Load the collected MLP data from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)

            # Get numeric passes and update slider
            numeric_passes = [int(k) for k in self.data.keys() if k.isdigit()]
            if numeric_passes:
                self.max_pass = max(numeric_passes)
                self.pass_slider.setMaximum(self.max_pass)
                self.pass_slider.setValue(1)
                self.current_pass = "1"
        except Exception as e:
            print(f"Error loading data: {e}")

    def change_pass(self, pass_value):
        """Change the visualization to show a different pass."""
        self.current_pass = str(pass_value)
        self.visualize_network(preserve_current_view=self._initial_fit_done)

    def zoom_in(self):
        """Zoom in the view."""
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        """Zoom out the view."""
        self.view.scale(0.8, 0.8)

    def visualize_network(self, preserve_current_view=True):
        """Visualize the MLP architecture based on the loaded data."""
        if not self.data:
            print("No data loaded.")
            return

        # Clear the scene
        self.scene.clear()
        architecture = self.data.get("architecture", {})

        # Identify and sort layers by their index
        all_layers = sorted([(k, v) for k, v in architecture.items()],
                            key=lambda x: int(x[0].split('_')[0]))

        linear_layers = [(k, v) for k, v in all_layers if "Linear" in k]

        neuron_radius = 20
        layer_spacing = 200
        neuron_spacing = 70

        layer_sizes = []
        for i, (layer_key, layer_info) in enumerate(linear_layers):
            if i == 0:
                in_features = layer_info.get("in_features", 5)
                out_features = layer_info.get(" out_features", 5)  # Note the space in key
                layer_sizes.append(int(in_features))
                layer_sizes.append(int(out_features))
            else:
                out_features = layer_info.get(" out_features", 2)  # Note the space in key
                layer_sizes.append(int(out_features))

        # Create and position neurons
        neurons = self.create_neurons(layer_sizes, layer_spacing, neuron_spacing, neuron_radius, linear_layers)

        # Draw connections (weights)
        self.draw_connections(neurons, linear_layers)

        # Add layer labels
        self.add_layer_labels(all_layers, layer_spacing)

        # Add pass title
        title = QGraphicsTextItem(f"Pass {self.current_pass}")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        title.setPos(10, 10)
        # title.setPos(self.scene.sceneRect().left() + 10 if not self.scene.sceneRect().isEmpty() else 10,
        #              self.scene.sceneRect().top() + 10 if not self.scene.sceneRect().isEmpty() else 10)
        self.scene.addItem(title)

        # Adjust view
        current_items_rect = self.scene.itemsBoundingRect()
        if not current_items_rect.isNull() and current_items_rect.isValid():
            padded_rect = current_items_rect.adjusted(-50, -50, 50, 50)
            self.scene.setSceneRect(padded_rect)
        elif not self.scene.items():  # Scene is empty
            self.scene.setSceneRect(0, 0, 1, 1)  # Minimal rect

        if not preserve_current_view or not self._initial_fit_done:
            if not self.scene.sceneRect().isEmpty() and self.scene.sceneRect().isValid():
                self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._initial_fit_done = True

        # Update image visualization
        self.visualize_input_image()

    def create_neurons(self, layer_sizes, layer_spacing, neuron_spacing, neuron_radius, linear_layers):
        """Create and position neurons for each layer."""
        neurons = []

        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_neurons = []
            x = layer_idx * layer_spacing + 100

            # Center the layer vertically
            layer_height = (layer_size - 1) * neuron_spacing
            y_offset = (600 - layer_height) / 2

            for neuron_idx in range(layer_size):
                y = y_offset + neuron_idx * neuron_spacing

                # Create neuron (circle)
                neuron = QGraphicsEllipseItem(x - neuron_radius, y - neuron_radius,
                                              2 * neuron_radius, 2 * neuron_radius)

                # Get activation value
                activation_value = self.get_activation_value(layer_idx, neuron_idx, linear_layers)

                # Set color based on activation
                if activation_value is not None:
                    intensity = min(255, int(abs(activation_value) * 200) + 50)
                    color = QColor(100, 100, 255, intensity) if activation_value >= 0 else QColor(255, 100, 100,
                                                                                                  intensity)
                    neuron.setBrush(QBrush(color))
                else:
                    neuron.setBrush(QBrush(QColor(200, 200, 200)))

                neuron.setPen(QPen(Qt.GlobalColor.black, 1))
                self.scene.addItem(neuron)

                # Add neuron label
                label_text = f"{layer_idx}.{neuron_idx}"
                if activation_value is not None:
                    label_text += f"\n{activation_value:.3f}"

                label = QGraphicsTextItem(label_text)
                label.setPos(x - neuron_radius - 10, y - neuron_radius - 35)
                self.scene.addItem(label)

                layer_neurons.append((neuron, QPointF(x, y)))

            neurons.append(layer_neurons)

        return neurons

    def get_activation_value(self, layer_idx, neuron_idx, linear_layers):
        """Get the activation value for a neuron if available."""
        if layer_idx >= len(linear_layers):
            return None

        layer_key = linear_layers[layer_idx][0]
        if self.current_pass not in self.data:
            return None

        layer_data = self.data[self.current_pass].get(layer_key, {})
        if 'input' in layer_data and neuron_idx < len(layer_data['input']):
            return layer_data['input'][neuron_idx]

        return None

    def draw_connections(self, neurons, linear_layers):
        """Draw connections (weights) between neurons."""
        for i in range(len(linear_layers)):
            layer_key, _ = linear_layers[i]

            # Get weight matrix for current pass
            current_pass_data = self.data.get(self.current_pass, {})
            layer_data = current_pass_data.get(layer_key, {})
            weight_matrix = layer_data.get("weight", [])

            if weight_matrix:
                for from_idx in range(len(neurons[i])):
                    for to_idx in range(len(neurons[i + 1])):
                        # Get weight if available
                        if to_idx < len(weight_matrix) and from_idx < len(weight_matrix[to_idx]):
                            weight = weight_matrix[to_idx][from_idx]

                            # Determine line thickness and color based on weight
                            weight_abs = abs(weight)
                            thickness = max(1, min(5, weight_abs * 5))
                            color = QColor(0, 0, 255, min(255, int(weight_abs * 200) + 50)) if weight >= 0 else QColor(
                                255, 0, 0, min(255, int(weight_abs * 200) + 50))

                            # Draw connection line
                            start_point = neurons[i][from_idx][1]
                            end_point = neurons[i + 1][to_idx][1]

                            line = QGraphicsLineItem(start_point.x(), start_point.y(),
                                                     end_point.x(), end_point.y())
                            line.setPen(QPen(color, thickness))
                            line.setZValue(-1)  # Put connections behind neurons
                            self.scene.addItem(line)

    def add_layer_labels(self, all_layers, layer_spacing):
        """Add labels for each layer."""
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(24)

        ind = 0
        layer_height = 0
        label_offset = 10

        for i, (layer_key, _) in enumerate(all_layers):
            if any(layer_type in layer_key for layer_type in [ "Linear", "ReLU", "tanh", "sigmoid"]):
                layer_key = layer_key.split("_")[1]
                x = ind * layer_spacing + 125
                y = layer_height + label_offset  # place below the layer
                ind += 1

                label = QGraphicsTextItem(layer_key)
                label.setFont(title_font)
                label.setPos(x, y)
                self.scene.addItem(label)

    def visualize_input_image(self):
        """Visualize the input image as grayscale."""
        if not self.data or self.current_pass not in self.data:
            return

        image_data = self.data[self.current_pass].get("input")
        if not image_data:
            return

        # Convert to normalized grayscale image
        image_array = np.array(image_data, dtype=np.float32)
        min_val = image_array.min()
        max_val = image_array.max()
        range_val = max_val - min_val

        if range_val != 0:
            image_array = (image_array - min_val) / range_val
        else:
            image_array = np.zeros_like(image_array)

        image_array = (image_array * 255).astype(np.uint8)

        # Create and display image
        h, w = image_array.shape[1:]
        qimage = QImage(image_array.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(
            200, 200, Qt.AspectRatioMode.KeepAspectRatio))

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        if self.data:
            self.visualize_network(preserve_current_view=False)


def visualize_mlp(json_path):
    """Create and show the MLP visualizer with the given data."""
    app = QApplication(sys.argv)
    visualizer = MLPVisualizer(json_path)
    visualizer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    visualize_mlp("../../example.json")