from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
                             QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
                             QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel, QGraphicsRectItem)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPen, QColor, QBrush, QPainter, QFont, QImage, QPixmap
import pyqtgraph as pg
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


def get_diverging_neuron_color(activation_value):
    """Maps an activation value to a diverging color scheme (blue-white-red)."""
    normalized_value = np.clip(activation_value, -1.0, 1.0)  # Clip to a reasonable range

    if normalized_value >= 0:
        # Map [0, 1] to [white, red]
        intensity = int(normalized_value * 255)
        red = 255
        green = 255 - intensity
        blue = 255 - intensity
        alpha = 255
    else:
        # Map [-1, 0) to [blue, white)
        intensity = int(abs(normalized_value) * 255)
        red = 255 - intensity
        green = 255 - intensity
        blue = 255
        alpha = 255

    return QColor(red, green, blue, alpha)

class MLPVisualizer(QMainWindow):
    def __init__(self, json_data_path=None):
        super().__init__()
        self.setWindowTitle("MLP Architecture Visualization")
        self.resize(1200, 800)

        self.data = None
        self.current_pass = "1"
        self.max_pass = 1
        self._initial_fit_done = False
        self.all_metrics_data = []  # Stores tuples: (pass_num, loss, accuracy)
        self.setup_ui()

        # Load data if path is provided
        if json_data_path:
            self.load_data(json_data_path)
            self.visualize_network(preserve_current_view=False)
            self.update_plot()
            self.update_architecture_label(self.data['architecture'])

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

        # Pass label
        self.pass_display_label = QLabel("Pass: 1/1")  # Initial text
        self.controls_layout.addWidget(self.pass_display_label)

        # Zoom buttons
        self.controls_layout.addWidget(QPushButton("Zoom Out", clicked=self.zoom_out))
        self.controls_layout.addWidget(QPushButton("Zoom In", clicked=self.zoom_in))
        self.controls_layout.addStretch(1)

        # Content layout
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # Image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.image_label)

        # Scene and view setup
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(Qt.GlobalColor.darkGray))
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setViewport(QOpenGLWidget())
        self.content_layout.addWidget(self.view)

        # Plot Panel (bottom)
        plot_panel_container = QWidget()  # Use a container for better sizing control if needed
        plots_hbox_layout = QHBoxLayout(plot_panel_container)  # Horizontal layout for two plots

        # Loss Plot Widget
        self.loss_plot_widget = pg.PlotWidget(name="LossPlot")
        self.loss_plot_widget.setTitle("Loss over Passes", color=pg.getConfigOption('foreground'), size="10pt")
        self.loss_plot_widget.setLabel('left', 'Loss Value', color=pg.getConfigOption('foreground'))
        self.loss_plot_widget.setLabel('bottom', 'Pass Number', color=pg.getConfigOption('foreground'))
        self.loss_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot_item = self.loss_plot_widget.plot(pen=pg.mkPen(color='r', width=2), name="Loss")
        plots_hbox_layout.addWidget(self.loss_plot_widget)

        # Accuracy Plot Widget
        self.accuracy_plot_widget = pg.PlotWidget(name="AccuracyPlot")
        self.accuracy_plot_widget.setTitle("Accuracy over Passes", color=pg.getConfigOption('foreground'), size="10pt")
        self.accuracy_plot_widget.setLabel('left', 'Accuracy Value', color=pg.getConfigOption('foreground'))
        self.accuracy_plot_widget.setLabel('bottom', 'Pass Number', color=pg.getConfigOption('foreground'))
        self.accuracy_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.accuracy_plot_item = self.accuracy_plot_widget.plot(pen=pg.mkPen(color='b', width=2), name="Accuracy")
        plots_hbox_layout.addWidget(self.accuracy_plot_widget)
        plot_panel_container.setMaximumHeight(250)  # Set min height for the plot area
        self.main_layout.addWidget(plot_panel_container, stretch=1)

        self.architecture_label = QLabel()
        label_font = QFont()
        label_font.setPointSize(12)
        self.architecture_label.setFont(label_font)
        self.controls_layout.addWidget(self.architecture_label)
        self.controls_layout.addStretch(1)

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
                self.pass_display_label.setText(f"Pass: {self.current_pass}/{self.max_pass}")

                for pass_num_int in numeric_passes:
                    pass_num_str = str(pass_num_int)
                    pass_data_dict = self.data.get(pass_num_str, {})
                    loss = pass_data_dict.get("loss")
                    accuracy = pass_data_dict.get("accuracy")

                    # Ensure values are valid numbers before appending
                    if isinstance(loss, (int, float)) and isinstance(accuracy, (int, float)):
                        self.all_metrics_data.append((pass_num_int, float(loss), float(accuracy)))

                self.all_metrics_data.sort(key=lambda x: x[0])
                # Calculate min/max loss and accuracy
                min_loss = min(self.all_metrics_data, key=lambda x: x[1])[1] if self.all_metrics_data else None
                max_loss = max(self.all_metrics_data, key=lambda x: x[1])[1] if self.all_metrics_data else None
                min_acc = min(self.all_metrics_data, key=lambda x: x[2])[2] if self.all_metrics_data else None
                max_acc = max(self.all_metrics_data, key=lambda x: x[2])[2] if self.all_metrics_data else None
                # Set plot limits
                self.loss_plot_widget.setXRange(1, self.max_pass)
                self.accuracy_plot_widget.setXRange(1, self.max_pass)
                self.loss_plot_widget.setYRange(min_loss, max_loss)
                self.accuracy_plot_widget.setYRange(min_acc, max_acc)
            else:  # No numeric passes found
                self.max_pass = 1
                self.current_pass = "1"
                self.pass_slider.setMaximum(1)
                self.pass_slider.setValue(1)
                self.pass_display_label.setText("Pass: 1/1")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None  # Ensure data is None if loading fails
            self.max_pass = 1
            self.current_pass = "1"
            self.pass_slider.setMaximum(1)
            self.pass_slider.setValue(1)
            self.pass_display_label.setText("Pass: N/A")

    def update_plot(self):
        if not self.all_metrics_data:
            if self.loss_plot_item:
                self.loss_plot_item.setData([], [])
            if self.accuracy_plot_item:
                self.accuracy_plot_item.setData([], [])
            return

        current_pass_int = int(self.current_pass)
        plot_data_tuples = [item for item in self.all_metrics_data if item[0] <= current_pass_int]

        if not plot_data_tuples:
            if self.loss_plot_item:
                self.loss_plot_item.setData([], [])
            if self.accuracy_plot_item:
                self.accuracy_plot_item.setData([], [])
            return

        passes = [item[0] for item in plot_data_tuples]
        losses = [item[1] for item in plot_data_tuples]
        accuracies = [item[2] for item in plot_data_tuples]

        self.loss_plot_item.setData(passes, losses)
        self.accuracy_plot_item.setData(passes, accuracies)

    def change_pass(self, pass_value):
        """Change the visualization to show a different pass."""
        self.current_pass = str(pass_value)
        self.pass_display_label.setText(f"Pass: {self.current_pass}/{self.max_pass}")
        self.visualize_network(preserve_current_view=self._initial_fit_done)
        self.update_plot()

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

        # Add prediction display
        self.add_prediction_display(all_layers, linear_layers, layer_spacing, neurons)

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

    def add_prediction_display(self, all_layers, linear_layers, layer_spacing, neurons):
        x_output_area_center = 150  # Default if no neurons/layers
        if neurons:
            # X-coord of the center of a neuron in the last layer
            x_last_neuron_layer_center_coord = neurons[-1][0][1].x()
            x_output_area_center = x_last_neuron_layer_center_coord + layer_spacing  # Adjust this factor as needed
        elif linear_layers:  # Fallback if no neurons drawn but linear layers exist
            # Estimate based on number of linear layers
            x_output_area_center = (len(linear_layers)) * layer_spacing + 200
        elif all_layers:  # Fallback if only generic layers
            x_output_area_center = (len(all_layers) - 1) * layer_spacing * 0.5 + 200

        if all_layers and self.current_pass in self.data:
            if 'prediction' in self.data[self.current_pass] and 'logits' in self.data[self.current_pass]:
                    prediction = self.data[self.current_pass]['prediction']

                    # --- Histogram Drawing ---
                    # 1. Calculate Softmax probabilities
                    logits = self.data[self.current_pass]['logits']
                    logits_array = np.array(logits, dtype=np.float32)
                    exp_logits = np.exp(logits_array - np.max(logits_array))  # Subtract max for numerical stability
                    probabilities = exp_logits / np.sum(exp_logits)

                    num_classes = len(probabilities)
                    hist_bar_width = 20
                    hist_bar_spacing = 5
                    max_bar_pixel_height = 360  # Max height for a bar representing 1.0 probability
                    hist_total_width = num_classes * hist_bar_width + (num_classes - 1) * hist_bar_spacing
                    hist_start_x = x_output_area_center - (hist_total_width / 2)
                    current_hist_x = int(hist_start_x)  # Initialize for the loop

                    # Y position for the baseline of the histogram bars
                    y_hist_baseline = 250

                    hist_bar_font = QFont()
                    hist_bar_font.setPointSize(8)

                    for i in range(num_classes):
                        bar_height = probabilities[i] * max_bar_pixel_height

                        # Bar
                        bar = QGraphicsRectItem(current_hist_x,
                                                y_hist_baseline - bar_height,
                                                hist_bar_width,
                                                bar_height)
                        bar_color = QColor(Qt.GlobalColor.blue)
                        if i == prediction:
                            bar_color = QColor(Qt.GlobalColor.green)  # Highlight predicted bar
                        bar.setBrush(QBrush(bar_color))
                        bar.setPen(QPen(Qt.GlobalColor.black, 0.5))  # Thin border
                        self.scene.addItem(bar)

                        # Label for the bar (class index)
                        label = QGraphicsTextItem(str(i))
                        label.setFont(hist_bar_font)
                        label_x_pos = current_hist_x + hist_bar_width / 2 - label.boundingRect().width() / 2
                        label_y_pos = y_hist_baseline + 2  # Just below the baseline
                        label.setPos(label_x_pos, label_y_pos)
                        self.scene.addItem(label)

                        current_hist_x += hist_bar_width + hist_bar_spacing

                    # --- Lines from last neurons to histogram bars ---
                    if neurons and neurons[-1] and len(neurons[-1]) == num_classes:
                        last_layer_neurons = neurons[-1]
                        line_pen = QPen(QColor(Qt.GlobalColor.gray), 1, Qt.PenStyle.DashLine)  # Dashed gray line
                        line_pen.setDashPattern([4, 2])  # Define dash pattern: 4px line, 2px gap

                        for i in range(num_classes):
                            if i < len(last_layer_neurons):  # Safety check
                                neuron_item, neuron_center_pos = last_layer_neurons[i]

                                # Target X for the line: center of the i-th histogram bar
                                target_x = hist_start_x + \
                                    (i * (hist_bar_width + hist_bar_spacing)) + (hist_bar_width / 2)
                                # Target Y for the line: slightly above the histogram baseline
                                target_y = y_hist_baseline - 5  # Adjust this offset as needed

                                line = QGraphicsLineItem(neuron_center_pos.x(), neuron_center_pos.y(),
                                                         target_x, target_y)
                                line.setPen(line_pen)
                                line.setZValue(-0.5)  # Behind histogram bars but above main connections
                                self.scene.addItem(line)

                    # --- Prediction Text (position below histogram) ---
                    prediction_text_content = f"Prediction: {prediction}"
                    prediction_text_item = QGraphicsTextItem(
                        prediction_text_content)  # Create item to measure its width
                    pred_font = QFont()
                    pred_font.setBold(True)
                    pred_font.setPointSize(12)
                    prediction_text_item.setFont(pred_font)
                    prediction_text_item.setDefaultTextColor(QColor(Qt.GlobalColor.black))

                    y_pred_text = y_hist_baseline + 2 + QGraphicsTextItem(str(0)).boundingRect().height() + 10

                    # Calculate starting X for prediction text so it's centered around x_output_area_center
                    pred_text_start_x = x_output_area_center - (prediction_text_item.boundingRect().width() / 2)

                    prediction_text_item.setPos(pred_text_start_x, y_pred_text)
                    self.scene.addItem(prediction_text_item)
            else:
                print(
                    f"Final Identity layer tag not found or no data for it in pass {self.current_pass}.")

    def update_architecture_label(self, architecture):
        """Updates the label in the controls layout with architecture details, wrapping long text."""
        architecture_info = []
        for key, details in architecture.items():
            if "Linear" in key and "in_features" in details and " out_features" in details:
                architecture_info.append(
                    f"{key.split('_')[1]}: {int(details['in_features'])} -> {int(details[' out_features'])}")
            elif "CNN" in key and "in_features" in details and " out_features" in details:
                architecture_info.append(
                    f"{key.split('_')[1]}: {int(details['in_features'])} -> {int(details[' out_features'])}")
            elif "Conv2d" in key:
                architecture_info.append(key.split('_')[1])
            elif "MaxPool2d" in key:
                architecture_info.append(key.split('_')[1])
            elif any(layer_type in key for layer_type in
                     ["ReLU", "tanh", "LeakyReLU", "sigmoid", "BatchNorm", "Dropout"]):
                architecture_info.append(key.split('_')[1])

        architecture_text = " | ".join(architecture_info)
        self.architecture_label.setText(architecture_text)
        self.architecture_label.setWordWrap(True)

    def create_neurons(self, layer_sizes, layer_spacing, neuron_spacing, neuron_radius, linear_layers):
        """Create and position neurons for each layer."""
        neurons = []

        scene_rect = self.view.rect()
        available_height = scene_rect.height()

        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_neurons = []
            x = layer_idx * layer_spacing + 100

            # Center the layer vertically
            layer_height = (layer_size - 1) * neuron_spacing
            y_offset = (available_height - layer_height) / 2

            for neuron_idx in range(layer_size):
                y = y_offset + neuron_idx * neuron_spacing

                # Create neuron (circle)
                neuron = QGraphicsEllipseItem(x - neuron_radius, y - neuron_radius,
                                              2 * neuron_radius, 2 * neuron_radius)

                # Get activation value
                activation_value = self.get_activation_value(layer_idx, neuron_idx, linear_layers)

                # Set color based on activation
                if activation_value is not None:
                    color = get_diverging_neuron_color(activation_value)
                    neuron.setBrush(QBrush(color))
                else:
                    neuron.setBrush(QBrush(QColor(200, 200, 200)))

                neuron.setPen(QPen(Qt.GlobalColor.black, 1))
                neuron.setZValue(2)
                self.scene.addItem(neuron)

                # Add neuron label
                label_text = f"L={layer_idx}.N={neuron_idx}"
                if activation_value is not None:
                    label_text += f"\nact={activation_value:.3f}"

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
                            color = QColor(255, 0, 0, min(255, int(weight_abs * 200) + 50)) if weight >= 0 else QColor(
                                0, 0, 255, min(255, int(weight_abs * 200) + 50))

                            # Draw connection line
                            start_point = neurons[i][from_idx][1]
                            end_point = neurons[i + 1][to_idx][1]

                            line = QGraphicsLineItem(start_point.x(), start_point.y(),
                                                     end_point.x(), end_point.y())
                            line.setPen(QPen(color, thickness))
                            line.setZValue(-2)  # Put connections behind neurons
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
            if any(layer_type in layer_key for layer_type in ["ReLU", "tanh", "LeakyReLU", "sigmoid"]):
                preceding_layer = all_layers[i-1][0]
                prepreciding_layer = all_layers[i-2][0] if i >= 2 else ""
                if "Linear" not in preceding_layer and "Linear" not in prepreciding_layer:
                    continue
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
        super().resizeEvent(event)


def visualize_mlp(json_path):
    """Create and show the MLP visualizer with the given data."""
    app = QApplication(sys.argv)
    visualizer = MLPVisualizer(json_path)
    visualizer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    visualize_mlp("../../example.json")
