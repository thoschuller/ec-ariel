from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
      QWidget, QVBoxLayout, QLabel, 
        QComboBox, QLineEdit, QHBoxLayout,
          QGraphicsScene, QGraphicsView, QGraphicsItem,
            QGraphicsEllipseItem, QGraphicsPathItem
)

from PyQt5.QtGui import QPainter, QPen, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QRectF, QPointF
import sys


# ========================
# Node Editor Components
# ========================

# from QNodeEditor import Node
# from QNodeEditor import NodeEditorDialog

# class AddNode(Node):
#     code = 0  # Unique code for each node type
    
#     def create(self):
#         self.title = 'Addition'  # Set the node title
        
#         self.add_label_output('Output')  # Add output socket
#         self.add_value_input('Value 1')  # Add input socket for first value
#         self.add_value_input('Value 2')  # Add input socket for second value
        
#     def evaluate(self, values: dict):
#         result = values['Value 1'] + values['Value 2']  # Add 'Value 1' and 'Value 2'
#         self.set_output_value('Output', result)         # Set as value for 'Output'

# class OutNode(Node):
#     code = 1  # Unique code for this node

#     def create(self):
#         self.title = 'Output'  # Set the title of the node
#         self.add_label_input('Value')  # Add input value


# if __name__ == "__main__":
#     app = QApplication(sys.argv)

#     dialog = NodeEditorDialog()
#     # Register both custom nodes
#     dialog.editor.available_nodes = {
#         'Addition': AddNode,
#         'Output': OutNode
#     }
#     dialog.editor.output_node = OutNode
#     if dialog.exec():
#         print(dialog.result)
#         sys.exit(app.exec_())

#     # Run the PyQt application
#     app.exec()

#     sys.exit(app.exec_())


# ========================
# Main GUI
# ========================

# class RobotEvolutionGUI(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Robot Evolution System")
#         self.setGeometry(100, 100, 1000, 700)

#         self.tab_widget = QTabWidget(self)
#         self.setCentralWidget(self.tab_widget)

#         # Keep your existing selection tab
#         self.tab_widget.addTab(self.create_selection_tab(), "Selection Algorithms")

#         # Add Node Editor tab
#         self.tab_widget.addTab(NodeEditor(), "Node Editor (Experimental)")

#     def create_selection_tab(self):
#         widget = QWidget()
#         layout = QVBoxLayout()
#         layout.addWidget(QLabel("Define Parent and Survivor Selection Types"))

#         # Mapping for display names and internal names
#         self.selection_display_to_internal = {
#             "Tournament": "tournament",
#             "Roulette Wheel": "roulette",
#             "Top-N": "topn"
#         }
#         self.selection_internal_to_display = {v: k for k, v in self.selection_display_to_internal.items()}

#         # Parent selection dropdown
#         self.parent_dropdown = QComboBox()
#         self.parent_dropdown.addItems(self.selection_internal_to_display.values())
#         self.parent_dropdown.setToolTip("Click the dropdown to choose the parent selection method to be used by your evolutionary algorithm.")
#         parent_title = QLabel("Parent Selection: ")
#         layout.addWidget(parent_title)
#         layout.addWidget(self.parent_dropdown)

#         # Parent selection parameters
#         self.parent_params_layout = QVBoxLayout()
#         layout.addLayout(self.parent_params_layout)

#         # Connect parent dropdown to update function
#         self.parent_dropdown.currentIndexChanged.connect(
#             lambda: self.update_selection_params(self.parent_dropdown, self.parent_params_layout)
#         )

#         # Survivor selection dropdown
#         self.survivor_dropdown = QComboBox()
#         self.survivor_dropdown.addItems(self.selection_internal_to_display.values())
#         self.survivor_dropdown.setToolTip("Click the dropdown to choose the survival selection method to be used by your evolutionary algorithm.")
#         layout.addWidget(QLabel("Survivor Selection:"))
#         layout.addWidget(self.survivor_dropdown)

#         # Survivor selection parameters
#         self.survivor_params_layout = QVBoxLayout()
#         layout.addLayout(self.survivor_params_layout)

#         # Connect survivor dropdown to update function
#         self.survivor_dropdown.currentIndexChanged.connect(
#             lambda: self.update_selection_params(self.survivor_dropdown, self.survivor_params_layout)
#         )

#         widget.setLayout(layout)

#         # Automatically update parameters for the initial selection
#         self.update_selection_params(self.parent_dropdown, self.parent_params_layout)
#         self.update_selection_params(self.survivor_dropdown, self.survivor_params_layout)

#         return widget

#     def update_selection_params(self, item, params_layout):
#         """Update the parameter input fields based on the selected function."""
#         if item is None:
#             return

#         # Clear existing parameter input fields and layouts
#         while params_layout.count():
#             layout_item = params_layout.takeAt(0)
#             if layout_item.widget():
#                 layout_item.widget().deleteLater()
#             elif layout_item.layout():
#                 child_layout = layout_item.layout()
#                 while child_layout.count():
#                     child_item = child_layout.takeAt(0)
#                     if child_item.widget():
#                         child_item.widget().deleteLater()
#                 child_layout.deleteLater()

#         # Selection parameters
#         selection_params = {
#             "tournament": {"k": 2},
#             "roulette": {"n": 1},
#             "topn": {"n": 1}
#         }

#         # Get the internal name from the display name
#         selected_function_display = item.currentText()
#         selected_function = self.selection_display_to_internal.get(selected_function_display, None)

#         if selected_function in selection_params:
#             for param, value in selection_params[selected_function].items():
#                 input_layout = QHBoxLayout()
#                 input_label = QLabel(f"{param}:")
#                 input_field = QLineEdit(str(value))
#                 input_layout.addWidget(input_label)
#                 input_layout.addWidget(input_field)
#                 params_layout.addLayout(input_layout)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = RobotEvolutionGUI()
#     window.show()
#     sys.exit(app.exec_())

###################### 

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
      QWidget, QVBoxLayout, QLabel, 
        QComboBox, QLineEdit, QHBoxLayout,
          QGraphicsScene, QGraphicsView, QGraphicsItem,
            QGraphicsEllipseItem
)
from PyQt5.QtGui import QPainter, QPen, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QRectF, QPointF
import sys


# ========================
# Node Editor Components
# ========================

class Port(QGraphicsEllipseItem):
    """A small circle that acts as a connection point."""
    def __init__(self, parent=None, radius=8):
        super().__init__(-radius/2, -radius/2, radius, radius, parent)
        self.setBrush(QBrush(Qt.darkBlue))
        self.setPen(QPen(Qt.black))
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)

    def scenePosCenter(self):
        """Return center position of the port in scene coordinates."""
        return self.scenePos() + QPointF(self.rect().width()/2, self.rect().height()/2)


class Connection(QGraphicsPathItem):
    """Bezier curve between two ports."""
    def __init__(self, port1, port2=None):
        super().__init__()
        self.port1 = port1
        self.port2 = port2
        self.setPen(QPen(Qt.black, 2))
        self.updatePath(QPointF(0, 0))  # temp path

    def updatePath(self, mouse_pos):
        p1 = self.port1.scenePosCenter()
        p2 = self.port2.scenePosCenter() if self.port2 else mouse_pos

        path = QPainterPath(p1)
        dx = (p2.x() - p1.x()) * 0.5
        path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
        self.setPath(path)


class Node(QGraphicsEllipseItem):
    """A draggable node with a port."""
    def __init__(self, x, y):
        super().__init__(0, 0, 80, 80)
        self.setBrush(Qt.yellow)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPos(x, y)

        # Add a port
        self.port = Port(self)
        self.port.setPos(80, 40)  # right side middle


class NodeEditor(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)

        # Add some test nodes
        self.node1 = Node(0, 0)
        self.node2 = Node(200, 100)
        self.scene.addItem(self.node1)
        self.scene.addItem(self.node2)

        self.temp_connection = None  # store temp connection

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, Port):
            # Start a new connection
            self.temp_connection = Connection(item)
            self.scene.addItem(self.temp_connection)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.temp_connection:
            mouse_pos = self.mapToScene(event.pos())
            self.temp_connection.updatePath(mouse_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.temp_connection:
            item = self.itemAt(event.pos())
            if isinstance(item, Port) and item is not self.temp_connection.port1:
                # Complete connection
                self.temp_connection.port2 = item
                self.temp_connection.updatePath(self.mapToScene(event.pos()))
            else:
                # Cancel connection
                self.scene.removeItem(self.temp_connection)
            self.temp_connection = None
        super().mouseReleaseEvent(event)


# ========================
# Main GUI
# ========================

class RobotEvolutionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Evolution System")
        self.setGeometry(100, 100, 1000, 700)

        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        # Keep your existing selection tab
        self.tab_widget.addTab(self.create_selection_tab(), "Selection Algorithms")

        # Add Node Editor tab
        self.tab_widget.addTab(NodeEditor(), "Node Editor (Experimental)")

    def create_selection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Define Parent and Survivor Selection Types"))

        # Mapping for display names and internal names
        self.selection_display_to_internal = {
            "Tournament": "tournament",
            "Roulette Wheel": "roulette",
            "Top-N": "topn"
        }
        self.selection_internal_to_display = {v: k for k, v in self.selection_display_to_internal.items()}

        # Parent selection dropdown
        self.parent_dropdown = QComboBox()
        self.parent_dropdown.addItems(self.selection_internal_to_display.values())
        self.parent_dropdown.setToolTip("Click the dropdown to choose the parent selection method to be used by your evolutionary algorithm.")
        parent_title = QLabel("Parent Selection: ")
        layout.addWidget(parent_title)
        layout.addWidget(self.parent_dropdown)

        # Parent selection parameters
        self.parent_params_layout = QVBoxLayout()
        layout.addLayout(self.parent_params_layout)

        # Connect parent dropdown to update function
        self.parent_dropdown.currentIndexChanged.connect(
            lambda: self.update_selection_params(self.parent_dropdown, self.parent_params_layout)
        )

        # Survivor selection dropdown
        self.survivor_dropdown = QComboBox()
        self.survivor_dropdown.addItems(self.selection_internal_to_display.values())
        self.survivor_dropdown.setToolTip("Click the dropdown to choose the survival selection method to be used by your evolutionary algorithm.")
        layout.addWidget(QLabel("Survivor Selection:"))
        layout.addWidget(self.survivor_dropdown)

        # Survivor selection parameters
        self.survivor_params_layout = QVBoxLayout()
        layout.addLayout(self.survivor_params_layout)

        # Connect survivor dropdown to update function
        self.survivor_dropdown.currentIndexChanged.connect(
            lambda: self.update_selection_params(self.survivor_dropdown, self.survivor_params_layout)
        )

        widget.setLayout(layout)

        # Automatically update parameters for the initial selection
        self.update_selection_params(self.parent_dropdown, self.parent_params_layout)
        self.update_selection_params(self.survivor_dropdown, self.survivor_params_layout)

        return widget

    def update_selection_params(self, item, params_layout):
        """Update the parameter input fields based on the selected function."""
        if item is None:
            return

        # Clear existing parameter input fields and layouts
        while params_layout.count():
            layout_item = params_layout.takeAt(0)
            if layout_item.widget():
                layout_item.widget().deleteLater()
            elif layout_item.layout():
                child_layout = layout_item.layout()
                while child_layout.count():
                    child_item = child_layout.takeAt(0)
                    if child_item.widget():
                        child_item.widget().deleteLater()
                child_layout.deleteLater()

        # Selection parameters
        selection_params = {
            "tournament": {"k": 2},
            "roulette": {"n": 1},
            "topn": {"n": 1}
        }

        # Get the internal name from the display name
        selected_function_display = item.currentText()
        selected_function = self.selection_display_to_internal.get(selected_function_display, None)

        if selected_function in selection_params:
            for param, value in selection_params[selected_function].items():
                input_layout = QHBoxLayout()
                input_label = QLabel(f"{param}:")
                input_field = QLineEdit(str(value))
                input_layout.addWidget(input_label)
                input_layout.addWidget(input_field)
                params_layout.addLayout(input_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotEvolutionGUI()
    window.show()
    sys.exit(app.exec_())