import sys

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# ENUMS as a base

# Import the backend components
from ariel.simulation.environments.__init__ import __all__ as envs
from ariel.simulation.tasks.__init__ import __all__ as ffs, tasks, _task_fitness_function_map_
from ariel.simulation.controllers import *

# ========================
# Node Editor Components
# ========================
from QNodeEditor import Node, NodeEditorDialog
from QNodeEditor.node import Node

class PhenotypeNode(Node):
    code = 111

    def create(self):
        self.title = "Select Phenotype"
        self.add_combo_box_entry("Phenotype", items=["CPG", "FFNN", "CTRNN"])
        self.add_label_output("Selected Phenotype")

    def evaluate(self, values: dict):
        phenotype = self.entries_dict["Phenotype"].get_value()
        self.set_output_value("Selected Phenotype", phenotype)
        return phenotype

class GenotypeNode(Node):
    code = 211

    def create(self):
        self.title = "Select Genotype"
        self.add_combo_box_entry("Genotype", items=["Direct", "Compositional Pattern Producing Network (CPPN)"])
        self.add_label_output("Selected Genotype")

    def evaluate(self, values: dict):
        genotype = self.entries_dict["Genotype"].get_value()
        self.set_output_value("Selected Genotype", genotype)
        return genotype

class EnvironmentNode(Node):
    code=121
    def create(self):
        self.title = "Select Environment"
        self.add_combo_box_entry("Environment", items=envs)
        self.add_label_output("Selected Environment")

    def evaluate(self, values: dict):
        env = self.entries_dict["Environment"].get_value()
        self.set_output_value("Selected Environment", env)
        return env

class TaskNode(Node):
    code = 122

    def create(self):
        self.title = "Select Task"
        self.add_combo_box_entry("Task", items=tasks)
        self.add_label_output("Selected Task")  # output socket

    def evaluate(self, values: dict):
        task = self.entries_dict["Task"].get_value()
        self.set_output_value("Selected Task", task)
        return task
    
class FitnessFunctionNode(Node):
    code = 221

    task_map = {
        "Gate Learning": ["xy_displacement_ff", "x_speed_ff", "y_speed_ff"],
        "Targeted Locomotion": ["distance_to_target_ff"],
        "Turning In Place": ["turning_in_place_ff"],
    }

    def create(self):
        self.title = "Select Fitness Function"
        # Input socket from TaskNode
        self.add_value_input("Task")

        # Dropdown for fitness functions (initially empty)
        self.add_combo_box_entry("Fitness Function", items=[])  # Default items
        self.add_label_output("Selected Fitness Function")

    def evaluate(self, values: dict):
        # This is the **value from the input socket**
        task_input = values.get("Task")
        combo = self.entries_dict["Fitness Function"]

        if task_input in self.task_map:
            # Update dropdown items dynamically
            combo.clear()
            combo.addItems(self.task_map[task_input])

        # Get currently selected fitness function
        selected = combo.get_value()
        self.set_output_value("Selected Fitness Function", selected)
        return selected

class MutationNode(Node):
    code=311

    def create(self):
        self.title = "Select Mutation"
        self.add_combo_box_entry("Mutation", items=["Gaussian", "Uniform", "Creep"])
        self.add_label_output("Selected Mutation")
    
    def evaluate(self, values: dict):
        mutation = self.entries_dict["Mutation"].get_value()
        self.set_output_value("Selected Mutation", mutation)
        return mutation
    
class CrossoverNode(Node):
    code=312

    def create(self):
        self.title = "Select Crossover"
        self.add_combo_box_entry("Crossover", items=["One-Point", "Two-Point", "Uniform"])
        self.add_label_output("Selected Crossover")
    
    def evaluate(self, values: dict):
        crossover = self.entries_dict["Crossover"].get_value()
        self.set_output_value("Selected Crossover", crossover)
        return crossover

class ParentSelectionNode(Node):
    code=321

    def create(self):
        self.title = "Select Parent Selection"
        self.add_combo_box_entry("Parent Selection", items=["Tournament", "Roulette Wheel", "Top-N"])
        self.add_label_output("Selected Parent Selection")
    
    def evaluate(self, values: dict):
        parent_selection = self.entries_dict["Parent Selection"].get_value()
        self.set_output_value("Selected Parent Selection", parent_selection)
        return parent_selection

class SurvivorSelectionNode(Node):
    code=322

    def create(self):
        self.title = "Select Survivor Selection"
        self.add_combo_box_entry("Survivor Selection", items=["Tournament", "Roulette Wheel", "Top-N"])
        self.add_label_output("Selected Survivor Selection")
    
    def evaluate(self, values: dict):
        survivor_selection = self.entries_dict["Survivor Selection"].get_value()
        self.set_output_value("Selected Survivor Selection", survivor_selection)
        return survivor_selection

class EAParametersNode(Node):
    code=422

    def create(self):
        self.title = "EA Parameters"
        self.add_line_edit_entry("Population Size", "100")
        self.add_line_edit_entry("Generations", "50")
        self.add_line_edit_entry("Mutation Rate", "0.01")
        self.add_line_edit_entry("Crossover Rate", "0.7")
        self.add_label_output("EA Parameters")
    
    def evaluate(self, values: dict):
        params = {
            "Population Size": int(self.entries_dict["Population Size"].get_value()),
            "Generations": int(self.entries_dict["Generations"].get_value()),
            "Mutation Rate": float(self.entries_dict["Mutation Rate"].get_value()),
            "Crossover Rate": float(self.entries_dict["Crossover Rate"].get_value()),
        }
        self.set_output_value("EA Parameters", params)
        return params
    

class EARun(Node):
    code = 999

    def create(self):
        self.title = "Output"
        self.add_label_input("Result")

    def evaluate(self, values: dict):
        value = self.get_input_value("Result")
        self.set_output_value("Result", value)
        self.result = value
        return value


app = QApplication(sys.argv)
dialog = NodeEditorDialog()
# Register both custom nodes
dialog.editor.available_nodes = {"Environment" : EnvironmentNode, 
                                 "Task": TaskNode,
                                 "FitnessFunction": FitnessFunctionNode,
                                 "Phenotype": PhenotypeNode,
                                 "Genotype": GenotypeNode,
                                 "Mutation": MutationNode,
                                 "Crossover": CrossoverNode,
                                 "ParentSelection": ParentSelectionNode,
                                 "SurvivorSelection": SurvivorSelectionNode,
                                 "EA_Parameters": EAParametersNode,
                                 "EA_run": EARun}
dialog.editor.output_node = EARun
if dialog.exec():
    print(dialog.result)


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
