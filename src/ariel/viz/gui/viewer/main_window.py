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
from PyQt5.QtWidgets import QPushButton, QGraphicsProxyWidget
from PyQt5.QtWidgets import QLineEdit, QComboBox
from QNodeEditor.node import Node
from QNodeEditor import Node, NodeEditorDialog

# ENUMS as a base

# Import the backend components
from ariel.simulation.environments.__init__ import __all__ as envs
from ariel.simulation.tasks.__init__ import _task_fitness_function_map_
from ariel.simulation.controllers import *

# ========================
# Node Editor Components
# ========================

# The extractor used is what should be specificied in this case I guess
class PhenotypeNode(Node):
    code = 111

    def create(self):
        self.title = "Select Phenotype"
        self.add_combo_box_entry("Phenotype", items=["RoboGen-robot", "Integer-Solution"])
        self.add_label_output("Selected Phenotype")

    def evaluate(self, values: dict):
        phenotype = self.entries_dict["Phenotype"].get_value()
        self.set_output_value("Selected Phenotype", phenotype)
        return phenotype

class GenotypeNode(Node):
    code = 211

    def create(self):
        self.title = "Select Genotype"
        self.add_combo_box_entry("Genotype", items=["HighProbEncoding", "IntegerEncoding"])
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
        self.add_combo_box_entry("Task", items=_task_fitness_function_map_.keys())
        self.add_label_output("Selected Task")  # output socket

    def evaluate(self, values: dict):
        task = self.entries_dict["Task"].get_value()
        self.set_output_value("Selected Task", task)
        # print(f"TaskNode selected task: {task}")
        return task
    
class FitnessFunctionNode(Node):
    code = 221


    task_map = _task_fitness_function_map_ # imported above from the actual file

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

# ----------------------------
# EA Parameters Node
# ----------------------------

class EAParametersNode(Node):
    code = 422

    def create(self):
        self.title = "EA Parameters"

        # Outputs only the parameters needed by EA
        self.add_value_output("Population Size")
        self.add_value_output("Generations")

        self.entries_dict = {entry.name: entry for entry in self.entries}

    def evaluate(self, values: dict):
        # Fetch values from line edits
        pop_size = self.entries_dict["Population Size"].get_value()
        generations = self.entries_dict["Generations"].get_value()

        # Store outputs for input nodes to fetch
        self.set_output_value("Population Size", pop_size)
        self.set_output_value("Generations", generations)

        return {"Population Size": pop_size, "Generations": generations}


# ========================
# EA Run Node
# ========================
class EARun(Node):
    code = 999

    def create(self):
        self.title = "Run EA"

        # Add input sockets for all configurable values
        input_names = [
            "Phenotype", "Genotype", "Environment", "Task",
            "Fitness Function", "Mutation", "Crossover",
            "Parent Selection", "Survivor Selection",
            "pop_size", "generations"
        ]
        for name in input_names:
            self.add_value_input(name)

    def evaluate(self, values: dict):
        # Automatically fetch values from connected nodes
        pop_size = self.fetch_input_value("pop_size")
        generations = self.fetch_input_value("generations")
        phenotype = self.fetch_input_value("Phenotype")
        genotype = self.fetch_input_value("Genotype")
        environment = self.fetch_input_value("Environment")
        task = self.fetch_input_value("Task")
        fitness_functions = self.fetch_input_value("Fitness Function")
        mutation = self.fetch_input_value("Mutation")
        crossover = self.fetch_input_value("Crossover")
        parent_selection = self.fetch_input_value("Parent Selection")
        survivor_selection = self.fetch_input_value("Survivor Selection")

        if None in [pop_size, generations]:
            raise ValueError("EAParametersNode values not available!")

        # --- Build EA backend ---
        from ariel.ec.a004 import BasicEA, EAStep, create_individual, \
            parent_selection as ps_op, crossover as co_op, mutation as mut_op, \
            evaluate as eval_op, survivor_selection as ss_op

        population = [create_individual() for _ in range(pop_size)]
        population = eval_op(population)

        operations = []
        if parent_selection:
            operations.append(EAStep("parent_selection", ps_op))
        if crossover:
            operations.append(EAStep("crossover", co_op))
        if mutation:
            operations.append(EAStep("mutation", mut_op))
        operations.append(EAStep("evaluation", eval_op))
        if survivor_selection:
            operations.append(EAStep("survivor_selection", ss_op))

        ea = BasicEA(population=population, operations=operations, num_of_generations=generations)
        ea.run()

        best = ea.get_solution("best", only_alive=False)
        self.set_output_value("EA Result", best)
        self.result = best
        return best




app = QApplication(sys.argv)
dialog = NodeEditorDialog()
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
