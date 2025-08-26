# --------------------------------------------------------------------
# from ariel.viz.analysis.parse_db import get_fitness_over_generations
# from results_viewer import EmbeddedPlotWidget

from QNodeEditor import Node
from PyQt5.QtWidgets import QApplication, QPushButton, QGraphicsProxyWidget
from QNodeEditor import NodeEditorDialog
from ariel.ec.a004 import BasicEA, EAStep, create_individual, \
    parent_selection as ps_op, crossover as co_op, mutation as mut_op, \
    evaluate as eval_op, survivor_selection as ss_op
from rich.console import Console

console = Console()

# ----------------------------
# EA Parameters Node
# ----------------------------
class EAParameters(Node):
    code = 0

    def create(self):
        self.title = 'Evolutionary Algorithm Parameters'

        # Editable values
        self.add_value_input("Population Size")
        self.add_value_input("Generations")

        # Outputs (different names to avoid clashes)
        self.add_label_output("PopSizeOut")
        self.add_label_output("GenerationsOut")

    def evaluate(self, values: dict):
        pop_size = values["Population Size"]
        generations = values["Generations"]

        print(f"EAParameters -> Pop={pop_size}, Gen={generations}")

        # Send values forward with output names
        self.set_output_value("PopSizeOut", pop_size)
        self.set_output_value("GenerationsOut", generations)



# ----------------------------
# EA Run Node
# ----------------------------
class EARun(Node):
    code = 1

    def create(self):
        self.title = 'Run EA'

        # Inputs from EAParameters
        self.add_value_input("PopSizeOut")
        self.add_value_input("GenerationsOut")

        # Optional: Add an output for the result
        self.add_label_output("Result")

    def evaluate(self, values: dict):

        """Run the EA using the parameters provided by EAParameters."""
        # Get parameter values from inputs
        pop_size = values["PopSizeOut"]
        generations = values["GenerationsOut"]

        console.log(f"Running EA with population={pop_size}, generations={generations}")

        # Build initial population
        population_list = [create_individual() for _ in range(int(pop_size))]
        population_list = eval_op(population_list)

        # Define EA steps
        ops = [
            EAStep("parent_selection", ps_op),
            EAStep("crossover", co_op),
            EAStep("mutation", mut_op),
            EAStep("evaluation", eval_op),
            EAStep("survivor_selection", ss_op),
        ]

        # Run EA
        ea = BasicEA(population_list, operations=ops, num_of_generations=int(generations))
        ea.run()

        # Collect solutions
        best = ea.get_solution(only_alive=False)
        median = ea.get_solution("median", only_alive=False)
        worst = ea.get_solution("worst", only_alive=False)

        console.log(f"[green]Best:[/green] {best}")
        console.log(f"[yellow]Median:[/yellow] {median}")
        console.log(f"[red]Worst:[/red] {worst}")

        # Save result to output so other nodes can use it
        self.set_output_value("Result", best)
        self.result = best  # store for retrieval after dialog.exec()

        return best

# ----------------------------
# Main
# ----------------------------
app = QApplication([])
dialog = NodeEditorDialog()
dialog.editor.available_nodes = {
    'EA Parameters': EAParameters,
    'Run EA': EARun
}

# Add output node instance
ea_run_node = EARun()
dialog.editor.output_node = ea_run_node

if dialog.exec():
    print("Editor finished. Best individual:",  ea_run_node.evaluate(dialog.result))
