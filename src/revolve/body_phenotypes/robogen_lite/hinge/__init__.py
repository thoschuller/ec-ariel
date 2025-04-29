import json
from pathlib import Path

# Open json file and read into dict
CD = Path(__file__).parent
with (CD / "physical_parameters.jsonc").open() as f:
    HINGE_PHYSICAL_PARAMS = json.load(f)
