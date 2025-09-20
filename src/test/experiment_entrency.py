from __future__ import annotations

# Init startup path, change current path to startup python file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

# import
from usyd_learning.ml_utils import console
from fl_lora_sample.lora_sample_entry import SampleAppEntry
from usyd_learning.ml_utils.model_utils import ModelUtils
from usyd_learning.ml_utils.training_utils import TrainingUtils

def main(config_path: str = "./fl_lora_sample/script_test-rbla.yaml"):
    g_app = SampleAppEntry()
    # Load app config set from yaml file
    g_app.load_app_config(config_path)
    device = ModelUtils.accelerator_device()

    # Run app
    g_app.run(device, g_app.training_rounds)

if __name__ == "__main__":
    main()