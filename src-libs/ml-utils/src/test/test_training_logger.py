from __future__ import annotations
import sys

sys.path.append("..")
sys.path.append("../ml_utils")

from ml_utils.config_dict_data_recorder import ConfigDictDataRecorder
from ml_utils.filename_maker import FileNameMaker
from ml_utils.training_logger import TrainingLogger

##############################################

config = {"c1": "a", "c2": "b", "c3" : "c"}
dict_pair = {"a": "1", "b": "2", "c" : "3"}

def test_config_dict_data_recorder():
    #log file name
    file_names = FileNameMaker.make(config, "config-data-record")
    print("Generated file name: " + file_names.fullname)
     
    #training logger
    training_logger = ConfigDictDataRecorder(file_names.fullname)

    try:
        #begin log
        training_logger.begin(config)

        #record log
        for i in range(0, 9):
            training_logger.record(dict_pair)

    except Exception as e:
        print(f"Something ERROR {e}")

    finally:
        #log end
        training_logger.end()

    return


def test_training_logger():
    #training logger
    training_logger = TrainingLogger("training-log", config)

    try:
        #begin log
        training_logger.begin()

        #record log
        for i in range(0, 9):
            training_logger.record(dict_pair)

    except Exception as e:
        print(f"Something ERROR {e}")

    finally:
        #log end
        training_logger.end()

    return


def main():
    print("test ConfigDictDataRecorder")
    test_config_dict_data_recorder()

    print("test TrainingLogger")
    test_training_logger()
    return

if __name__ == "__main__":
    main()
