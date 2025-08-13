from __future__ import annotations

from .model_trainer import ModelTrainer, ModelTrainerArgs

class ModelTrainerFactory:
    """
    Model trainer factory
    """
    
    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> ModelTrainerArgs:
        """
        Static method to create trainer args
        """
        return ModelTrainerArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(args: ModelTrainerArgs) -> ModelTrainer:
        """
        Static method to create trainer
        """

        match args.trainer_type:
            case "standard":
                from .trainer._model_trainer_standard import ModelTrainer_Standard
                return ModelTrainer_Standard(args)
            case _:
                raise Exception(f"Undeinfed trainer type {args.trainer_type}.")

        return None
