import argparse

from transformers import AutoConfig, AutoModel
from utils import Evaluator, load_json

from reward_model import RewardModel, RewardModelConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="reward_model/",
        help="Path to the model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="eval_dataset_final/m2_reward_dataset_modern_nlm_QandA_bis_formatted_chosen_rejected_val.json",
        help="Path to the test dataset",
    )
    args = parser.parse_args()

    reward_dataset = load_json(args.data_path)

    AutoConfig.register("xlm-roberta-base-finetuned-reward-model", RewardModelConfig)
    AutoModel.register(RewardModelConfig, RewardModel)

    evaluator = Evaluator(args.model_path, reward_dataset)
    evaluator.evaluate()
