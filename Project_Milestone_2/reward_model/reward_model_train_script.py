import json
import os
import random

import numpy as np
import torch
from reward_model_dataset import RewardModelDataset
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer

from reward_model import RewardModel, RewardModelConfig, train


def main():
    # get the current directory
    root_dir = os.getcwd() + "/"

    seed = int(0)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Random seed (for reproducibility): ", seed)

    # use our own fine-tuned xlm roberta model
    model_name = "lucazed/xlm-roberta-base-finetuned-questions"
    config = RewardModelConfig(model_name=model_name)
    reward_model = RewardModel(config=config)
    tokenizer = reward_model.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)

    # read the train dataset from all json files and concatenate them
    train_data = []

    # Texas train dataset
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_texas_train_data1.json",
        "r",
    ) as f:
        train_data += json.load(f)
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_texas_train_data2.json",
        "r",
    ) as f:
        train_data += json.load(f)
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_texas_train_data3.json",
        "r",
    ) as f:
        train_data += json.load(f)
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_texas_train_data4.json",
        "r",
    ) as f:
        train_data += json.load(f)
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_texas_train_data5.json",
        "r",
    ) as f:
        train_data += json.load(f)
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_texas_train_data6.json",
        "r",
    ) as f:
        train_data += json.load(f)
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_QandA_bis_formatted_train.json",
        "r",
    ) as f:
        train_data += json.load(f)
    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_class_processed_train_data.json",
        "r",
    ) as f:
        train_data += json.load(f)

    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_hh-rlhf_train_detoxified.json",
        "r",
    ) as f:
        hh_data = json.loads(f.read())

    with open(
        root_dir
        + "train_dataset_final/m2_reward_dataset_modern_nlm_synthetic-instruct-gptj-pairwise_train.json",
        "r",
    ) as f:
        synthetic_data = json.load(f)
        # take only the first 10000 examples
        synthetic_data = synthetic_data[:10000]

    train_data += hh_data
    train_data += synthetic_data

    eval_data = []
    with open(
        root_dir
        + "eval_dataset_final/m2_reward_dataset_modern_nlm_QandA_bis_formatted_chosen_rejected_val.json",
        "r",
    ) as f:
        eval_data += json.load(f)

    reward_model_dataset_train = RewardModelDataset(
        tokenizer=tokenizer,
        dataset=train_data,
        maximum_sentence_length=512,
        use_label=False,
    )

    train(
        model=reward_model,
        train_data=reward_model_dataset_train,
        val_data=eval_data,
        device=device,
        epochs_num=5,
        batch_size=8,
        lr=0.00001,
        warmup_p=0.1,
        max_gradient_norm=1.0,
        save_path="reward_model/",
        model_name="roberta_finetuned_reward_model",
    )


if __name__ == "__main__":
    main()
