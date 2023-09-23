import json

import torch

# Here we load your CustomRewardModelConfig and CustomRewardModel classes,
# so we have the implementation of your get_rewards function
# and load the weights from your saved model.
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel


def save_hf_model(hf_model, model_path):
    """Save the model and tokenizer to the specified path"""
    hf_model.save_pretrained(model_path)
    hf_model.config.save_pretrained(model_path)


def load_json(filename):
    """Load json file"""
    with open(filename, "r") as read_file:
        data = json.load(read_file)
    return data


def save_dictlist_to_json(mydictlist, filename):
    """Save a list of dictionaries to json file"""
    f = open(filename, "w", encoding="utf-8")
    json.dump(mydictlist, f, ensure_ascii=False, indent=4)
    f.close()


class Reward(torch.nn.Module):
    """
    Wrapper class for the reward model,
    which handles loading the model and tokenizers,
    and the forward pass for final predictions
    """

    def __init__(self, model_path):
        super().__init__()

        # Load student-defined reward model and its associated config
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, config=self.config)

    def check_reward_type(self, rewards):
        return isinstance(rewards, list) and all(isinstance(r, dict) for r in rewards)

    def forward(self, demonstrations):
        """
        Get the reward predictions from student's reward model
        Args:
            demonstrations: list of dicts in the format of
            {'chosen': str, 'rejected': str}
        Return:
            rewards: list of dicts in the format of
            {'chosen': float, 'rejected': float}
        """
        # ===== Get the rewards from student's reward model =====
        # NOTE: You should implement the "get_rewards" method in your reward model
        rewards = self.model.get_rewards(demonstrations)

        # ===== Check the reward format =====
        # assert self.check_reward_type(rewards), "The rewards must be a list of dicts"
        assert len(rewards) == len(
            demonstrations
        ), "The number of rewards must match the number of demonstration pairs"
        return rewards


class TestDataset(Dataset):
    """Simple dataset module for testing the reward model"""

    def __init__(self, test_ds, maximum_sentence_length=512):
        self.ds = test_ds
        self.maximum_sentence_length = maximum_sentence_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ix):
        self.ds[ix]["question"] = self.ds[ix]["question"][
            : self.maximum_sentence_length - 2
        ]
        self.ds[ix]["answer"] = self.ds[ix]["answer"][
            : self.maximum_sentence_length - 2
        ]
        return self.ds[ix]


class Evaluator:
    def __init__(self, model_path, ds_test):
        # Load the model and dataset
        self.load_model(model_path)
        self.ds_test = ds_test
        self.dataset = TestDataset(ds_test)
        self.dataloader = DataLoader(
            self.dataset, batch_size=2, shuffle=False, collate_fn=lambda x: x
        )

    def load_model(self, model_path):
        """Load the reward model from the specified path"""
        self.model = Reward(model_path)

    def evaluate(self):
        """Evaluate the model on the test dataset"""
        rewards = []
        for batch in tqdm(self.dataloader):
            rewards.extend(self.model(batch))

        # ===== Check the rewards by doing pair-wise ranking =====
        # num_correct = sum(reward["chosen"] > reward["rejected"] for reward in rewards)
        # acc = num_correct / len(self.ds_test)
        # print(f"Evaluation Complete, Accuracy: {acc}")
        avg_score = sum(reward for reward in rewards) / len(rewards)
        print(f"Evaluation Complete, Average score: {avg_score}")
