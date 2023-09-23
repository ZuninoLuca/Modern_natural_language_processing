from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Class to build the dataset for the reward model
class RewardModelDataset(Dataset):
    def __init__(
        self, tokenizer, dataset, maximum_sentence_length=512, use_label=False
    ):
        self.tokenizer = tokenizer
        self.padding_id = tokenizer.pad_token_id
        self.processed_samples = []
        self.maximum_sentence_length = maximum_sentence_length

        # Iterate over the dataset
        for entry in tqdm(dataset):
            # Get the chat and chat_ids
            chat = entry["chat"]

            # split the chat into sentences
            split_chats = chat.split("\n\n")

            chat_list = []
            for index, chat in enumerate(split_chats):
                chat_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(chat))
                chat_ids_input = tokenizer.build_inputs_with_special_tokens(chat_ids)

                # keep the first token for only index 0, then remove the first token for the other indexes
                if index > 0:
                    chat_ids_input = chat_ids_input[1:]

                chat_list.append(chat_ids_input)

            # transform the list of lists into a list
            chat_list = [item for sublist in chat_list for item in sublist]
            chat_list = chat_list[: self.maximum_sentence_length]

            # Get the label for the chat and the label_ids
            if use_label:
                label = entry["label"]
                label_ids = 0 if label == "negative" else 1
            else:
                label_ids = entry["grade"]

            # Append the processed samples (one for each answer)
            self.processed_samples.append(
                {
                    "chat": chat_list,
                    "label": label_ids,
                }
            )

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        return deepcopy(self.processed_samples[idx])

    def pad(self, inputs):
        return [
            inp + [self.padding_id] * (self.maximum_sentence_length - len(inp))
            for inp in inputs
        ]

    def get_sample(self, idx):
        return {
            "chat": self.processed_samples[idx]["chat"],
            "label": self.processed_samples[idx]["label"],
        }

    def collate_batch(self, batch):
        ids = [dictionary["chat"] for dictionary in batch]
        labels = [dictionary["label"] for dictionary in batch]
        return (
            torch.tensor(self.pad(ids)),
            torch.tensor(np.array(labels)),
        )
