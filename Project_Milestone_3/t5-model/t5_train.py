from __future__ import print_function

import MyDataset
import torch

# from t5_dataset import Dataset
from MyDataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)

from datasets import concatenate_datasets, load_dataset


def train(
    model: T5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizer,
    optimizer: AdamW,
    train_set: Dataset,
    validation_set: Dataset,
    num_train_epochs: int,
    device: str,
    batch_size: int,
    max_input_length: int = 512,
):
    """_summary_

    Args:
        model (T5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        optimizer (AdamW): _description_
        train_set (Dataset): _description_
        validation_set (Dataset): _description_
        num_train_epochs (int): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    my_trainset_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=lambda data: train_set.pack_minibatch(data),
    )
    my_validation_dataloader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=lambda data: validation_set.pack_minibatch(data),
    )

    # set training mode on the model
    model.train()

    # model to device
    model.to(device)

    f1_old: int = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.0
        for contexts, questions, answers in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            inputs = list(
                map(
                    lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}",
                    zip(questions, contexts),
                )
            )
            encoded_inputs = tokenizer(
                inputs,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = tokenizer(
                answers,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )

            input_ids, attention_mask = (
                encoded_inputs.input_ids,
                encoded_inputs.attention_mask,
            )
            encoded_targets = encoded_targets.input_ids

            # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
            encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

            input_ids = input_ids.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=encoded_targets,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                model_predictions_encoded = []
                target_encoded = []
                for contexts, questions, answers in tqdm(my_validation_dataloader):
                    inputs = list(
                        map(
                            lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}",
                            zip(questions, contexts),
                        )
                    )
                    encoded_inputs = tokenizer(
                        inputs,
                        padding="longest",
                        max_length=max_input_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoded_targets = tokenizer(
                        answers,
                        padding="longest",
                        max_length=max_input_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoded_inputs, attention_mask = (
                        encoded_inputs.input_ids,
                        encoded_inputs.attention_mask,
                    )
                    encoded_targets = encoded_targets.input_ids

                    encoded_inputs = encoded_inputs.to(device)
                    encoded_targets = encoded_targets.to(device)
                    attention_mask = attention_mask.to(device)
                    model_predictions = model.generate(
                        input_ids=encoded_inputs,
                        max_length=1024,
                        attention_mask=attention_mask,
                    )

                    model_predictions_encoded += model_predictions.tolist()
                    target_encoded += encoded_targets.tolist()
            f1, exact_match = validation_set.evaluate(
                model_predictions_encoded, target_encoded
            )

            print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
            if f1 > f1_old:
                model.save_pretrained(f"results/{model.name_or_path}/model/best-f1")
                tokenizer.save_pretrained(
                    f"results/{model.name_or_path}/tokenizer/best-f1"
                )
                f1_old = f1
        if (epoch + 1) % 3 == 0:
            model.save_pretrained(
                f"results/{model.name_or_path}/model/checkpoint-{epoch+1}"
            )
            tokenizer.save_pretrained(
                f"results/{model.name_or_path}/tokenizer/checkpoint-{epoch+1}"
            )
        model.train()

    model.save_pretrained(f"results/{model.name_or_path}/model/checkpoint-{epoch+1}")
    tokenizer.save_pretrained(
        f"results/{model.name_or_path}/tokenizer/checkpoint-{epoch+1}"
    )


class Args:
    t5_model = "t5-base"
    batch_size = 2
    epochs = 10
    lr = 1e-4
    workers = 1
    max_input_length = 512
    seed = 7


if __name__ == "__main__":
    args = Args()

    # Set seed
    set_seed(args.seed)

    # Load the datasets
    datasets = []

    import os

    folder_path = "DATASETS/"
    file_paths = []

    # List all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    train_datasets = []
    test_datasets = []
    for file_path in file_paths:
        dataset = load_dataset("json", data_files=file_path, split="train")
        # dataset = dataset[: int(len(dataset) * 0.1)]
        print(f"Loaded dataset from {file_path} is of type {type(dataset)}")

        split_dataset = dataset.train_test_split(test_size=0.05)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        train_dataset = train_dataset.shard(num_shards=2, index=0)
        test_dataset = test_dataset.shard(num_shards=2, index=0)

        print(f"Train dataset is of type {type(train_dataset)}")
        print(f"Test dataset is of type {type(test_dataset)}")

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    # Concatenate all training datasets and all test datasets
    train_data = concatenate_datasets(train_datasets)
    test_data = concatenate_datasets(test_datasets)

    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

    # creating the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create an instance of DatasetMap
    my_dataset_map = MyDataset.DatasetMap(tokenizer)
    train_set = Dataset(train_data, tokenizer, parser=my_dataset_map.my_dataset)
    validation_set = Dataset(test_data, tokenizer, parser=my_dataset_map.my_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_set=train_set,
        validation_set=validation_set,
        num_train_epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
    )
