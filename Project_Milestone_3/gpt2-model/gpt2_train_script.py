from __future__ import print_function

import argparse
import json
import os

import torch
from SFT_dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    set_seed,
)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="CLI for training T5 T2T model")

    parser.add_argument(
        "--t5_model",
        type=str,
        default="t5-base",
        help="What type of T5 model do you want use?",
    )

    parser.add_argument("--batch_size", type=int, default=4, help="mini-batch size")

    parser.add_argument(
        "--epochs", type=int, default=5, help="number of training epochs (default: 40)"
    )

    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (Adam) (default: 1e-4)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of working units used to load the data",
    )

    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="Maximum lenght of input text",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for random initialization (default: 7)",
    )

    parsed_arguments = parser.parse_args()

    return parsed_arguments


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
        model (GPT2LMHeadModel): _description_
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
        collate_fn=train_set.pack_minibatch,
        shuffle=True,
    )
    my_validation_dataloader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=validation_set.pack_minibatch,
        shuffle=False,
    )

    # set training mode on the model
    model.train()

    # model to device
    model.to(device)

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    f1_old: float = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.0
        for questions, answers in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            # inputs = list(questions)
            # inputs should have the following format
            # "question: <question>  context: <answer>"
            # inputs = [
            #     "question: " + q + " context: " + a for q, a in zip(questions, answers)
            # ]

            inputs = ["question: " + q for q in questions]

            # split each question into a list of sentences
            for i, q in enumerate(inputs):
                split_qst = q.split(". ")

                chat_list = []
                for index, chat in enumerate(split_qst):
                    chat_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(chat))
                    chat_ids_input = tokenizer.build_inputs_with_special_tokens(
                        chat_ids
                    )

                    # then remove the first token
                    chat_ids_input = chat_ids_input[1:]
                    chat_list.append(chat_ids_input)

                # transform the list of lists into a list
                chat_list = [item for sublist in chat_list for item in sublist]
                chat_list = chat_list[:max_input_length]

                # pad to max length
                chat_list = chat_list + [0] * (max_input_length - len(chat_list))

                inputs[i] = torch.tensor(chat_list).unsqueeze(0).to(device)

            # transform the list into a tensor
            inputs = torch.cat(inputs, dim=0)

            # encoded_inputs = tokenizer(
            #     inputs,
            #     padding="longest",
            #     max_length=max_input_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )

            # add the pad token manually to the targets
            targets = [tokenizer.pad_token + " " + x for x in answers]
            # encoded_targets = tokenizer(
            #     targets,
            #     padding="longest",
            #     max_length=max_input_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )

            # split each question into a list of sentences
            for i, q in enumerate(targets):
                split_qst = q.split(". ")

                chat_list = []
                for index, chat in enumerate(split_qst):
                    chat_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(chat))
                    chat_ids_input = tokenizer.build_inputs_with_special_tokens(
                        chat_ids
                    )

                    # then remove the first token
                    chat_ids_input = chat_ids_input[1:]
                    chat_list.append(chat_ids_input)

                # transform the list of lists into a list
                chat_list = [item for sublist in chat_list for item in sublist]
                chat_list = chat_list[:max_input_length]

                # pad to max length
                chat_list = chat_list + [0] * (max_input_length - len(chat_list))

                targets[i] = torch.tensor(chat_list).unsqueeze(0).to(device)

            # transform the list into a tensor
            targets = torch.cat(targets, dim=0)

            # input_ids, attention_mask = (
            #     encoded_inputs.input_ids.to(device),
            #     encoded_inputs.attention_mask.to(device),
            # )

            # encoded_targets = encoded_targets.input_ids.to(device)

            # replace padding target token id's of the labels by -100, crossEntropy
            # skip target label == -100
            targets[targets == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=inputs,
                # attention_mask=attention_mask,
                labels=targets,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        # model.eval()
        # with torch.no_grad():
        #     model_predictions_encoded = []
        #     target_encoded = []
        #     for questions, answers in tqdm(my_validation_dataloader):
        #         inputs = list(questions)

        #         encoded_inputs = tokenizer(
        #             inputs,
        #             padding="longest",
        #             max_length=max_input_length,
        #             truncation=True,
        #             return_tensors="pt",
        #         )
        #         encoded_targets = tokenizer(
        #             answers,
        #             padding="longest",
        #             max_length=max_input_length,
        #             truncation=True,
        #             return_tensors="pt",
        #         )
        #         encoded_inputs, attention_mask = (
        #             encoded_inputs.input_ids,
        #             encoded_inputs.attention_mask,
        #         )
        #         encoded_targets = encoded_targets.input_ids

        #         encoded_inputs = encoded_inputs.to(device)
        #         encoded_targets = encoded_targets.to(device)
        #         attention_mask = attention_mask.to(device)
        #         model_predictions = model.generate(
        #             input_ids=encoded_inputs,
        #             attention_mask=attention_mask,
        #             max_length=200,
        #             num_beams=5,
        #             repetition_penalty=5.0,
        #         )

        #         model_predictions_encoded += model_predictions.tolist()
        #         target_encoded += encoded_targets.tolist()
        # f1, exact_match = validation_set.evaluate(
        #     model_predictions_encoded, target_encoded
        # )

        # print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        # if f1 > f1_old:
        #     model.save_pretrained(f"results/{model.name_or_path}/model/best-f1")
        #     tokenizer.save_pretrained(f"results/{model.name_or_path}/tokenizer/best-f1")
        #     f1_old = f1
        if epoch + 1 % 10 == 0:
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


if __name__ == "__main__":
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + "=" + str(v))

    # Set seed
    set_seed(args.seed)

    # Read data from json file
    root_dir = os.getcwd() + "/"

    train_data = []
    eval_data = []
    # with open(
    #     root_dir + "train_datasets/gen_dataset_modern_nlm_class_dataset.json",
    #     encoding="cp1252",
    #     errors="ignore",
    # ) as f:
    #     train_data += json.loads(f.read())

    with open(root_dir + "train_datasets/gen_dataset_modern_nlm_texas_SFT.json") as f:
        train_data += json.load(f)

    with open(root_dir + "train_datasets/gen_dataset_modern_nlm_texas_SFT.json") as f:
        eval_data += json.load(f)

    # with open(
    #     root_dir + "train_datasets/gen_dataset_modern_nlm_eli5_questions_answers.json"
    # ) as f:
    #     train_data += json.load(f)

    # with open(
    #     root_dir + "train_datasets/gen_dataset_modern_nlm_hh_questions_answers.json"
    # ) as f:
    #     train_data += json.load(f)

    # with open(
    #     root_dir
    #     + "train_datasets/gen_dataset_modern_nlm_synthetic_questions_answers.json"
    # ) as f:
    #     train_data += json.load(f)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    # creating the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_set = Dataset(train_data, tokenizer)
    validation_set = Dataset(eval_data, tokenizer)

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
        max_input_length=args.max_input_length,
    )
