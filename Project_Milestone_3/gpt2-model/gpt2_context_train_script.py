import json
import os

from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


def tokenize_inputs(example, tokenizer, max_length=1024):
    # check how much of answer is kept in the input with max length for each entry in the batch
    qst_len = len(example["question"])
    context_len = len(example["context"])

    # We will pad the inputs and return tensors
    encodings = tokenizer(
        example["question"],
        example["context"],
        example["answer"],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    labels = encodings["input_ids"].clone()
    qs_token_index = qst_len + context_len
    labels[:, :qs_token_index] = -100

    encodings["labels"] = labels
    return encodings


def flatten_and_concatenate(nested_list):
    # If the input is a string, return it
    if isinstance(nested_list, str):
        return nested_list

    # If the input is a list, apply the function to each element and concatenate
    # the results
    if isinstance(nested_list, list):
        return ", ".join(flatten_and_concatenate(element) for element in nested_list)

    # If the input is neither a string nor a list, return an empty string
    return " "


def main():
    max_length = 1024
    root_dir = os.getcwd() + "/"
    train_data = []
    with open(
        root_dir
        + "context_train_datasets/gen_dataset_modern_nlm_texas_SFT_with_context.json"
    ) as f:
        train_data += json.load(f)

    with open(
        root_dir
        + "context_train_datasets/gen_dataset_modern_nlm_eli5_questions_answers_with_context.json"
    ) as f:
        eli5_data = json.load(f)
        # take 10% of the data
        eli5_data = eli5_data[: int(len(eli5_data) * 0.1)]
        train_data += eli5_data

    with open(
        root_dir
        + "context_train_datasets/gen_dataset_modern_nlm_class_dataset_with_context.json",
        encoding="cp1252",
        errors="ignore",
    ) as f:
        class_data = json.load(f)
        class_data = class_data[: int(len(class_data) * 0.1)]

    class_data_list = []
    for item in class_data:
        if "question" not in item or "answer" not in item:
            continue

        question = item["question"]
        answer = flatten_and_concatenate(item["answer"])
        context = item["context"]

        class_data_list.append(
            {"question": question, "context": context, "answer": answer}
        )
    train_data += class_data_list

    with open(
        root_dir
        + "context_train_datasets/gen_dataset_modern_nlm_stack_exchange_with_context.json"
    ) as f:
        stack_data = json.load(f)
        stack_data = stack_data[: int(len(stack_data) * 0.1)]
        train_data += stack_data

    with open(
        root_dir
        + "context_train_datasets/gen_dataset_modern_nlm_sciq_with_context.json"
    ) as f:
        sciq_data = json.load(f)
        sciq_data = sciq_data[: int(len(sciq_data) * 0.1)]
        train_data += sciq_data

    with open(
        root_dir + "context_train_datasets/gen_dataset_modern_nlm_ai_with_context.json"
    ) as f:
        ai_data = json.load(f)
        ai_data = ai_data[: int(len(ai_data) * 0.1)]
        train_data += ai_data

    with open(
        root_dir + "context_train_datasets/gen_modern_nlm_QandA_definitions_EN.json"
    ) as f:
        QandA_data = json.load(f)
        QandA_data = QandA_data[: int(len(QandA_data) * 0.1)]
        train_data += QandA_data

    with open(
        root_dir + "context_train_datasets/gen_modern_nlm_QandA_definitions_FR.json"
    ) as f:
        QandA_data = json.load(f)
        QandA_data = QandA_data[: int(len(QandA_data) * 0.1)]
        train_data += QandA_data

    # # make sure to keep the answer and truncate from the context
    for item in train_data:
        qst_len = len(item["question"])
        context_len = (
            len(item["context"]) if "context" in item and item["context"] != None else 0
        )
        answer_len = len(item["answer"])
        if qst_len + context_len + answer_len > max_length:
            context_len_remaining = max_length - qst_len - answer_len
            if context_len_remaining < 0:
                item["context"] = ""
            else:
                item["context"] = item["context"][:context_len_remaining]

    train_data = {
        "question": [item["question"] for item in train_data],
        "context": [
            item["context"] if "context" in item and item["context"] != None else ""
            for item in train_data
        ],
        "answer": [item["answer"] for item in train_data],
    }

    dataset = Dataset.from_dict(train_data)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.sep_token = "<SEP>"
    # tokenizer.cls_token = "<CLS>"

    # dataset = dataset.map(tokenize_inputs, batched=True)
    dataset = dataset.map(
        lambda example: tokenize_inputs(example, tokenizer, max_length=max_length),
        batched=True,
    )

    # Format the dataset to outputs suitable for training
    # columns = ["input_ids", "attention_mask", "labels"]
    # dataset.set_format(type="torch", columns=columns)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        evaluation_strategy="no",
        save_strategy="no",
        do_eval=False,
    )
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # model.resize_token_embeddings(len(tokenizer) + 2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    # Train the model
    trainer.train()
    model_path = "gpt2_lm_model_context"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained("gpt2_lm_tokenizer_context")


if __name__ == "__main__":
    main()
