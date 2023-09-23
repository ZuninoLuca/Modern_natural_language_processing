# Import necessary libraries
import json

import torch
from gpt2_lm_train_script import flatten_and_concatenate
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Create a function that can run the model on a given question and context
def ask_question(model, tokenizer, question, context, device):
    # # Combine the question and context
    input_text = "%s %s" % (question, context)

    # # Encode the input text
    # input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    input_ids = tokenizer(
        input_text,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Run the model
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_length=1024,
            num_beams=5,
            repetition_penalty=5.0,
            length_penalty=0.5,
            # no_repeat_ngram_size=2,
            # early_stopping=True,
            # do_sample=True,
            # top_k=50,
            # top_p=0.95,
            # temperature=0.7
        )

    # Decode the output
    answer = tokenizer.batch_decode(output, skip_special_tokens=True)

    return answer


def main():
    # Make sure to use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define a question and context
    with open(
        "context_train_datasets/prompts_context.json",
        "r",
    ) as f:
        questions = json.load(f)

    # with open(
    #     "context_train_datasets/gen_dataset_modern_nlm_texas_SFT_with_context.json"
    # ) as f:
    #     questions = json.load(f)

    # with open("train_datasets/gen_dataset_modern_nlm_ai.json") as f:
    #     ai_data = json.load(f)
    #     ai_data = ai_data[: int(len(ai_data) * 0.1)]
    #     questions = ai_data

    prompts_data_list = []
    for item in questions:
        if "question" not in item or "answer" not in item:
            continue

        question = item["question"]
        # guid = item["guid"]
        answer = flatten_and_concatenate(item["answer"])
        context = flatten_and_concatenate(item["context"])

        # append to the dict
        prompts_data_list.append(
            {
                "question": question,
                "context": context,
                "answer": answer,
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2_lm_tokenizer_context")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = "<SEP>"
    tokenizer.cls_token = "<CLS>"
    tokenizer.padding_side = "left"

    model = GPT2LMHeadModel.from_pretrained("gpt2_lm_model_context").to(device)
    # Use the base model to answer the question
    answers = []
    for entry in prompts_data_list:
        question = entry["question"][:1024] + " <SEP> "

        context = entry["context"][:1024] + " <CLS> "
        answer = ask_question(model, tokenizer, question, context, device)

        dict = {
            # "guid": guid,
            "question": question,
            "answer": answer,
        }
        answers.append(dict)

    with open("answers.json", "w") as f:
        json.dump(answers, f, indent=4)


if __name__ == "__main__":
    main()
