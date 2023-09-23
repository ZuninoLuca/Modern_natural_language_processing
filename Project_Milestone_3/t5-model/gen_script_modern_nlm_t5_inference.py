import json

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


def question_answer(model, tokenizer, question, text, device):
    # tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text, max_length=512, truncation=True)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    # number of tokens in segment A (question)
    num_seg_a = sep_idx + 1
    # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s for segment embeddings
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model(
        torch.tensor([input_ids]).to(device),
        token_type_ids=torch.tensor([segment_ids]).to(device),
    )

    # reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    answer = ""
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    return answer


if __name__ == "__main__":
    print("Loading model..")
    tokenizer = T5Tokenizer.from_pretrained("final_model/tokenizer")
    model = T5ForConditionalGeneration.from_pretrained("final_model/model")
    print("Model loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inference_dataset = load_dataset(
        "json", data_files="prompts_context_final.jsonl", split="train"
    )
    # with open("prompts_context_final.jsonl", "r") as f:
    #     inference_dataset = json.load(f)

    answers = []
    with torch.no_grad():
        for entry in inference_dataset:
            question = entry["question"]
            context = entry["context"]
            if "choices" in entry:
                choices = entry["choices"]
                question += " choices: " + choices

            max_length = 512
            max_context_length = max_length - len(
                tokenizer.encode(question, truncation=True)
            )

            # truncate context if necessary
            context = context[:max_context_length]

            outputs = ""
            input_ids = tokenizer(
                f"question: {question}  context: {context}", return_tensors="pt"
            ).input_ids
            input_ids = input_ids.to(device)

            # output = model.generate(input_ids)
            output = model.generate(
                input_ids,
                max_length=1024,
                num_beams=5,
                repetition_penalty=2.0,
                eos_token_id=None,
            )
            output = tokenizer.decode(output[0], skip_special_tokens=True)

            dict = {"question": question, "answer": output}
            answers.append(dict)

        # save the questions and answers in dict to a json file
        with open("answers.json", "w") as f:
            json.dump(answers, f)
