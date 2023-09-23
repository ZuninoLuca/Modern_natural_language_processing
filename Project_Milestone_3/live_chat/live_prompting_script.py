import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import numpy as np
import random
import wikipediaapi
import regex as re


def generate_keywords_and_languages(question, model, tokenizer, num_return_sequences=10, num_beams=10):
    try:
        # Encode the question and return a tensor in Pytorch
        input_ids = tokenizer.encode('Keyword and Language of: ' + question, return_tensors="pt")

        # Generate a sequence of ids
        output_ids = model.generate(
            input_ids,
            max_length=10,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=3,
            num_beams=num_beams,
            early_stopping=True
        )

        # Decode the sequences
        keyword_and_language_pairs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        # Split the keyword and language
        keywords_and_languages = [pair.split("|") for pair in keyword_and_language_pairs]

    except Exception as e:
        keywords_and_languages = []

    return keywords_and_languages


def remove_parentheses(text):
    # Use regular expression to remove everything between parentheses
    pattern = r"\([^()]*\)"
    result = re.sub(pattern, "", text)
    return result


def get_context(question, model, tokenizer):

    # Generate the keywords and languages (for the Wikipedia search)
    keywords_list = generate_keywords_and_languages(question, model, tokenizer)

    context = ""
    finished = False

    # For each keyword and language in the list
    for keyword_and_language in keywords_list:
        # If the keyword and language are both present, use them
        if len(keyword_and_language) == 2:
            keyword, language = keyword_and_language
        # If only the keyword is present, use it and keep the language empty (to use both English and French Wikipedia)
        elif len(keyword_and_language) == 1:
            keyword = keyword_and_language[0]
            language = ""
        else:
            keyword = ""
            language = ""
        try:
            if language == "EN":
                # Use the English Wikipedia
                wiki_wiki = wikipediaapi.Wikipedia("en")
            elif language == "FR":
                # Use the French Wikipedia
                wiki_wiki = wikipediaapi.Wikipedia("fr")
            else:
                # Use both the English and French Wikipedia
                wiki_wiki_1 = wikipediaapi.Wikipedia("en")
                wiki_wiki_2 = wikipediaapi.Wikipedia("fr")
            if not finished:
                if language == "EN" or language == "FR" and keyword != "":
                    # Get the Wikipedia page for the keyword
                    page = wiki_wiki.page(keyword)
                    # If the page exists
                    if page.exists():
                        # If the page is a disambiguation page, skip it
                        if "may refer to" in page.text or "plusieurs concepts" in page.text or "dans les articles suivants" in page.text or "Suivant le contexte, le terme" in page.text:
                            pass
                        else:
                            # Get the summary of the page and use it as the context
                            context = page.summary
                            finished = True
                    else:
                        # If the page doesn't exist, try to remove the parentheses from the keyword
                        page = wiki_wiki.page(remove_parentheses(keyword))
                        if page.exists():
                            # If the page is a disambiguation page, skip it
                            if "may refer to" in page.text or "plusieurs concepts" in page.text or "dans les articles suivants" in page.text or "Suivant le contexte, le terme" in page.text:
                                pass
                            else:
                                # Get the summary of the page and use it as the context
                                context = page.summary
                                finished = True
                elif keyword != "":
                    page_en = wiki_wiki_1.page(keyword)
                    page_fr = wiki_wiki_2.page(keyword)
                    # If the page exists in English
                    if page_en.exists():
                        # If the page is a disambiguation page, skip it
                        if "may refer to" in page_en.text or "plusieurs concepts" in page_en.text or "dans les articles suivants" in page_en.text or "Suivant le contexte, le terme" in page_en.text:
                            pass
                        else:
                            # Get the summary of the page and use it as the context
                            context = page_en.summary
                            finished = True
                    # If the page exists in French
                    elif page_fr.exists():
                        # If the page is a disambiguation page, skip it
                        if "may refer to" in page_fr.text or "plusieurs concepts" in page_fr.text or "dans les articles suivants" in page_fr.text or "Suivant le contexte, le terme" in page_fr.text:
                            pass
                        else:
                            # Get the summary of the page and use it as the context
                            context = page_fr.summary
                            finished = True
                    else:
                        # If the page doesn't exist, try to remove the parentheses from the keyword
                        page_en = wiki_wiki_1.page(remove_parentheses(keyword))
                        page_fr = wiki_wiki_2.page(remove_parentheses(keyword))
                        # If the page exists in English
                        if page_en.exists():
                            # If the page is a disambiguation page, skip it
                            if "may refer to" in page_en.text or "plusieurs concepts" in page_en.text or "dans les articles suivants" in page_en.text or "Suivant le contexte, le terme" in page_en.text:
                                pass
                            else:
                                # Get the summary of the page and use it as the context
                                context = page_en.summary
                                finished = True
                        # If the page exists in French
                        elif page_fr.exists():
                            # If the page is a disambiguation page, skip it
                            if "may refer to" in page_fr.text or "plusieurs concepts" in page_fr.text or "dans les articles suivants" in page_fr.text or "Suivant le contexte, le terme" in page_fr.text:
                                pass
                            else:
                                # Get the summary of the page and use it as the context
                                context = page.summary
                                finished = True
        except Exception as e:
            pass
    return context


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
    # Set the seed value
    seed_value = 0

    random.seed(seed_value) # Python
    np.random.seed(seed_value) # numpy
    torch.manual_seed(seed_value) # PyTorch

    # If a GPU is used, set the seed for it as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("This script allows to test the QA model and the context retrieval model.")
        
    print("Loading the QA model from local...")
    tokenizer = T5Tokenizer.from_pretrained(
        "FLAN-T5/FLAN-T5/google/flan-t5-base/tokenizer/checkpoint-10"
    )
    model = T5ForConditionalGeneration.from_pretrained(
        "FLAN-T5/FLAN-T5/google/flan-t5-base/model/checkpoint-10/"
    )
    print("QA model loaded.")

    print("Loading keyword generator model from HuggingFace...")
    model_k = T5ForConditionalGeneration.from_pretrained("lucazed/keyword-generator-complete")
    tokenizer_k = T5Tokenizer.from_pretrained("lucazed/keyword-generator-complete")
    print("Keyword generator model loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    while True:
        question = input("Enter a question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        context = get_context(question, model_k, tokenizer_k)
        print(f'------------------------------------')
        print(f'Generated context: {context}')
        print(f'------------------------------------')

        max_length = 512
        max_context_length = max_length - len(
            tokenizer.encode(question, truncation=True)
        )

        # truncate context if necessary
        context = context[:max_context_length]

        with torch.no_grad():
            input_ids = tokenizer(
                f"question: {question}  context: {context}", return_tensors="pt"
            ).input_ids
            input_ids = input_ids.to(device)

            output = model.generate(input_ids, max_length=1024, eos_token_id=None)
            output = tokenizer.decode(output[0], skip_special_tokens=True)

            print(f'Answer: {output}')
            print(f'------------------------------------')