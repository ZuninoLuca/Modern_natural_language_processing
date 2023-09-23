import json

with open("answers_fixed_format_1.json", "r") as file:
    data = json.load(file)

cleaned_data = []
# Truncate the answer, remove <CLS> tokens, and take the first occurrence of "Provided answer"
for item in data:
    answer = item["answer"][0]
    start_index = answer.find("Provided Answer:")

    if start_index != -1:
        start_index += len("Provided Answer:")
        truncated_answer = answer[start_index:].strip().replace("<CLS>", "")
        item["answer"][0] = truncated_answer
        cleaned_data.append(item)

print("Original data length: ", len(data))
print("Cleaned data length: ", len(cleaned_data))
with open("cleaned_gpt2.json", "w") as file:
    json.dump(cleaned_data, file, indent=4)
