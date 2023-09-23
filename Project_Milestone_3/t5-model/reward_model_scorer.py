import json
import logging
import sys

from reward_model import RewardModel, RewardModelConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")


reward_model_path = "/home/mariamhhegazy/project-m2-modernnlm/reward_model/"
AutoConfig.register("xlm-roberta-base-finetuned-reward-model", RewardModelConfig)
AutoModel.register(RewardModelConfig, RewardModel)

reward_config = AutoConfig.from_pretrained(reward_model_path)
reward_model = AutoModel.from_pretrained(reward_model_path, config=reward_config)
reward_max_length = 1024


with open("cleaned_gpt2.json") as f:
    questions = json.load(f)

score_dataset = []
counter = 0
for entry in questions:
    print("counter: ", counter)
    ans_dict = {"question": entry["question"], "answer": entry["answer"]}
    score_dataset.append(ans_dict)
    counter += 1

scores = reward_model.get_rewards(score_dataset)
print("scores ", scores)
print("Scores length: ", len(scores))
print("dataset length: ", len(score_dataset))
print("average score: ", sum(scores) / len(scores))
