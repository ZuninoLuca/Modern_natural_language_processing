import json
import logging
import sys

from reward_model import RewardModel, RewardModelConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")

checkpoint = "/home/nayabiakl/project-m3-modernnlm/bloom-test/best"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype="auto", device_map="auto"
)

model = model.cuda()


reward_model_path = "/home/nayabiakl/project-m3-modernnlm/gpt2_sft/reward_model/"
AutoConfig.register("xlm-roberta-base-finetuned-reward-model", RewardModelConfig)
AutoModel.register(RewardModelConfig, RewardModel)

reward_config = AutoConfig.from_pretrained(reward_model_path)
reward_model = AutoModel.from_pretrained(reward_model_path, config=reward_config)
reward_max_length = 1024


with open("Result-GPT2/answers.json") as f:
    questions = json.load(f)

score_dataset = []
for entry in questions:
    ans_dict = {"question": entry["question"], "answer": entry["answer"]}
    score_dataset.append(ans_dict)

    scores = reward_model.get_rewards(score_dataset)

    print(scores)
