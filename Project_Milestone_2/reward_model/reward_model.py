import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    get_constant_schedule_with_warmup,
)
from utils import Evaluator, save_hf_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class RewardModelConfig(XLMRobertaConfig):
    model_type = "xlm-roberta-base-finetuned-reward-model"
    model_name = "lucazed/xlm-roberta-base-finetuned-questions"
    hidden_size = 768


class RewardModel(XLMRobertaModel):
    config_class = RewardModelConfig

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)
        self.model = XLMRobertaModel.from_pretrained(config.model_name)

        self.fc1 = torch.nn.Linear(config.hidden_size, int(config.hidden_size / 2))
        self.fc2 = torch.nn.Linear(int(config.hidden_size / 2), 1)

    def forward(self, chat_input):
        # output = super().forward(chat_input)
        output = self.model(chat_input)

        # add linear layers for regression task
        output = self.fc1(output.last_hidden_state[:, 0, :])
        output = torch.nn.functional.relu(output)
        output = self.fc2(output)

        # output score should be between 0 and 5 for regression
        output = torch.sigmoid(output) * 5.0

        return output

    def get_rewards(self, chat_input):
        rewards = []
        for pair in chat_input:
            chosen = pair["chosen"]
            rejected = pair["rejected"]

            chosen_split = chosen.split("\n\n")
            rejected_split = rejected.split("\n\n")

            chosen_list = []
            rejected_list = []

            for index, chosen in enumerate(chosen_split):
                chosen_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(chosen)
                )
                chosen_ids_input = self.tokenizer.build_inputs_with_special_tokens(
                    chosen_ids
                )

                if index > 0:
                    chosen_ids_input = chosen_ids_input[1:]

                chosen_list.append(chosen_ids_input)

            for index, rejected in enumerate(rejected_split):
                rejected_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(rejected)
                )
                rejected_ids_input = self.tokenizer.build_inputs_with_special_tokens(
                    rejected_ids
                )
                if index > 0:
                    chosen_ids_input = rejected_ids_input[1:]

                rejected_list.append(rejected_ids_input)

            # transform the list of lists into a list then to a tensor
            chosen_list = [item for sublist in chosen_list for item in sublist]
            chosen_list = chosen_list[:512]
            chosen_list = torch.tensor(chosen_list).unsqueeze(0)

            rejected_list = [item for sublist in rejected_list for item in sublist]
            rejected_list = rejected_list[:512]
            rejected_list = torch.tensor(rejected_list).unsqueeze(0)

            scores_chosen = self.forward(chosen_list)
            scores_reject = self.forward(rejected_list)
            rewards.append(
                {"chosen": scores_chosen.item(), "rejected": scores_reject.item()}
            )
        return rewards


# Function to train the model
def train(
    model,
    train_data,
    val_data,
    device,
    epochs_num,
    batch_size,
    lr,
    warmup_p,
    max_gradient_norm,
    save_path,
    model_name,
):
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=train_data.collate_batch,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(warmup_p * epochs_num * len(train_dataloader)),
    )

    # Instantiate the custom loss function
    loss_function = nn.MSELoss()

    model.zero_grad()

    # Define an empty list to store the train loss values
    train_loss_values = []

    for epoch in range(epochs_num):
        model.train()

        total_loss = 0
        train_step = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            model.zero_grad()
            train_step = train_step + 1
            chat, labels = tuple(input_t.to(device) for input_t in batch)

            # The model returns the scores for the batch
            # output format: (batch_size, 1)
            output = model(chat)
            output = output.squeeze()
            loss = loss_function(output, labels.float())

            # use MSE loss instead
            loss.backward()

            # Clip the gradients for stability
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_gradient_norm
            )
            total_loss = total_loss + loss.item()
            optimizer.step()
            scheduler.step()

        train_loss = total_loss / train_step

        # Append the train loss to the list
        train_loss_values.append(train_loss)

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss:.3f}")

        save_hf_model(model, save_path)

        torch.save(
            model.state_dict(),
            save_path
            + "learning_rate_{}-warmup_{}-model_{}.pt".format(lr, warmup_p, model_name),
        )
        #     print("Best model saved at epoch {}".format(epoch))

        print("-----------------------------------------------")

    # save the model
    torch.save(
        model.state_dict(),
        save_path
        + "learning_rate_{}-warmup_{}-model_{}.pt".format(lr, warmup_p, model_name),
    )

    with open("train_loss.txt", "w") as f:
        for loss in train_loss_values:
            f.write(str(loss) + "\n")

    # Plot the train loss values
    plt.plot(range(1, epochs_num + 1), train_loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss over Epochs")
    plt.show()

    # Save the plot as an image file
    plt.savefig("train_loss_plot.png")

    # save model
    save_hf_model(model, save_path)
    AutoConfig.register("xlm-roberta-base-finetuned-reward-model", RewardModelConfig)
    AutoModel.register(RewardModelConfig, RewardModel)

    evaluator = Evaluator(save_path, val_data)
    evaluator.evaluate()
