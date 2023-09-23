## Fine tune on Language Modeling task

To fine tune the "xlm-roberta-base" model on a language modeling task, the "Modern_NLP_Finetuning_RoBERTa.ipynb" notebook was used.

## Implement and train reward model 

The folder corresponding to the reward model implementation is "reward_model/".

The class to handle the training data for this model can be found in "reward_model/reward_model_dataset.py".

The reward model, its config and the training loop can be found in "reward_model/reward_model.py".

The script for training can be found in "reward_model/reward_model_train_script.py".

To be able to evaluate our model, we made use of the provided script by the course staff. 
We saved the functions in "reward_model/utils.py".
The model can be evaluated using the "reward_model/evaluate.py" script by passing the corresponding arguments.

