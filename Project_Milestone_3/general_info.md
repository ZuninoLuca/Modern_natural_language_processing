## Final Answers

Since we explored multiple approaches, we have multiple files containign the answers generated by our generative models for the testing questions in "prompts.json".
These files are separated based on the model used in the following folders:
1) generated-answers-flan-t5
2) generated-answers-gpt2
3) generated-answers-rlhf-agent
4) generated-answers-t5

Since we found the T5 model to have the best performance, our final and main answer file is the one produced by the T5 model and can be found at "generated-answers-t5/answers_modern_nlm_10e.json".


## Live Chat
To try our live chat with the keyword extractor and context retriever, use the notebook in the following path. This is the recommended way of testing the live chat, since both models are fetched from HuggingFace Hub. 

```
live_chat/live_prompting_notebook.ipynb
```

Alternatively, the script can be used by running the following: 

```
python live_chat/live_prompting_script.py
```




## Checkpoints

Our checkpoints for T5 and Flan-T5 can be accessed through this [link](https://drive.google.com/drive/u/2/folders/1R8IKxF66ucL13kn1FN8BCpyU6n2wg_Sh).

Due to repository size limit, we also did not add the reward model. However, this model can be retrieved from our Milestone 2 repository.

## Training and Testing Scripts
We provide the final T5 model and tokenizer in the folder "final_model"

For T5 and Flan-T5 finetuning, run the following:

```
python t5-model/t5_train.py
```
make sure to change the path for Flan-t5 in case the checkpoint was downloaded from the [link](https://drive.google.com/drive/u/2/folders/1R8IKxF66ucL13kn1FN8BCpyU6n2wg_Sh).

For finetuning GPT2:

```
python gpt2-model/gpt2_context_train_script.py
```

To access our generative models and produce answers for the testing questions, run the following:

For T5 and Flan-T5:

```
python t5-model/gen_script_modern_nlm_t5_inference.py
```

For GPT2:

```
python gpt2-model/gen_script_modern_nlm_inference_gpt2.py 
```

## References Declaration

We based our code for training T5 on this [repo](https://github.com/nunziati/bert-vs-t5-for-question-answering) and based our code for RLHF on the this [repo](https://github.com/voidful/TextRL)