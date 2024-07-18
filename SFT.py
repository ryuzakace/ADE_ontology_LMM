from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
import pandas as pd    
import re
import random
from multiprocessing import cpu_count
from datasets import load_dataset
from datasets import DatasetDict
import json
import copy
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments



def get_instruction(input_di):

    drug_list = [input_di['input_text'][i[0]:i[1]] for i in input_di['Medicine_list']]
    event_list = [input_di['input_text'][i[0]:i[1]] for i in input_di['ADE_list']]
    adr_list = [(drug_list[i[0]],event_list[i[1]]) for i in input_di['ADR_list']]
    status = True if len(adr_list) else False

    # print(drug_list, event_list)
    
    content_p = "Your task is to detect Adverse Drug Relation(ADR) between Drug and Events. You will be given a Medical_text, Drug_list and Event_list.\n You have two tasks:\n 1. Status: Status means if ADR is present in text. Values will be True or False.\n 2.ADR_list: This will be a list of Drug-Event pair if they have ADR present.\n Generate a JSON wit Status and ADR_list as two keys"


    
    
    prompt = {
        "role": "user",
        "content": f'{content_p}\n Medical_text:{input_di["input_text"]}\nDrug_list:{drug_list}\nEvent_list:{event_list}\n. Please generate a json with STATUS and ADR_list keys'
    }
    
    assistant = {
        "role": "assistant",
        "content": json.dumps({
            "STATUS": status,
            "ADR_list" : adr_list
        })
    }

    return [prompt, assistant]

def apply_chat_template(example, tokenizer):
    messages = get_instruction(example)
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

def create_dataset():
    jsonObj = pd.read_json(path_or_buf='adr_dataset_train_test.jsonl', lines=True)
    for i,j in enumerate(jsonObj['test'][0]):
        jsonObj['test'][0][i]['input_text'] = j['text']
    
    for i,j in enumerate(jsonObj['train'][0]):
        jsonObj['train'][0][i]['input_text'] = j['text']
    return jsonObj

def main():
    dataset = create_dataset()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
      tokenizer.model_max_length = 2048
    
    # Set chat template
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    train_set = [apply_chat_template(i, tokenizer) for i in jsonObj['train'][0]]
    val_set = [apply_chat_template(i, tokenizer) for i in jsonObj['test'][0]]
    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", #ch
                bnb_4bit_compute_dtype=torch.float16, #ch
    )
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    model_kwargs = dict(
        attn_implementation="eager", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map="auto",
        # quantization_config=quantization_config,
    )

    output_dir = 'data/best_epoch'
    
    
    # based on config
    training_args = TrainingArguments(
        bf16=True, # specify bf16=True instead when training on GPUs that support bf16
        do_eval=True,
        evaluation_strategy="epoch", #ch
        gradient_accumulation_steps=64,
        gradient_checkpointing=True, #ch
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2.0e-05,
        log_level="info",
        logging_steps=5,
        logging_strategy="steps",
        lr_scheduler_type="cosine", #ch
        max_steps=-1,
        num_train_epochs=200,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=2, # originally set to 8
        per_device_train_batch_size=2, # originally set to 8
        # push_to_hub=True,
        # hub_model_id="zephyr-7b-sft-lora",
        # hub_strategy="every_save",
        # report_to="tensorboard",
        save_strategy="epoch", #ch
        # save_total_limit=None,
        seed=42,
        load_best_model_at_end = True
    )
    # based on config
    peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    trainer = SFTTrainer(
            model=model_id,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            dataset_text_field="text",
            tokenizer=tokenizer,
            packing=True,
            peft_config=peft_config,
            max_seq_length=tokenizer.model_max_length,
        )
    train_result = trainer.train()

        

if __name__ == '__main__':
    main()