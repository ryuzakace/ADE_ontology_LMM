from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import json
import pandas as pd


import os

# Set CUDA_VISIBLE_DEVICES to only use GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


jsonObj = pd.read_json(path_or_buf='data/biodex_test.jsonl', lines=True)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = 'data/best_epoch'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="cuda:0")


def get_instruction_cadec(input_di):

    drug_list = [i for i in input_di['Drug_list']]
    event_list = [i for i in input_di['Event_list']]
    adr_list = [i for i in input_di['ADR_list']]
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

def get_instruction_biodex(input_di):

    # drug_list = [i for i in input_di['Medicine_list']]
    # event_list = [i for i in input_di['ADE_list']]
    # adr_list = [i for i in input_di['ADR_list']]
    # status = True if len(adr_list) else False

    drug_list = [input_di['text'][i[0]:i[1]] for i in input_di['Medicine_list']]
    event_list = [input_di['text'][i[0]:i[1]] for i in input_di['ADE_list']]
    adr_list = [(drug_list[i[0]],event_list[i[1]]) for i in input_di['ADR_list']]
    status = True if len(adr_list) else False


    # print(drug_list, event_list)
    
    content_p = "Your task is to detect Adverse Drug Relation(ADR) between Drug and Events. You will be given a Medical_text, Drug_list and Event_list.\n You have two tasks:\n 1. Status: Status means if ADR is present in text. Values will be True or False.\n 2.ADR_list: This will be a list of Drug-Event pair if they have ADR present.\n Generate a JSON wit Status and ADR_list as two keys"


    
    
    prompt = {
        "role": "user",
        "content": f'{content_p}\n Medical_text:{input_di["text"]}\nDrug_list:{drug_list}\nEvent_list:{event_list}\n. Please generate a json with STATUS and ADR_list keys.\n Assistant:'
    }
    
    assistant = {
        "role": "assistant",
        "content": json.dumps({
            "STATUS": status,
            "ADR_list" : adr_list
        })
    }

    return [prompt, assistant]

def get_instruction(input_di):

    drug_list = [input_di['text'][i[0]:i[1]] for i in input_di['Drug_list']]
    event_list = [input_di['text'][i[0]:i[1]] for i in input_di['Event_list']]
    adr_list = [(drug_list[i[0]],event_list[i[1]]) for i in input_di['ADR_list']]
    status = True if len(adr_list) else False

    # print(drug_list, event_list)

    ADR_ontology = [f'{i[0]} causes {i[1]}' for i in adr_list]
    
    content_p = "Your task is to detect Adverse Drug Relation(ADR) between Drug and Events. You will be given a Medical_text, Drug_list and Event_list.\n You have two tasks:\n 1. Status: Status means if ADR is present in text. Values will be True or False.\n 2.ADR_list: This will be a list of Drug-Event pair if they have ADR present.\n . Generate a JSON wit Status and ADR_list as two keys."


    
    
    prompt_o = {
        "role": "user",
        "content": f'{content_p}\n Medical_text:{input_di["text"]}\nDrug_list:{drug_list}\nEvent_list:{event_list}\n. Also, \n {ADR_ontology}\nPlease generate a json with STATUS and ADR_list keys.\n Assistant:'
    }
    prompt_ = {
        "role": "user",
        "content": f'{content_p}\n Medical_text:{input_di["text"]}\nDrug_list:{drug_list}\nEvent_list:{event_list}\n. Please generate a json with STATUS and ADR_list keys.\n Assistant:'
    }
    
    # assistant = {
    #     "role": "assistant",
    #     "content": json.dumps({
    #         "STATUS": status,
    #         "ADR_list" : adr_list
    #     })
    # }

    return [prompt_o]

def apply_chat_template(example, tokenizer):
    messages = get_instruction_biodex(example)
    # We add an empty system message if there is none
    # if messages[0]["role"] != "system":
    #     messages.insert(0, {"role": "system", "content": ""})
    example["text"] = messages[0]['content']

    return example


import random

random.seed(42)

rand_ind = random.sample(range(1, 1992), 100)

response_li = []
parsed_li = []
for j in tqdm.tqdm(rand_ind):#len(jsonObj['test'][0]))):
    # j = 0
    messages = [
        # {"role": "user", "content": "Your task is to detect Adverse Drug Relation(ADR) between Drug and Events. You will be given a Medical_text, Drug_list and Event_list.\n You have two tasks:\n 1. Status: Status means if ADR is present in text. Values will be True or False.\n 2.ADR_list: This will be a list of Drug-Event pair if they have ADR present.\n Generate a JSON wit Status and ADR_list as two keys\n Medical_text:As these cases revealed, close monitoring of blood chemistry is mandatory after starting spironolactone, and patients should be advised to stop spironolactone immediately if diarrhoea develops.\nDrug_list:['spironolactone']\nEvent_list:['diarrhoea']\n. Please generate a json with STATUS and ADR_list keys"},
        {"role": "user", "content": apply_chat_template(jsonObj[j],tokenizer)['text']},
        # {"role":"user", "content": " All three patients (and possibly a fourth) who developed AML were postmenopausal, received continuous chlorambucil for greater than or equal to 4 years, had acute red cell anemia at the time of treatment, and had a wbc count in the range of 2700-7700/mm3."}
    ]
    
    # messages = apply_chat_template(jsonObj['test'][0][0],tokenizer)['text']
    
    # prepare the messages for the model
    input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    # inference
    outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

    )
    jsonObj[j]['response'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # print(jsonObj[j]['response'] )
    jsonObj[j]['response_parsed'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('Assistant')[1]

with open('results/biodex_zs_mistral.json','w') as f:
    json.dump({'test':jsonObj}, f, indent=4)