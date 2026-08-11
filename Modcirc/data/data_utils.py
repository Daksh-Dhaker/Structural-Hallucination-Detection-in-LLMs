import json
from utils import ROOT_PATH

DATA_FILE_MAP = {"medal": "medal/medal_1500_mc.jsonl", 
                 "symptom2disease": "symptom_to_diagnosis/s2d_1065_mcq.jsonl",
                 "medmcqa": "medmcqa/medmcqa_1500.jsonl",
                 "medicalabstract": "medical-abstract/medical_abstract_1500.jsonl"}

def get_dataset(dataset_name:str, corrupt_token:str = "None"):
    data_path = f"{ROOT_PATH}/text_datasets/{DATA_FILE_MAP[dataset_name]}"
    data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            item = json.loads(line)
            clean_text = f"{item['question']}\na. {item['opa']}\nb. {item['opb']}\nc. {item['opc']}"
            corrupted_text = " ".join([corrupt_token] * len(item["question"].split(" "))) + "\na. " + clean_text.split("\na. ")[-1]
            data.append({
                "clean_text": clean_text,
                "corrupted_text": corrupted_text,
                "answer": chr(item['cop'] + 97) # we expect the LLM to return only a, b, or c

            })
    return data
       