import argparse
import json
import os
import random


ROOT_PATH = os.path.dirname(os.path.abspath(__file__)).split("ModCirc")[0] + "ModCirc/ModCirc"

DATA_FILE_MAP = {
    "medal": "medal/medal_1500_mc.jsonl",
    "symptom2disease": "symptom_to_diagnosis/s2d_1065_mcq.jsonl",
    "medmcqa": "medmcqa/medmcqa_1500.jsonl",
    "medicalabstract": "medical-abstract/medical_abstract_1500.jsonl",
    "medi_attr": "eval_datasets/medication_attr.jsonl",
    "medi_status": "eval_datasets/medication_status.jsonl",
    "coreference": "eval_datasets/coreference.jsonl",
    "pubmed_summ": "eval_datasets/pubmed_summarization.jsonl",
    "hallucination_data" : "halucination_data/numerical_dataset.jsonl",
    "Non_hallucination" : "Non_hallucination/numerical_dataset.jsonl"
}

# DATA_FILE_MAP = {
    
#     "numerical" : "numerical/numerical_dataset.jsonl",
#     "numerical_eval" : "eval_dataset/numerical_eval.jsonl"
# }

def get_dataset(dataset_name, tokenizer) -> list[dict[str, str]]:
    vocab = list(tokenizer.vocab.keys())
    print("root_path: ",ROOT_PATH)
    
    
    data_path = f"{ROOT_PATH}/text_datasets/{DATA_FILE_MAP[dataset_name]}"
    
    #HALoGEN dataset path
    # data_path = f"{ROOT_PATH}/HALoGEN_datasets/{DATA_FILE_MAP[dataset_name]}"
    data = []
    if ".jsonl" in data_path:
        with open(data_path, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
    else:
        data = json.load(open(data_path, "r"))
    processed_data = []
    for d in data:
        prompt = [
            d["question"],
            "\na. ",
            d["opa"].lower(),
            "\nb. ",
            d["opb"].lower(),
            "\nc. ",
            d["opc"].lower(),
        ]
        if "opd" in d:
            prompt += ["\nd. ", d["opd"].lower()]
        processed_data.append(
            {
                "clean_text": "".join(prompt),
                "answer": chr(
                    d["cop"] + 97
                ),  # we expect the LLM to return only a, b, c, d
            }
        )
        # replace_idx = random.choice(
        #     [idx for idx in range(4 - ("opd" not in d)) if idx != d["cop"]]
        # )
        # prompt[replace_idx * 2 + 1], prompt[d["cop"] * 2 + 1] = (
        #     prompt[d["cop"] * 2 + 1],
        #     prompt[replace_idx * 2 + 1],
        # )
        idx = random.sample(list(range(len(vocab))), 1)[0]
        prompt[d["cop"] * 2 + 2] = vocab[idx]
        processed_data[-1]["corrupted_text"] = "".join(prompt)
    random.shuffle(processed_data)
    print( "size of the processed_Data : " , len(processed_data)) 
    return processed_data


def get_save_paths(
    args: argparse.Namespace, exp_num: int
) -> tuple[str, str, str, str, str]:
    # root = f""
    root = os.path.join(args.save_path, f"exp_{exp_num}")
    # nodes_path = f""
    nodes_path = os.path.join(root, "nodes.pt")
    # eval_nodes_path = f""
    eval_nodes_path = os.path.join(root, "eval_nodes.pt")
    
    circuit_dataset = (
        f"{root}_circuit_dataset_topk={args.topk_nodes}_num-graphs={args.num_graphs}.pt"
    )

    if "modcirc" in args.exp_type or "ablation" in args.exp_type:
        circuit_dataset = f"{root}_circuit_dataset_topk={args.topk_nodes}.pt"
        model_path = f"{root}_gnn_x={args.x_dim}_hdim={args.hidden_dim}_edim={args.emb_dim}_mp={args.max_partitions}_a={args.alpha}_b={args.beta}_g={args.gamma}_inf={args.infty_val}_ep={args.epochs}_nl={args.num_layers}_lr={args.lr}_tk={args.topk_nodes}.pt"
        result_path = model_path.replace("_gnn_", "_results_").replace(".pt", ".txt")
    else:
        circuit_dataset = f"{root}_circuit_dataset_topk={args.topk_nodes}_num-graphs={args.num_graphs}.pt"
        model_path = f"{root}_gnn_num-graphs={args.num_graphs}.pt"
        result_path = model_path.replace("_gnn_", "_results_").replace(".pt", ".txt")

    importance_scores_path = model_path.replace("_gnn_", "_imp_scores_").replace(
        ".pt", ".json"
    )
    func_interp_path = importance_scores_path.replace("_imp_scores_", "_func_interp_")
    partitions_path = model_path.replace("_gnn_", "_partitions_")
    os.makedirs(os.path.dirname(root), exist_ok=True)
    print("Nodes Path :", nodes_path)
    print("Eval Nodes path :", eval_nodes_path)
    print("circuit dataset :", circuit_dataset)
    print("model path :", model_path)
    print("result path :", result_path)
    return (
        nodes_path,
        eval_nodes_path,
        circuit_dataset,
        model_path,
        result_path,
        importance_scores_path,
        func_interp_path,
        partitions_path,
    )


