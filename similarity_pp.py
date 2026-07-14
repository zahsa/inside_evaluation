# import openai
import os

import json
import argparse
import tqdm
import time
import textstat

import re
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import prettytable
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression

summeval_fp = '/users/data/summeval.json'

prompt_fp1 = '/users/prompts/summeval/coh_detailed.txt'
prompt_fp1 = '/users/prompts/summeval/rel_detailed.txt'
save_fp = '/users/results/gpt4_con_detailed_openai.json'

input_fp_rel = '/users/results/gpt4_rel_detailed.json'
input_fp_coh = '/users/results/gpt4_coh_detailed.json'

save_dir = '/users/results'

def get_embedding(text, model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # Enable outputting hidden states
    model.config.output_hidden_states = True
    model.eval()
    
    # model = AutoModel.from_pretrained("gpt2")
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
         outputs = model(**inputs, output_hidden_states=True)
       

    # Get last hidden states (same as GPT2Model output)
    last_hidden = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_dim]
    
    # Average over all tokens (excluding padding if needed)
    sentence_embedding = last_hidden.mean(dim=1).squeeze(0)  # Shape: [hidden_dim]

    #print("Embedding shape:", sentence_embedding.shape)
    return sentence_embedding

def get_inp_out_relation(input_fp, dimension='relevance'):
    ''' measure linguistic complexity '''
    with open(input_fp, 'r', encoding='utf-8') as f:
        jobj = json.load(f)

    model_scores, human_scores, cos_sim_list, cka_score_list = {}, {}, {}, {}

    for idx,item in enumerate(jobj):
        print(f"item: {idx} ")
        doc_id = item["doc_id"]
        # print(doc_id)
        if doc_id not in model_scores:
            model_scores[doc_id] = []
            human_scores[doc_id] = []
            cos_sim_list[doc_id] = []
            #mi_vector[doc_id] = []
            #cka_score_list[doc_id] = []
           
        # Model score
        all_responses = item["all_responses"]
        all_scores = [parse_output(x) for x in all_responses]
        score = sum(all_scores) / len(all_scores)

        model_scores[doc_id].append(score)
        human_scores[doc_id].append(item['scores'][dimension])

        paragraph_output = item.get("system_output", "").strip() #based on summary
        paragraph_input = item.get("source", "").strip() # based on input

        if paragraph_input:
            embedding_input = get_embedding(paragraph_input, model_name="gpt2")
            embedding_output = get_embedding(paragraph_output, model_name="gpt2")

           
            #compute mutual info and cosine sim here
            cos_sim = F.cosine_similarity(embedding_input.unsqueeze(0), embedding_output.unsqueeze(0)).item()
            # print("Cosine similarity:", cos_sim)
            cos_sim_list[doc_id].append(cos_sim)

            
    return model_scores, human_scores, cos_sim_list

def parse_output(output):
    matched = re.search("^ ?([\d\.]+)", output)
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = 0
    else:
        score = 0
    return score


def main():
   
    model_scores_rel,human_scores_rel, cos_sim_rel = get_inp_out_relation(input_fp_rel, dimension = 'relevance')
    model_scores_coh,human_scores_coh, cos_sim_coh, avgmi_coh = get_inp_out_relation(input_fp_coh, dimension = 'coherence')
   
    
    file_path = os.path.join(save_dir, 'inp_out_sim_scores_rel.pkl')
    rel_dicts = [model_scores_rel, human_scores_rel, cos_sim_rel]
    
    with open(file_path, 'wb') as f:
        pickle.dump(rel_dicts, f)
    
       
    file_path = os.path.join(save_dir, 'inp_out_sim_scores_con.pkl')
    rel_dicts = [model_scores_con, human_scores_con, cos_sim_con]
    
    with open(file_path, 'wb') as f:
        pickle.dump(rel_dicts, f)
        

if __name__ == "__main__":
    main()

