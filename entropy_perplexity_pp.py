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
from collections import Counter
import math
from collections import Counter
from typing import List, Tuple

summeval_fp = '/users/data/summeval.json'
prompt_fp1 = '/users/prompts/summeval/con_detailed.txt'
prompt_fp1 = '/users/prompts/summeval/coh_detailed.txt'
prompt_fp1 = '/users/prompts/summeval/flu_detailed.txt'
prompt_fp1 = '/users/prompts/summeval/rel_detailed.txt'
save_fp = '/users/results/gpt4_con_detailed_openai.json'

input_fp_rel = '/users/results/gpt4_rel_detailed.json'
input_fp_coh = '/users/results/gpt4_coh_detailed.json'
input_fp_flu = '/users/results/gpt4_flu_detailed.json'
input_fp_con = '/users/results/gpt4_con_detailed.json'
save_dir = '/users/results'

def compute_entropy_from_loss(text, model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval() 

    # Encode input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        log_likelihood = outputs.loss 

   
    entropy = log_likelihood.item() / math.log(2)
    perplexity = 2 ** (-log_likelihood.item() / math.log(2))

    # print('ent','prep',entropy,perplexity)
    return entropy, perplexity



def calculate_entropy_from_stat(text):
    # Basic word tokenization (splits on non-word characters)
    words = re.findall(r'\b\w+\b', text.lower())

    # Count word frequencies
    word_counts = Counter(words)
    total_words = len(words)

    # Calculate word probabilities
    probabilities = [count / total_words for count in word_counts.values()]

    # Calculate entropy
    entropy = -sum(p * math.log2(p) for p in probabilities)
    perplexity = 2 ** entropy
    return entropy, perplexity


def measure_all_entropy(input_fp, dimension='relevance'):
    ''' measure linguistic complexity '''
    with open(input_fp, 'r', encoding='utf-8') as f:
        jobj = json.load(f)

    model_scores, human_scores, entropy_source, prep_source, entropy_summary, prep_summary = {}, {}, {}, {}, {}, {}

    for idx,item in enumerate(jobj):
        print(f"item: {idx} ")
        doc_id = item["doc_id"]
        # print(doc_id)
        if doc_id not in model_scores:
            model_scores[doc_id] = []
            human_scores[doc_id] = []
            entropy_source[doc_id] = []
            prep_source[doc_id] = []
            entropy_summary[doc_id] = []
            prep_summary[doc_id] = []
        # Model score
        all_responses = item["all_responses"]
        all_scores = [parse_output(x) for x in all_responses]
        score = sum(all_scores) / len(all_scores)

        model_scores[doc_id].append(score)
        human_scores[doc_id].append(item['scores'][dimension])

        # entropy measurement
        paragraph_summary = item.get("system_output", "").strip() #based on summary
        paragraph_source = item.get("source", "").strip() # based on input

        if paragraph_source:
            print('source')
            # text_entropy,text_prep = compute_entropy_from_loss(paragraph, model_name="gpt2")
            text_entropy,text_prep = calculate_entropy_from_stat(paragraph_source)
            entropy_source[doc_id].append(text_entropy)
            prep_source[doc_id].append(text_prep)
            
        if paragraph_summary:
            print('summary')
            # text_entropy,text_prep = compute_entropy_from_loss(paragraph, model_name="gpt2")
            text_entropy,text_prep = calculate_entropy_from_stat(paragraph_summary)
            entropy_summary[doc_id].append(text_entropy)
            prep_summary[doc_id].append(text_prep)

    return model_scores, human_scores, entropy_source, prep_source, entropy_summary, prep_summary

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
    model_scores_rel,human_scores_rel, ent_source_rel, prep_source_rel, ent_sum_rel, prep_sum_rel = measure_all_entropy(input_fp_rel, dimension = 'relevance')
    model_scores_coh,human_scores_coh, ent_source_coh, prep_source_coh, ent_sum_coh, prep_sum_coh = measure_all_entropy(input_fp_coh, dimension = 'coherence')

    
    file_path = os.path.join(save_dir, 'ent_prep_stats_rel.pkl')

    dict = [model_scores_rel, human_scores_rel, ent_source_rel, prep_source_rel, ent_sum_rel, prep_sum_rel]
    
    with open(file_path, 'wb') as f:
        pickle.dump(dict, f)
    
    file_path = os.path.join(save_dir, 'ent_prep_stats_coh.pkl')
    dict = [model_scores_coh, human_scores_coh, ent_source_coh, prep_source_coh, ent_sum_coh, prep_sum_coh]
    
    with open(file_path, 'wb') as f:
        pickle.dump(dict, f)
    
  
        

if __name__ == "__main__":
    main()

