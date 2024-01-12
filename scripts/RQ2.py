"""Compute aggregate statistics of attention edge features over a dataset
Adapted from: 
    Vig et al. (https://github.com/salesforce/provis)
    Wan et al. (https://github.com/CGCL-codes/naturalcc)
"""
import re, os
from collections import defaultdict
import json
import numpy as np
import torch
from tqdm import tqdm
import sys
from custom_transformers import RobertaModel

sm = torch.nn.functional.softmax
normalize = torch.nn.functional.normalize

def group_indices(tokens, raw_tokens):
    mask = []
    raw_i = 0
    collapsed = ''
    special ='Ġ'

    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]
        collapsed += token
        mask.append(raw_i)
        if collapsed == raw_tokens[raw_i]:
            raw_i += 1
            collapsed = ''
    if raw_i != len(raw_tokens):
        raise Exception(f'Token mismatch: \n{tokens}\n{raw_tokens}')
    return torch.tensor(mask)
def compute_mean_attention(model,
                           n_layers,
                           n_heads,
                           items,
                           tokenizer,
                           model_version,
                           cuda=True,
                           min_attn=0,
                           mode="attn"):
    model.eval()

    with torch.no_grad():

        # Dictionary that maps feature_name to array of shape (n_layers, n_heads), containing
        # weighted sum of feature values for each layer/head over all examples
        feature_to_weighted_sum = defaultdict(lambda: torch.zeros((n_layers, n_heads), dtype=torch.double))

        # Sum of attention_analysis weights in each layer/head over all examples
        weight_total = torch.zeros((n_layers, n_heads), dtype=torch.double)

        for item in tqdm(items):
            # Get attention weights, shape is (num_layers, num_heads, seq_len, seq_len)
            attns = get_attention(model,
                                  item,
                                  tokenizer,
                                  model_version,
                                  cuda,
                                  mode,)
            if attns is None:
                print('Skipping due to not returning attention')
                continue
            # Update total attention_analysis weights per head. Sum over from_index (dim 2), to_index (dim 3)
            mask = attns >= min_attn
            weight_total += mask.long().sum((2, 3))
            # weight_total+=attns.sum((2,3))

            # Update weighted sum of feature values per head
            seq_len = attns.size(2)
            feature_map=item['feature_map']
            for to_index in range(seq_len):
                for from_index in range(seq_len):
                    value=feature_map[from_index][to_index]
                    mask=attns[:,:,from_index,to_index]>=min_attn
                    # attns_item=attns[:,:,from_index,to_index]
                    feature_to_weighted_sum['contact_map']+=mask*value
        return feature_to_weighted_sum, weight_total


def get_attention(model,
                  item,
                  tokenizer,
                  model_version,
                  cuda,
                  mode,):

    raw_tokens = item['code_tokens']
    code_tokens=tokenizer.tokenize(' '.join(raw_tokens))
    tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    token_idxs=tokenizer.convert_tokens_to_ids(tokens)

    inputs = torch.tensor(token_idxs).unsqueeze(0)

    if cuda:
        inputs=inputs.cuda()
    last_hidden_state, pooler_output, hidden_states, attention, norms = model(input_ids=inputs, output_hidden_states=True, output_attentions=True, output_norms=True)

    if mode == "norm":
        new_norms = ()
        for l in range(12):
            heads_tensor = []
            for h in range(12):
                old_head = norms[l][1][0,h]
                new_head = normalize(old_head, dim=(0, 1)).cpu().numpy()
                heads_tensor.append(new_head)
            heads_tensor = torch.tensor(heads_tensor)
            heads_tensor = torch.unsqueeze(heads_tensor, 0)
            new_norms = new_norms + (heads_tensor, )
        
        attention = new_norms


    else:
        attention = list(attention)
    n_layers = 12
    all_att = torch.cat([attention[n][:, :, 1:-1, 1:-1] for n in range(n_layers)], dim=0)

    mask = group_indices(code_tokens, raw_tokens)
    raw_seq_len = len(raw_tokens)
    all_att = torch.stack(
        [all_att[:, :, :, mask == i].sum(dim=3)
         for i in range(raw_seq_len)], dim=3)
    all_att = torch.stack(
        [all_att[:, :, mask == i].mean(dim=2)
         for i in range(raw_seq_len)], dim=2)

    return all_att.cpu()

def getData(path):
    code_list = []  
    with open(path, 'r') as f:
        code_dicts = f.readlines()
    for code_dict in code_dicts:
        code_item = json.loads(code_dict)
        code_list.append(code_item)
    return code_list

if __name__ == "__main__":
    import pickle
    import pathlib

    from transformers import RobertaTokenizer

    model_version='microsoft/codebert-base'
    model=RobertaModel.from_pretrained(model_version)
    tokenizer=RobertaTokenizer.from_pretrained(model_version,do_lower_case=False)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    print('Layers:', num_layers)
    print('Heads:', num_heads)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.split(current_dir)[0]

    model.to('cuda')
    python_file_dir = os.path.join(parent_directory, 'data', '5k_csn_python.jsonl')
    dataset=getData(python_file_dir)
    shuffle=True
    num_sequences=5000
    if shuffle:
        random_indices = torch.randperm(len(dataset))[:num_sequences].tolist()
        items = []
        print('Loading dataset')
        for i in tqdm(random_indices):
            item = dataset[i]
            items.append(item)
    else:
        raise NotImplementedError
    min_attn=0.3

    feature_to_weighted_sum, weight_total = compute_mean_attention(
        model,
        num_layers,
        num_heads,
        items,
        tokenizer,
        model_version,
        cuda=True,
        min_attn=min_attn,
        mode='attn')
    
    feature_to_weighted_sum_afx, weight_total_afx = compute_mean_attention(
        model,
        num_layers,
        num_heads,
        items,
        tokenizer,
        model_version,
        cuda=True,
        min_attn=min_attn,
        mode='norm')

    print(feature_to_weighted_sum)
    print(weight_total)

    print(feature_to_weighted_sum_afx)
    print(weight_total_afx)

    exp_name_alpha='edge_features_contact_mean_codebert_python_noneighbor'
    exp_name_afx='edge_features_contact_afx_mean_codebert_python_noneighbor'
    
    path = os.path.join(parent_directory, "results", f'{exp_name_alpha}.pickle')
    pickle.dump((dict(feature_to_weighted_sum), weight_total), open(path, 'wb'))
    print('Wrote to', path)


    path_afx = os.path.join(parent_directory, "results", f'{exp_name_afx}.pickle')
    pickle.dump((dict(feature_to_weighted_sum_afx), weight_total_afx), open(path_afx, 'wb'))
    print('Wrote to', path_afx)