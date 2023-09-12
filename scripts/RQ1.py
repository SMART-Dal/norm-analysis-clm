import os
import gc
import sys
sys.path.append('../custom_transformers')
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
import custom_modelling_roberta
from transformers import RobertaTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

MODEL_TAG = "mi"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_TAG)
model = custom_modelling_roberta.RobertaModel.from_pretrained(MODEL_TAG)
model.to(device)

def get_word_fx(f_x, words_to_tokens, mode="mean"):
    word_word_attention = np.array(f_x)
    not_word_starts = []
    for word in words_to_tokens:
        not_word_starts += word[1:]

    # sum up the attentions for all tokens in a word that has been split

    for word in words_to_tokens:
        word_word_attention[word[0]] = word_word_attention[word].mean(axis=-1)
    word_word_attention = np.delete(word_word_attention,
                                    not_word_starts, -1)
    
    return word_word_attention

def get_word_word_attention(token_token_attention, words_to_tokens,
                            mode='first'):
    """Convert token-token attention to word-word attention (when tokens are
  derived from words using something like byte-pair encodings).
  Adapted from: Kevin Clark et al. (https://github.com/clarkkev/attention-analysis)
  """

    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in words_to_tokens:
        not_word_starts += word[1:]

  # sum up the attentions for all tokens in a word that has been split

    for word in words_to_tokens:
        word_word_attention[:, word[0]] = word_word_attention[:,
                word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention,
                                    not_word_starts, -1)

  # several options for combining attention maps for words that have been split
  # we use "mean" in the paper

    for word in words_to_tokens:
        if mode == 'first':
            pass
        elif mode == 'mean':
            word_word_attention[word[0]] = \
                np.mean(word_word_attention[word], axis=0)
        elif mode == 'max':
            word_word_attention[word[0]] = \
                np.max(word_word_attention[word], axis=0)
            word_word_attention[word[0]] /= \
                word_word_attention[word[0]].sum()
        else:
            raise ValueError('Unknown aggregation mode', mode)
    word_word_attention = np.delete(word_word_attention,
                                    not_word_starts, 0)

    return word_word_attention


def read_jsonl_file(file_path):
    try:
        data_list = []
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                data_list.append(data)
        return data_list
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{file_path}'")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    
def save_pkl(name, data):
    with open(name, 'wb') as file:
        pickle.dump(data, file)


def transform_attn(attn_matx):
     # Update the `attentions` key in each row so that it represents layer-wise attention.
     updated_attentions = ()
     for l in range(12):
             attn_layer_l = attn_matx[l] # (bs, num_heads, seq_len, seq_len)
             attn_layer_l = attn_layer_l.mean(1) # (bs, seq_len, seq_len)
             updated_attentions = updated_attentions + (attn_layer_l.cpu(), )
     return updated_attentions

def transform_norms(afx_matx):
     # afx_matx is a 12-tuple (12 as in the number of layers).
     # Each tuple is <||f(x)||, ||αf(x)||, ||Σαf(x)||> . We only care about layerwise, hence we extract
     # ||αf(x)||
     updated_afx = ()
     for l in range(12):
             afx_layer_l = afx_matx[l]
             updated_afx = updated_afx + (afx_layer_l[1].cpu(), )
     return updated_afx

# To generate norms for using java corpus, change the path
# of `file_name` to `5k_csn_java.jsonl` and `lang` to `java`

file_name = "../data/5k_csn_python.jsonl"
data = read_jsonl_file(file_name)
lang = "python"
output = os.path.join("..", "results", lang)

cls_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
cls_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
cls_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]

sep_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
sep_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
sep_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]

kw_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
kw_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
kw_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]

op_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
op_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
op_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]

spec_symb_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
spec_symb_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
spec_symb_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]

literal_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
literal_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
literal_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]

ident_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
ident_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
ident_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]


cls_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
sep_id = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))


with torch.no_grad():
    """
    Generating α, ||f(x)||, ||αf(x)|| maps.
    Adapted from: Kobayashi et al. https://github.com/gorokoba560/norm-analysis-of-transformer
    """
    for data_point in tqdm(data):
        tokens=[tokenizer.cls_token]+data_point['hf_tokens']+[tokenizer.sep_token]
        tokens_ids=torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).to(device)
        _, _, _, attentions, norms = model(input_ids=tokens_ids, output_hidden_states=True, output_attentions=True, output_norms=True)
        
        # Add CLS and SEP tokens
        data_point['cat_ids'] = [-9] + data_point['cat_ids'] + [-10]
        ids = torch.tensor([data_point['cat_ids']])
        seq_len = ids.shape[1]

        words_to_tokens = list(map(lambda x: x[1], data_point['hf_mapping']))

        # Convert to word-word attention and norms
        new_attentions = ()
        for l in range(12):
            heads_tensor = [] # All heads for layer `l`
            for h in range(12):
                token_token_attn = attentions[l][0,h].cpu().numpy()
                word_word_attn = get_word_word_attention(token_token_attn, words_to_tokens, mode="mean")
                heads_tensor.append(word_word_attn)
            heads_tensor = torch.tensor(heads_tensor)
            heads_tensor = torch.unsqueeze(heads_tensor, 0)
            new_attentions = new_attentions + (heads_tensor, )

        new_norms = () # Will contain <f(x), ||αf(x)||>

        for l in range(12):
            new_afx_heads = []
            new_fx_heads = []
            for h in range(12):
                token_token_afx = norms[l][1][0, h].cpu().numpy()
                token_fx = norms[l][0][0, h].cpu().numpy()

                word_word_afx = get_word_word_attention(token_token_afx, words_to_tokens, mode="mean")
                word_fx = get_word_fx(token_fx, words_to_tokens, mode="mean")

                new_afx_heads.append(word_word_afx)
                new_fx_heads.append(word_fx)

            new_afx_heads = torch.tensor(new_afx_heads)
            new_afx_heads = torch.unsqueeze(new_afx_heads, 0)

            new_fx_heads = torch.tensor(new_fx_heads)
            new_fx_heads = torch.unsqueeze(new_fx_heads, 0)

            fx_afx = (new_fx_heads, new_afx_heads)

            new_norms = new_norms + (fx_afx, )

        for l in range(12):
            for h in range(12):
                ## For CLS token (id = -9)
                cls_attn = float(torch.mean(new_attentions[l][0,h,:seq_len,ids[0]==-9].sum(1)).cpu())
                cls_attention_lis[l][h].append(cls_attn)

                cls_fx = float(torch.mean(new_norms[l][0][0,h,ids[0]==-9].mean(-1)).cpu())
                cls_fx_lis[l][h].append(cls_fx)

                cls_afx = float(torch.mean(new_norms[l][1][0,h,0:seq_len,ids[0]==-9].sum(1)).cpu())
                cls_afx_lis[l][h].append(cls_afx)

                ## For SEP token (id = -10)
                sep_attn = float(torch.mean(new_attentions[l][0,h,:seq_len,ids[0]==-10].sum(1)).cpu()) 
                sep_attention_lis[l][h].append(sep_attn)

                sep_fx = float(torch.mean(new_norms[l][0][0,h,ids[0]==-10].mean(-1)).cpu())
                sep_fx_lis[l][h].append(sep_fx)

                sep_afx = float(torch.mean(new_norms[l][1][0,h,0:seq_len,ids[0]==-10].sum(1)).cpu())
                sep_afx_lis[l][h].append(sep_afx)

                ## For Keyword tokens (id = -1)
                kw_attn = float(torch.mean(new_attentions[l][0,h,:seq_len,ids[0]==-1].sum(1)).cpu()) 
                kw_attention_lis[l][h].append(kw_attn)

                kw_fx = float(torch.mean(new_norms[l][0][0,h,ids[0]==-1].mean(-1)).cpu())
                if not np.isnan(kw_fx):
                    kw_fx_lis[l][h].append(kw_fx)

                kw_afx = float(torch.mean(new_norms[l][1][0,h,0:seq_len,ids[0]==-1].sum(1)).cpu())
                kw_afx_lis[l][h].append(kw_afx)

                ## For Operator tokens (id = -2)
                op_attn = float(torch.mean(new_attentions[l][0,h,:seq_len,ids[0]==-2].sum(1)).cpu()) 
                op_attention_lis[l][h].append(op_attn)

                op_fx = float(torch.mean(new_norms[l][0][0,h,ids[0]==-2].mean(-1)).cpu())
                if not np.isnan(op_fx):
                    op_fx_lis[l][h].append(op_fx)

                op_afx = float(torch.mean(new_norms[l][1][0,h,0:seq_len,ids[0]==-2].sum(1)).cpu())
                op_afx_lis[l][h].append(op_afx)

                ## For SpecialSymbol tokens (id = -3)
                spec_symb_attn = float(torch.mean(new_attentions[l][0,h,:seq_len,ids[0]==-3].sum(1)).cpu()) 
                spec_symb_attention_lis[l][h].append(spec_symb_attn)

                spec_symb_fx = float(torch.mean(new_norms[l][0][0,h,ids[0]==-3].mean(-1)).cpu())
                if not np.isnan(spec_symb_fx):
                    spec_symb_fx_lis[l][h].append(spec_symb_fx)

                spec_symb_afx = float(torch.mean(new_norms[l][1][0,h,0:seq_len,ids[0]==-3].sum(1)).cpu())
                spec_symb_afx_lis[l][h].append(spec_symb_afx)

                ## For Literal tokens (id = -4)
                literal_attn = float(torch.mean(new_attentions[l][0,h,:seq_len,ids[0]==-4].sum(1)).cpu()) 
                literal_attention_lis[l][h].append(literal_attn)

                literal_fx = float(torch.mean(new_norms[l][0][0,h,ids[0]==-4].mean(-1)).cpu())
                if not np.isnan(literal_fx):
                    literal_fx_lis[l][h].append(literal_fx)

                literal_afx = float(torch.mean(new_norms[l][1][0,h,0:seq_len,ids[0]==-4].sum(1)).cpu())
                literal_afx_lis[l][h].append(literal_afx)

                ## For Identifier tokens (id = -5)
                ident_attn = float(torch.mean(new_attentions[l][0,h,:seq_len,ids[0]==-5].sum(1)).cpu()) 
                ident_attention_lis[l][h].append(ident_attn)

                ident_fx = float(torch.mean(new_norms[l][0][0,h,ids[0]==-5].mean(-1)).cpu())
                if not np.isnan(ident_fx):
                    ident_fx_lis[l][h].append(ident_fx)

                ident_afx = float(torch.mean(new_norms[l][1][0,h,0:seq_len,ids[0]==-5].sum(1)).cpu())
                ident_afx_lis[l][h].append(ident_afx)

        del attentions, norms, _
        torch.cuda.empty_cache()

torch.save(cls_attention_lis, os.path.join(output, "cls_a_instance_lis"))
torch.save(cls_fx_lis, os.path.join(output, "cls_fx_instance_lis"))
torch.save(cls_afx_lis, os.path.join(output, "cls_afx_instance_lis"))

torch.save(sep_attention_lis, os.path.join(output, "sep_a_instance_lis"))
torch.save(sep_fx_lis, os.path.join(output, "sep_fx_instance_lis"))
torch.save(sep_afx_lis, os.path.join(output, "sep_afx_instance_lis"))

torch.save(kw_attention_lis, os.path.join(output, "kw_a_instance_lis"))
torch.save(kw_fx_lis, os.path.join(output, "kw_fx_instance_lis"))
torch.save(kw_afx_lis, os.path.join(output, "kw_afx_instance_lis"))

torch.save(op_attention_lis, os.path.join(output, "op_a_instance_lis"))
torch.save(op_fx_lis, os.path.join(output, "op_fx_instance_lis"))
torch.save(op_afx_lis, os.path.join(output, "op_afx_instance_lis"))

torch.save(spec_symb_attention_lis, os.path.join(output, "spec_symb_attention_lis"))
torch.save(spec_symb_fx_lis, os.path.join(output, "spec_symb_fx_instance_lis"))
torch.save(spec_symb_afx_lis, os.path.join(output, "spec_symb_afx_lis"))

torch.save(literal_attention_lis, os.path.join(output, "literal_attention_lis"))
torch.save(literal_fx_lis, os.path.join(output, "literal_fx_instance_lis"))
torch.save(literal_afx_lis, os.path.join(output, "literal_afx_lis"))

torch.save(ident_attention_lis, os.path.join(output, "ident_attention_lis"))
torch.save(ident_fx_lis, os.path.join(output, "ident_fx_instance_lis"))
torch.save(ident_afx_lis, os.path.join(output, "ident_afx_lis"))

def convert_head_avg(a_lis, fx_lis, afx_lis):
    a = [[] for _ in range(12)]
    fx = [[] for _ in range(12)]
    afx = [[] for _ in range(12)]
    for l in range(12):
        for h in range(12):
            a[l].append(float(sum(a_lis[l][h])/len(a_lis[l][h])))
            fx[l].append(float(sum(fx_lis[l][h])/len(fx_lis[l][h])))
            afx[l].append(float(sum(afx_lis[l][h])/len(afx_lis[l][h])))
    return a, fx, afx

cls_a_head_lis, cls_fx_head_lis, cls_afx_head_lis = convert_head_avg(cls_attention_lis, cls_fx_lis, cls_afx_lis)

sep_a_head_lis, sep_fx_head_lis, sep_afx_head_lis = convert_head_avg(sep_attention_lis, sep_fx_lis, sep_afx_lis)

kw_a_head_lis, kw_fx_head_lis, kw_afx_head_lis = convert_head_avg(kw_attention_lis, kw_fx_lis, kw_afx_lis)

op_a_head_lis, op_fx_head_lis, op_afx_head_lis = convert_head_avg(op_attention_lis, op_fx_lis, op_afx_lis)

spec_symb_a_head_lis, spec_symb_fx_head_lis, spec_symb_afx_head_lis = convert_head_avg(spec_symb_attention_lis, spec_symb_fx_lis, spec_symb_afx_lis)

literal_a_head_lis, literal_fx_head_lis, literal_afx_head_lis = convert_head_avg(literal_attention_lis, literal_fx_lis, literal_afx_lis)

ident_a_head_lis, ident_fx_head_lis, ident_afx_head_lis = convert_head_avg(ident_attention_lis, ident_fx_lis, ident_afx_lis)


torch.save(cls_a_head_lis, os.path.join(output, "cls_a_head_lis"))
torch.save(cls_fx_head_lis, os.path.join(output, "cls_fx_head_lis"))
torch.save(cls_afx_head_lis, os.path.join(output, "cls_afx_head_lis"))

torch.save(sep_a_head_lis, os.path.join(output, "sep_a_head_lis"))
torch.save(sep_fx_head_lis, os.path.join(output, "sep_fx_head_lis"))
torch.save(sep_afx_head_lis, os.path.join(output, "sep_afx_head_lis"))

torch.save(kw_a_head_lis, os.path.join(output, "kw_a_head_lis"))
torch.save(kw_fx_head_lis, os.path.join(output, "kw_fx_head_lis"))
torch.save(kw_afx_head_lis, os.path.join(output, "kw_afx_head_lis"))

torch.save(op_a_head_lis, os.path.join(output, "op_a_head_lis"))
torch.save(op_fx_head_lis, os.path.join(output, "op_fx_head_lis"))
torch.save(op_afx_head_lis, os.path.join(output, "op_afx_head_lis"))

torch.save(spec_symb_a_head_lis, os.path.join(output, "spec_symb_a_head_lis"))
torch.save(spec_symb_fx_head_lis, os.path.join(output, "spec_symb_fx_head_lis"))
torch.save(spec_symb_afx_head_lis, os.path.join(output, "spec_symb_afx_head_lis"))

torch.save(literal_a_head_lis, os.path.join(output, "literal_a_head_lis"))
torch.save(literal_fx_head_lis, os.path.join(output, "literal_fx_head_lis"))
torch.save(literal_afx_head_lis, os.path.join(output, "literal_afx_head_lis"))

torch.save(ident_a_head_lis, os.path.join(output, "ident_a_head_lis"))
torch.save(ident_fx_head_lis, os.path.join(output, "ident_fx_head_lis"))
torch.save(ident_afx_head_lis, os.path.join(output, "ident_afx_head_lis"))

def convert_layer_avg(a_lis, fx_lis, afx_lis):
    a_head_avg, fx_head_avg, afx_head_avg = convert_head_avg(a_lis, fx_lis, afx_lis)
    a = []
    fx = []
    afx = []
    for l in range(12):
        a.append(float(sum(a_head_avg[l])/len(a_head_avg[l])))
        fx.append(float(sum(fx_head_avg[l])/len(fx_head_avg[l])))
        afx.append(float(sum(afx_head_avg[l])/len(afx_head_avg[l])))
    return a, fx, afx


cls_a_layer_lis, cls_fx_layer_lis, cls_afx_layer_lis = convert_layer_avg(cls_attention_lis, cls_fx_lis, cls_afx_lis)

sep_a_layer_lis, sep_fx_layer_lis, sep_afx_layer_lis = convert_layer_avg(sep_attention_lis, sep_fx_lis, sep_afx_lis)

kw_a_layer_lis, kw_fx_layer_lis, kw_afx_layer_lis = convert_layer_avg(kw_attention_lis, kw_fx_lis, kw_afx_lis)

op_a_layer_lis, op_fx_layer_lis, op_afx_layer_lis = convert_layer_avg(op_attention_lis, op_fx_lis, op_afx_lis)

spec_symb_a_layer_lis, spec_symb_fx_layer_lis, spec_symb_afx_layer_lis = convert_layer_avg(spec_symb_attention_lis, spec_symb_fx_lis, spec_symb_afx_lis)

literal_a_layer_lis, literal_fx_layer_lis, literal_afx_layer_lis = convert_layer_avg(literal_attention_lis, literal_fx_lis, literal_afx_lis)

ident_a_layer_lis, ident_fx_layer_lis, ident_afx_layer_lis = convert_layer_avg(ident_attention_lis, ident_fx_lis, ident_afx_lis)

torch.save(cls_a_layer_lis, os.path.join(output, "cls_a_layer_lis"))
torch.save(cls_fx_layer_lis, os.path.join(output, "cls_fx_layer_lis"))
torch.save(cls_afx_layer_lis, os.path.join(output, "cls_afx_layer_lis"))

torch.save(sep_a_layer_lis, os.path.join(output, "sep_a_layer_lis"))
torch.save(sep_fx_layer_lis, os.path.join(output, "sep_fx_layer_lis"))
torch.save(sep_afx_layer_lis, os.path.join(output, "sep_afx_layer_lis"))

torch.save(kw_a_layer_lis, os.path.join(output, "kw_a_layer_lis"))
torch.save(kw_fx_layer_lis, os.path.join(output, "kw_fx_layer_lis"))
torch.save(kw_afx_layer_lis, os.path.join(output, "kw_afx_layer_lis"))

torch.save(op_a_layer_lis, os.path.join(output, "op_a_layer_lis"))
torch.save(op_fx_layer_lis, os.path.join(output, "op_fx_layer_lis"))
torch.save(op_afx_layer_lis, os.path.join(output, "op_afx_layer_lis"))

torch.save(spec_symb_a_layer_lis, os.path.join(output, "spec_symb_a_layer_lis"))
torch.save(spec_symb_fx_layer_lis, os.path.join(output, "spec_symb_fx_layer_lis"))
torch.save(spec_symb_afx_layer_lis, os.path.join(output, "spec_symb_afx_layer_lis"))

torch.save(literal_a_layer_lis, os.path.join(output, "literal_a_layer_lis"))
torch.save(literal_fx_layer_lis, os.path.join(output, "literal_fx_layer_lis"))
torch.save(literal_afx_layer_lis, os.path.join(output, "literal_afx_layer_lis"))

torch.save(ident_a_layer_lis, os.path.join(output, "ident_a_layer_lis"))
torch.save(ident_fx_layer_lis, os.path.join(output, "ident_fx_layer_lis"))
torch.save(ident_afx_layer_lis, os.path.join(output, "ident_afx_layer_lis"))
