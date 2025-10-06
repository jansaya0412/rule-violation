import pandas as pd

from tqdm import tqdm
import json
import os
from collections import defaultdict
from ast import literal_eval
import re

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from ruler.data.constants import URL_REPLACEMENT_TOKEN
from ruler.data.text_utils import markdown_to_text, replace_urls


def get_modlog_community_descriptions(data_dir='../../data'):
    
    modlog_folder = os.path.join(data_dir, 'raw', 'modlogs')
    community_descriptions = defaultdict(set)
    for instance in tqdm(os.listdir(modlog_folder)): 
        modlog_fpath = f"{modlog_folder}/{instance}/removed_comments.jsonl"
        if not os.path.exists(modlog_fpath): continue
        with open(modlog_fpath, encoding='utf8') as f:
            for entry in map(json.loads, f):
                if 'description' not in entry['community']: continue
                try:
                    if len(entry['community']['description']) > 10:
                        community_descriptions[entry['community']['actor_id']].add(entry['community']['description'])
                except:
                    pass
    return community_descriptions


def augment_with_languages(descs_df):
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt, device_map="auto")
    descs_df['language'] = list(map(lambda x: x[0]['label'], pipe(descs_df.description.tolist(), top_k=1, truncation=True)))
    return descs_df


# this method overruns GPU memory, even with batchsize=1
# def predict_NuExtract(model, tokenizer, texts, template, batch_size=1, max_length=10_000, max_new_tokens=4_000):
#     template = json.dumps(json.loads(template), indent=4)
#     prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]

#     outputs = []
#     with torch.no_grad():
#         for i in range(0, len(prompts), batch_size):
#             batch_prompts = prompts[i:i + batch_size]
#             batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True,
#                                         max_length=max_length).to(model.device)

#             pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
#             outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

#     return [output.split("<|output|>")[1] for output in outputs]

def predict_NuExtract(text,community, communities_meta, model, tokenizer,  schema, example=["", "", ""]):
    if len(text) < 50:
        return ""

    #check if the description already exists
    if community in communities_meta:
        for descript in communities_meta[community]:
            if text == descript['description']:
                return descript['rules']

    text = text.replace('"','').replace("'","")

    schema = json.dumps(json.loads(schema))
    input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
    for i in example:
      if i != "":
          input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"
    
    input_llm +=  "### Text:\n"+text +"\n<|output|>\n"    
    try:
        input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to("cuda")
        with torch.no_grad():
            output = tokenizer.decode(model.generate(**input_ids, max_new_tokens=4000)[0], skip_special_tokens=True)
        return output.split("<|output|>")[1].split("<|end-output|>")[0]
    except Exception as e:
        print(f"Error during model inference: {e}")
        return ""

def augment_with_rules(descs_df, communities_meta, cache_dir='../../models/huggingface', max_chars=800, batch_size=100):

    #model_name = 'numind/NuExtract-tiny'
    model_name = "numind/NuExtract-v1.5"
    #model_name = 'numind/NuExtract'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(" device: ", device)

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(
        device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    
    schema = """{
    "Rules": [{
    "Number": 1,
    "Description": ""
    }]
    }"""
    #predictions = predict_NuExtract(model, tokenizer, [desc[:max_chars] for desc in descs_df.description.fillna('').tolist()],
    #                                 schema, batch_size=batch_size)

    descs_df['rules'] = descs_df.apply(lambda x: predict_NuExtract(x['description'], x['actor_id'],communities_meta, model=model, tokenizer=tokenizer,  schema=schema, example=["", "", ""]), axis=1)
    #descs_df['rules'] = predictions
    return descs_df

def convert_json_format(text):
    try:
        rule=json.loads(text)
        simplified_rules={}
        simplified_rules = {    "rules": {int(rul["Number"]): rul["Description"] for rul in rule["Rules"]}}
        return simplified_rules
    except Exception as e:
        return e

def check_pointers(text):
    if "rule" in text.lower():
        return 1
    

    pattern = r"^\s*(\d+\.\s*|\*\s+|-\s+|•\s+|[ivxlcdm]+\.\s+|[IVXLCDM]+\.\s+)"
    if re.match(pattern, text):
        return 1


def main(data_dir='../../data',max_chars=800, min_rules=1, max_rules=10, recompute_logs=False, recompute_language=False):
    description_fpath = os.path.join(data_dir, 'interim', 'v2_community_descriptions.json')
    if os.path.exists(description_fpath) and not recompute_logs:
        with open(description_fpath, encoding='utf8') as f:
            descs = json.load(f)
    else:
        print('getting descriptions')
        descs = get_modlog_community_descriptions(data_dir)
        with open(description_fpath, 'w+', encoding='utf8') as f:
            json.dump({k:list(v) for k, v in descs.items()}, f, sort_keys=True, ensure_ascii=False)

    #descs = {k: v.pop() for k, v in descs.items() if len(v) == 1}
    #descs = {k: v for k, v in descs.items() if v and len(v.strip())>0}
    descslist = []
    #descs_df = pd.DataFrame(columns=['actor_id', 'description'])
    for k, v in descs.items():
        for v_ in list(v):
            descslist.append({'actor_id':k, 'description':v_})

    descs_df = pd.DataFrame(descslist)

    #descs = {k: v_ for v_ in v for v in k, v in descs.items()}
    #descs_df = pd.DataFrame(descs.items(), columns=['actor_id', 'description'])
    descs_df = descs_df.drop_duplicates(subset=['actor_id', 'description'])

    print(' number of communities: ', len(set(descs_df['actor_id'].tolist())))
    print(' number of descriptions: ', len(set(descs_df['description'].tolist())))

    print('inferring language')
    language_fpath = os.path.join(data_dir, 'interim', 'v2_communities_with_languages.json')
    if os.path.exists(language_fpath) and not recompute_language:
        with open(language_fpath, encoding='utf-8') as f_l:
            descs_l = json.load(f_l)

        #descs_df = pd.DataFrame(descs_l.items(), columns=['actor_id', 'description', 'language'])
        descs_df = pd.DataFrame(descs_l)

    else:

        descs_df.dropna(subset=['description'], inplace=True)
        # replace urls
        descs_df['description'] = descs_df.description.apply(lambda x: replace_urls(markdown_to_text(x), URL_REPLACEMENT_TOKEN))
        # discard entries with descriptions that contain only urls
        descs_df = descs_df[descs_df.description.apply(lambda x: len(''.join(x.split(URL_REPLACEMENT_TOKEN)).strip()) > 0)]
        descs_df = augment_with_languages(descs_df)
        descs_df.to_json(os.path.join(data_dir, 'interim', 'v2_communities_with_languages.json'))

    community2languages = descs_df.groupby('actor_id')['language'].apply(set).apply(list).reset_index()
    community2languages['n_lang'] = community2languages['language'].apply(lambda x: len(x))
    community2languages['lang'] = community2languages['language'].apply(lambda x: x[0])

    english_communities = community2languages[(community2languages['n_lang']==1)&(community2languages['lang']=='en')]['actor_id'].tolist()

    descs_df = descs_df[descs_df['actor_id'].isin(english_communities)]

    print(" descriptions in english: ", len(descs_df))

    descs_df['extractable_rules'] = descs_df['description'].apply(lambda x: check_pointers(x))

    print(" descriptions with extractables rules: ", len(descs_df[descs_df['extractable_rules']==1]))

    #sampling for test
    #descs_df = descs_df.sample(200)

    # check presence of rules or pointer chatacters

    print('getting already extracted descriptions')
    with open(os.path.join(data_dir, 'interim', 'community_meta.json'), 'r', encoding='utf8') as extracted:
        communities_meta = json.load(extracted)

    print('inferring rules')
    descs_df= augment_with_rules(descs_df, communities_meta, max_chars=max_chars, cache_dir='../../models/huggingface')
    descs_df.to_json(os.path.join(data_dir, 'interim', 'v2_communities_with_all_rules.json'))
    descs_df['rules'] = descs_df['rules'].apply(lambda x: convert_json_format(x))

    #some tules are malformed
    def get_rule_len(x):
        try:
            return len(x['rules'].keys())
        except:
            return 0
    
    descs_df.to_json(os.path.join(data_dir, 'interim', 'v2_communities_with_all_rules.json'))
    descs_df['n_rules'] = descs_df['rules'].apply(lambda x: get_rule_len(x))
    with open(os.path.join(data_dir, 'interim', 'v2_communities_with_all_rules.jsonl'), 'w+', encoding='utf8') as f:
        for _, row in descs_df.iterrows():
            try:
                f.write(json.dumps(
                    dict(actor_id=row.actor_id, description=row.description, language=row.language, rules=row.rules, nrules=row.n_rules),
                    sort_keys=True, ensure_ascii=False) + '\n')
            except:
                print("malformed row")
                print(row)
               
    
    descs_df = descs_df[descs_df['n_rules']<=max_rules]
    #descs_df = descs_df[descs_df.rules.apply(lambda x: min_rules <= len(x.rules.get("Rules", [])) <= max_rules)]
    descs_df.to_json(os.path.join(data_dir, 'interim', 'v2_filtered_communities.json'))
    with open(os.path.join(data_dir, 'interim', 'v2_filtered_communities.jsonl'), 'w+', encoding='utf8') as f:
        for _, row in descs_df.iterrows():
            try:
                f.write(json.dumps(
                    dict(actor_id=row.actor_id, description=row.description, language=row.language, rules=row.rules, nrules=row.n_rules),
                    sort_keys=True, ensure_ascii=False) + '\n')
            except:
                print("malformed row")
                print(row)

if __name__ == '__main__':
    main()
