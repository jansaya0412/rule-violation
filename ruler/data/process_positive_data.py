import json
import os
from collections import defaultdict
from tqdm import tqdm
import pickle as pkl
from ruler.data.constants import URL_REPLACEMENT_TOKEN
from ruler.data.sampling_utils import AlgorithmL
from ruler.data.text_utils import replace_urls, markdown_to_text


def encode_positive_data(entry, community_rules):
    return {
        'ap_id': entry['comment']['ap_id'],
        'content': entry['comment']['content'],
        'removed': entry['comment']['removed'],
        'applied_rule_n': 0,
        'applied_rule_text': "Safe",
        'reason': None,
        'community': {
            'actor_id': entry['community']['actor_id'],
            'rules': community_rules[entry['community']['actor_id']],
            'description': entry['community']['description'],
            'name': entry['community']['name'],
            'nsfw': entry['community']['nsfw'],
        },
        'instance': entry['instance'],
        'modlog_id': -1,
        'mod_person_id': -1,
    }


def sample_positive_data(selected_modlog_comments, all_modlog_comments, data_dir='../../data', min_char=30, max_char=400):
    modlog_ids_by_community = defaultdict(set)
    modlog_rules = dict()
    modlog_descriptions = dict()
    for entry in selected_modlog_comments:
        modlog_ids_by_community[(entry['community']['actor_id'])].add(entry['ap_id'])
        modlog_rules[entry['community']['actor_id']] = entry['community']['rules']
        modlog_descriptions[entry['community']['actor_id']] = entry['community']['description']
    #  sample randomly as many comments as in modlog
    samplers = {k: AlgorithmL(len(v)) for k, v in modlog_ids_by_community.items()}

    # for actor_id, sampler in samplers.items(): # I could extract instance and spare a lot of comparisons
    
    comment_root = os.path.join(data_dir, 'raw', 'comment')
    

    processed_comments = set()
    for instance in tqdm(os.listdir(comment_root)):
        for community_file in os.listdir(os.path.join(comment_root, instance)):
            cfilepath = os.path.join(comment_root, instance, community_file)
            if str(cfilepath).endswith(".jsonl"):
                with open(os.path.join(comment_root, instance, community_file), encoding='utf8') as f:
                    for entry in map(json.loads, f):
                        community = entry['community']['actor_id']
                        if community not in samplers: break  # assume all comments are from the community
                        comment_id = entry['comment']['ap_id']
                        if comment_id in all_modlog_comments: continue
                        sampler = samplers[community]

                        # comments that are unique
                        if comment_id in processed_comments: continue
                        # comments that are distinct from modlog
                        #if comment_id in modlog_ids_by_community[community]: continue
                        # # local=true
                        # if not entry['comment']['local']: continue
                        # comments w/ same description as in modlog
                        if entry['community'].get('description', '') != modlog_descriptions[community]: continue
                        # comments w/ remove=false
                        if entry['comment']['removed']: continue
                        if entry['comment']['deleted']: continue
                        # comments that are not removed (string match with *Permanently deleted*, mass removed, etc)
                        if entry['comment']['content'].strip().lower() == '*permanently deleted*': continue
                        # comments w/ more than just urls
                        entry['comment']['content'] = replace_urls(markdown_to_text(entry['comment']['content']),
                                                                URL_REPLACEMENT_TOKEN)
                        if len(''.join(entry['comment']['content'].split(URL_REPLACEMENT_TOKEN)).strip()) == 0: continue
                        # comments w/ 30<=chars excluding urls<=400 (decide after plotting)
                        if not (min_char <= len(entry['comment']['content']) <= max_char): continue

                        entry['instance'] = instance
                        sampler.add(entry)
                        processed_comments.add(comment_id)
    return [encode_positive_data(entry, modlog_rules) for sampler in samplers.values() for entry in sampler.reservoir]

def get_all_modlog_comments(data_dir = '../../data'):
    
    modlog_folder = os.path.join(data_dir, 'raw', 'modlogs')
    
    comments = set()
    for instance in tqdm(os.listdir(modlog_folder)):
        modlog_file = os.path.join(modlog_folder, instance, 'removed_comments.jsonl')
        if not os.path.exists(modlog_file): continue
        with open(modlog_file, encoding='utf8') as f:
            for entry in map(json.loads, f):
                if "comment" not in entry: continue
                if "ap_id" not in entry['comment']: continue
                comments.add(entry['comment']['ap_id'])
    with open(os.path.join(data_dir, 'interim', 'all_modlogs.pkl'), 'wb') as pfile:
        pkl.dump(comments, pfile)
    return list(comments)
                
def get_selected_modlogs(data_dir):
    with open(os.path.join(data_dir, 'interim', 'v2_binary_modlogs.jsonl'), encoding='utf8') as mfile:
        selected_modlogs = list(map(json.loads, mfile))
    return selected_modlogs


def main(data_dir='../../data', min_char=30, max_char=400, recompute=False):
    print("getting all modlog comments")
    if recompute:
        all_modlog_comments = get_all_modlog_comments(data_dir)  #TODO
    else:
        with open(os.path.join(data_dir, 'interim', 'v2_all_modlogs.pkl'), 'rb') as pfile:
            all_modlog_comments = pkl.load(pfile)

    print("getting rule mapped modlog comments")

    binary_mods = get_selected_modlogs(data_dir)
    
    nonbinary_mods = [smc for smc in binary_mods if smc['applied_rule_n']!=-1]

    nonbinary_positive = sample_positive_data(nonbinary_mods, all_modlog_comments, data_dir=data_dir, min_char=min_char, max_char=max_char)

    print("creating nonbinary positive dat")
    with open(os.path.join(data_dir, 'interim', 'v2_nonbinary_positive.json'), 'w+', encoding='utf8') as f:
        json.dump(nonbinary_positive, f, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(data_dir, 'interim', 'v2_nonbinary_positive.jsonl'), 'w+', encoding='utf8') as f:
        for entry in nonbinary_positive:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + '\n')

    print("creating binary positive data")

    binary_positive = sample_positive_data(binary_mods, all_modlog_comments, data_dir=data_dir, min_char=min_char, max_char=max_char)
    with open(os.path.join(data_dir, 'interim', 'v2_binary_positive.json'), 'w+', encoding='utf8') as f:
        json.dump(binary_positive, f, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(data_dir, 'interim', 'v2_binary_positive.jsonl'), 'w+', encoding='utf8') as f:
        for entry in binary_positive:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
