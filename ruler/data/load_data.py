## reads a csv with extracted rules and converts into trainable data
import copy
import json
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = '../../data/processed'
"""
# TODO:

## filter 
- remove *Permanently deleted* from positive comments
[x] keep only English [rules]
- remove modlog cases for which we know multiple rules apply (or create one each?)
- crop comments/rules so that they fit comment context?
- remove "mass removed"?
- remove communities with more than 20 rules? e.g., 'community_id': 56378,  'instance': 'lemmy.ca', has 99 rules
[x] communities whose rules don't change?

## format
- custom token to divide rule section [BOR][EOR] and possibly, comment section [BOC][EOC]
- custom token per rule? [RULEn]
- format for BERTQA:
{
    question: "Check if in the following message there is unsafe content according to the specified policy and categories:"
    rules = "".join([f"{n}: {rule}\n" for n, rule in entry["rules"].items()])
    entry = {
        // as below
    },
    answers: {
        rule_number: 1,
        text: "1: ...",
        start_char:1,
        end_char:10,
        removed: True,
    },
    context: "
            [BOR]
            {rules}
            [EOR]
            [BOC]
            {entry['comment']}
            [EOC]
        ",
    id:"123"
}
- format for entries:
{
    rules:{1:"...",},
    community_id:"123",
    comment_id:"123",
    comment:"...",
    instance:"...",
    rule_number: 1,
    reason:"...",
    removed: True,
}

## data splits
### single community
- test 1: held out rule
- test 2: the rest
### multi community
- test 1: held out community
- test 2: held out rule
- test 3: the rest
- val?
- train: stratify per community/rule?

## ablations
- exclude rule number (the order of the rule may carry information about frequency of application as well as priority for the community)

## data augmentation
- randomize rule number <> rule text associations
- add variants of the data by excluding rules selectively (keep the `safe` rule?)
- shuffle rule order in the context
"""


def load_data(training_dir='../../data/interim/training_Data', task='nonbinary', min_rules=2,
              max_rules=20):  #TODO: shall we handle min/max rules here?
    data = list()
    with open(f'{training_dir}/{task}/{task}_modlogs.jsonl', encoding='utf-8') as f:
        data.extend(map(json.loads, f))
    with open(f'{training_dir}/{task}/{task}_positive.jsonl', encoding='utf-8') as f:
        data.extend(map(json.loads, f))
    to_return = list()
    for i in data:
        rules = i['community'].pop('rules')
        rules = rules.pop('rules')
        rules = {int(k): v for k, v in rules.items() if (v is not None) and (len(v.strip()) > 0)}
        if not (min_rules <= len(rules) <= max_rules):
            print(f"discarding {len(rules)} rules")
            continue
        if 0 in rules:  # could also remap to [1, N]
            rules = {k + 1: v for k, v in rules.items()}
            i['applied_rule_n'] += 1
        rules[0] = "Safe"
        i['community']['rules'] = copy.deepcopy(rules)
        # i['community']['rules'] = list(sorted(rules.items()))
        to_return.append(i)
    return to_return

def split(data, strategy='stratified', test_size=0.2, n_sample=10, random_state=0):
    df = pd.DataFrame(data)
    original_columns = df.columns
    df['community_actor_id'] = df.community.apply(lambda x: x['actor_id'])

    if strategy == 'stratified':
        train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['removed'],
                                       shuffle=True, )
    elif strategy == 'leave_n_communities_out':
        sampled_communities = df.community_actor_id.drop_duplicates().sample(n=n_sample, random_state=random_state,
                                                                             replace=False)
        train, test = df[~df.community_actor_id.isin(sampled_communities)], df[
            df.community_actor_id.isin(sampled_communities)]
        train, test = train.sample(frac=1).reset_index(drop=True), test.sample(frac=1).reset_index(drop=True)
    elif strategy == 'leave_n_rules_out':
        df['rule_id'] = df.apply(lambda row: (row.applied_rule_text, row.community_actor_id), axis=1)
        sampled_rules = df.rule_id.drop_duplicates().sample(n=n_sample, random_state=random_state, replace=False)

        sampled_rules_dict = defaultdict(list)
        for i, j in sampled_rules.values:
            sampled_rules_dict[j].append(i)
        train, test = df[~df.rule_id.isin(sampled_rules)].copy(), df[df.rule_id.isin(sampled_rules)].copy()
        def remove_rule(row):
            community = row.community
            community['rules'] = {k:v for k, v in community['rules'].items() if v not in sampled_rules_dict[row.community_actor_id]}
            assert row.applied_rule_n in community['rules']
            return community
        train.community = train.apply(remove_rule, axis=1)
        train, test = train.sample(frac=1).reset_index(drop=True), test.sample(frac=1).reset_index(drop=True)
    else:
        raise NotImplementedError(f'strategy {strategy} not recognized')
    return train[original_columns].to_dict(orient='records'), test[original_columns].to_dict(orient='records')


"""
## data augmentation
- randomize rule number <> rule text associations
- add variants of the data by excluding rules selectively (keep the `safe` rule?)
- shuffle rule order in the context
"""

def load_normvio(base_folder = '../../data/external/NormVio'):
    with open(f'{base_folder}/train.jsonl', encoding='utf8') as f, open(f'{base_folder}/dev.jsonl',
                                                                                  encoding='utf8') as ff:
        normvio = list(map(json.loads, f)) + list(map(json.loads, ff))
    community_rules = defaultdict(set)
    for entry in normvio:
        community_rules[entry['subreddit']].update(entry['rule_texts'].split(' ||| '))
    community_rules = {k: {i: j for i, j in enumerate(sorted(v))} for k, v in community_rules.items()}
    reverse_community_rules = {k: {j: i for i, j in v.items()} for k, v in community_rules.items()}
    normvio = [dict(community=dict(actor_id=entry['subreddit'], rules=community_rules[entry['subreddit']]),
                    id=entry['comment_id'], content=entry['final_comment']['tokens'],
                    applied_rule_n=[reverse_community_rules[entry['subreddit']][i] for i in
                                    entry['rule_texts'].split(' ||| ')],
                    applied_rule_text=entry['rule_texts'].split(' ||| '), removed=True) for entry in normvio]
    normvio_linear = list()
    cntr = 0
    for entry in normvio:
        for rule_n, rule_text in zip(entry.pop('applied_rule_n'), entry.pop('applied_rule_text')):
            entry_ = copy.deepcopy(entry)
            entry_['applied_rule_n'] = rule_n
            entry_['applied_rule_text'] = rule_text
            entry_['id'] = f'normvio_{cntr}'
            cntr += 1
            normvio_linear.append(entry_)
    return normvio_linear

def load_aegis(base_folder = '../../data/external'):
    aegis = pd.read_csv(f'{base_folder}/aegis_ai_content_safety_dataset.csv').rename(
        columns={'label': "applied_rule_n", 'rule_name': 'applied_rule_text', 'comment': 'content'})
    rules = {i['applied_rule_n']: i['applied_rule_text'] for i in
             aegis[['applied_rule_n', 'applied_rule_text']].drop_duplicates().to_dict(orient='records')}
    aegis = aegis.to_dict(orient='records')
    for idx, i in enumerate(aegis):
        i['id'] = f'aegis_{idx}'
        i['community'] = {'actor_id': 'aegis_ai_content_safety', 'rules': rules}
        i['removed'] = i['applied_rule_text'] == 'safe'
    return aegis

# def load_merged(base_folder = '../../data/external'):
#     aegis = pd.read_csv(f'{base_folder}/merged.csv').rename(
#         columns={'label': "applied_rule_n", 'rule_name': 'applied_rule_text', 'comment': 'content'})
#     rules = {i['applied_rule_n']: i['applied_rule_text'] for i in
#              aegis[['applied_rule_n', 'applied_rule_text']].drop_duplicates().to_dict(orient='records')}
#     aegis = aegis.to_dict(orient='records')
#     for idx, i in enumerate(aegis):
#         i['id'] = f'merged_{idx}'
#         i['community'] = {'actor_id': 'merged', 'rules': rules}
#         i['removed'] = i['applied_rule_text'] == 'safe'
#     return aegis
if __name__ == '__main__':
    data = load_data()
    print(f'date len: {len(data)}')

    print(f"data looks like this: \n{data[0]}")
    train, test = split(data, random_state=42, strategy='stratified')
    print(f"len train: {len(train)}, test: {len(test)}")

