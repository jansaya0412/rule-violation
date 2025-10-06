import json
import os

from tqdm import tqdm

from ruler.data.load_data import load_data, split


def read_split(task='nonbinary', split_n=0, base_folder = 'data/interim/splits'):
    in_folder = f'{base_folder}/{task}/{split_n}'
    print(f'{base_folder}/{task}/{split_n}')
    with open(f'{in_folder}/train.json', 'r', encoding='utf8') as f:
        train = json.load(f)
    with open(f'{in_folder}/test_stratified.json', 'r', encoding='utf8') as f:
        test_stratified = json.load(f)
    with open(f'{in_folder}/test_n_rules_out.json', 'r', encoding='utf8') as f:
        test_n_rules_out = json.load(f)
    with open(f'{in_folder}/test_n_communities_out.json', 'r', encoding='utf8') as f:
        test_n_communities_out = json.load(f)
    for data in [train, test_stratified, test_n_rules_out, test_n_communities_out]:
        for entry in data:
            entry['community']['rules'] = {int(i):j for i, j in entry['community']['rules'].items()}
    return train, test_stratified, test_n_rules_out, test_n_communities_out


def main(task='nonbinary', min_rules=2, max_rules=20, random_state=42, test_size=.2, n_communities=10, n_rules=20, n_spits=10):
    data = load_data(task=task, min_rules=min_rules, max_rules=max_rules)
    for split_n in tqdm(range(n_spits)):
        train, test_n_communities_out = split(data, random_state=random_state + split_n,
                                              strategy='leave_n_communities_out', n_sample=n_communities)
        train, test_n_rules_out = split(train, random_state=random_state + split_n, strategy='leave_n_rules_out',
                                        n_sample=n_rules)
        train, test_stratified = split(train, random_state=random_state + split_n, strategy='stratified',
                                       test_size=test_size)
        out_folder = f'../../data/interim/splits/{task}/{split_n}'
        os.makedirs(out_folder, exist_ok=True)
        with open(f'{out_folder}/train.json', 'w+', encoding='utf8') as f:
            json.dump(train, f)
        with open(f'{out_folder}/test_stratified.json', 'w+', encoding='utf8') as f:
            json.dump(test_stratified, f)
        with open(f'{out_folder}/test_n_rules_out.json', 'w+', encoding='utf8') as f:
            json.dump(test_n_rules_out, f)
        with open(f'{out_folder}/test_n_communities_out.json', 'w+', encoding='utf8') as f:
            json.dump(test_n_communities_out, f)


if __name__ == '__main__':
    task = 'nonbinary'
    min_rules = 2
    max_rules = 20
    random_state = 42
    test_size = .2
    n_communities = 20
    n_rules = 20
    n_spits = 10
    main(task=task, min_rules=min_rules, max_rules=max_rules, random_state=random_state, test_size=test_size, n_communities=n_communities, n_rules=n_rules, n_spits=n_spits)
    for i in range(n_spits):
        train, test_stratified, test_n_rules_out, test_n_communities_out = read_split(task=task, split_n=i)
        print(f"split {i}: len train = {len(train)}, len test_stratified = {len(test_stratified)}, len test_n_rules_out = {len(test_n_rules_out)}, len test_n_communities_out = {len(test_n_communities_out)}")
