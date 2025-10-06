import copy
import random
from hashlib import md5
import string
import os
from ruler.features.make_splits import read_split

import torch
import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer
import datasets
from functools import partial
from ruler.data.load_data import load_aegis, load_normvio
from torch.utils.data import DataLoader
from ruler.models.grokfast import gradfilter_ema
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import json


def prep_for_qa(data):
    to_return = list()
    for entry in data:
        rules = list(sorted(entry['community']['rules'].items()))
        entry = copy.deepcopy(entry)
        lettered_rules = [(letter, text) for letter, (_, text) in zip(string.ascii_uppercase, rules)]
        entry['community']['rules'] = lettered_rules
        for letter, text in lettered_rules:
            if text.strip() == entry['applied_rule_text'].strip():
                entry['applied_rule_n'] = letter
        entry['reason'] = f"Rule {entry['applied_rule_n']}"
        to_return.append(entry)
    return to_return

def shuffled_rules(entry, only_unsafe):
    entry = copy.deepcopy(entry)
    if only_unsafe:
        random.shuffle(entry['community']['rules'])
    else:
        unsafe_rules = entry['community']['rules'][1:]
        random.shuffle(unsafe_rules)
        entry['community']['rules'] = [entry['community']['rules'][0]] + unsafe_rules
    return entry

def permutated_numbers(entry, only_unsafe):
    entry = copy.deepcopy(entry)
    rules = entry['community'].pop('rules')
    if only_unsafe:
        rule_keys = [i[0] for i in rules[1:]]
        new_rule_keys = rule_keys[:]
        random.shuffle(new_rule_keys)
        remap = dict(zip(rule_keys, new_rule_keys))
        remap['A'] = 'A'
    else:
        rule_keys = [i[0] for i in rules]
        new_rule_keys = rule_keys[:]
        random.shuffle(new_rule_keys)
        remap = dict(zip(rule_keys, new_rule_keys))
    entry['community']['rules'] = [(remap[i], j) for i, j in rules]
    try:
        entry['applied_rule_n'] = (remap[(entry['applied_rule_n'])])
    except:
        print(entry['applied_rule_n'], entry['applied_rule_text'], entry['community']['rules'], rules, remap)
        raise
    return entry

def excluded_rules(entry):

    entry = copy.deepcopy(entry)
    original = entry['community'].pop('rules')
    rules = original
    removable_letters = [letter for letter, text in rules if letter != 'A' ]
    letter_to_remove = random.choice(removable_letters)
    rules = [r for r in rules if r[0] != letter_to_remove]
    entry['community']['rules'] = rules
    if entry['applied_rule_n'] == letter_to_remove:
        entry['applied_rule_n'] = 'A'
        try:
            entry['applied_rule_text'] = dict(entry['community']['rules'])['A']
        except KeyError: # 0. Safe was removed from the rule set: add it back
            entry['applied_rule_text'] = 'Safe'
            entry['community']['rules'].insert(0, ('A', 'Safe'))
    return entry

def augment_qa(data, only_unsafe=True, shuffle_rules=True, permutate_numbers=True, exclude_rules=True,
               n_replicas_per_strategy=3):
    augmentations = list()
    for entry in data:
        for _ in range(n_replicas_per_strategy):
            if shuffle_rules:
                augmentations.append(shuffled_rules(entry, only_unsafe=only_unsafe))
            if permutate_numbers:
                augmentations.append(permutated_numbers(entry, only_unsafe=only_unsafe))
            if exclude_rules:
                augmentations.append(excluded_rules(entry))
    return data + augmentations

def format_rule(rule_n, rule_text, custom_rule_tokens=False):
    to_return = rule_text
    if rule_n is not None:
        if custom_rule_tokens:
            to_return = f"[RULE {rule_n}] " + to_return
        else:
            to_return = f"{rule_n}. " + to_return
    return to_return

def data_to_qa(entry, max_choices, custom_tokens=True, custom_rule_tokens=False, skip_numbers=False):

    rules = "[BOR]\n" if custom_tokens else ""

    for rule_n, rule_text in entry['community']['rules']:
        rules += format_rule(rule_n if (not skip_numbers) else None, rule_text, custom_rule_tokens=custom_rule_tokens) + '\n'
    rules = rules.rstrip()  # убираем последнюю \n
    if custom_tokens:
        rules = f"{rules}\n[EOR]"
    question = "Which rule is violated based on the given comment?"
    
    choices = []
    correct_idx = None

    for idx, (letter, _) in enumerate(entry['community']['rules']):
        if custom_tokens:
            context = f"""{rules}
[BOC]
{entry['content']}
[EOC]
[Answer]
Rule {letter}"""
        
        else:
            context = f"""{rules}\n{entry['content']}\nAnswer: Rule {rule_n}"""

        choices.append(context)
    
            # Найдём правильный индекс
        if letter == entry['applied_rule_n']:
            correct_idx = idx

    if len(choices) < max_choices:
        padding_needed = max_choices - len(choices)
        choices += ["[BOR]\n[EOR]\n[BOC]\n[EOC]\n[Answer]\nRule X"] * padding_needed

    return {
        "id": entry.get("ap_id", entry.get("modlog_id", "no_id")),
        "question": question,
        "choices": choices,
        "labels": correct_idx
    }

def tokenize(examples, tokenizer, truncation='only_second'):
    num_choices = len(examples["choices"][0])
    second_sentence = sum(examples["choices"], [])
    first_sentence = [""] * len(second_sentence)
    inputs = tokenizer(
        first_sentence,
        second_sentence,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    result = {k: [inputs[k][i:i + num_choices] for i in range(0, len(inputs[k]), num_choices)]
              for k in inputs}
    
    result["labels"] = list(examples["labels"])
    
    return result

def prepare_data(data, tokenizer, max_choices, custom_tokens=True, custom_rule_tokens=False, skip_numbers=False, augment=False,
                 only_unsafe=False, shuffle_rules=True, permutate_numbers=True, exclude_rules=True,
                 n_replicas_per_strategy=1, model_name='bert', external_datasets = False):
    if augment:
        qa_ = augment_qa(data, only_unsafe=only_unsafe, shuffle_rules=shuffle_rules,
                            permutate_numbers=permutate_numbers, exclude_rules=exclude_rules,
                            n_replicas_per_strategy=n_replicas_per_strategy)
        if external_datasets:
            qa_+=prep_for_qa(random.sample(load_aegis(), 5000))
            qa_+=prep_for_qa(random.sample(load_normvio(), 5000))
    else:
        qa_ = data

    qa = list(map(partial(data_to_qa, max_choices = max_choices, custom_tokens=custom_tokens, custom_rule_tokens=custom_rule_tokens,
                            skip_numbers=skip_numbers), qa_))

    for idx, q in enumerate(qa):
            q['id'] = str(idx)

    dataset_qa = datasets.Dataset.from_list(qa)

    tokenized_dataset = dataset_qa.map(
            partial(tokenize, tokenizer=tokenizer, truncation="only_second") if model_name == 'bert' 
            else partial(tokenize, tokenizer=tokenizer, truncation=True),
            batched=True,
            remove_columns=dataset_qa.column_names,
            num_proc=4,
        )

    dataset = tokenized_dataset.with_format("torch")
    return dataset, dataset_qa

def train_step(gradient_acc_step, grads, device, epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    steps = 0
    for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}| Loss: {train_loss / len(train_loader)}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss_value = outputs.loss
        loss_value = loss_value / gradient_acc_step
        loss_value.backward()

        if steps % gradient_acc_step == 0:
            optimizer.step()
            grads = gradfilter_ema(model, grads=grads)
            optimizer.zero_grad()
        
        train_loss += loss_value.item()
        steps += 1

    print(f"Epoch {epoch}, train loss: {train_loss / len(train_loader)}")
    return grads

def print_classification_metrics(labels, preds):
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    print(f"Accuracy Score: {accuracy:.4f}")
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_true=labels, y_pred=preds, zero_division=0))
    confusion = confusion_matrix(y_true=labels, y_pred=preds)
    print(confusion)

    return accuracy, f1

def eval_step(model, dataset, formatted_eval_dataset, tokenizer, device, batch_size=16):
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    loss_value = 0
    preds = list()
    labels = list()

    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            final_logit = logits.argmax(dim=1)

            loss_value += outputs.loss.item()
            labels.extend(batch['labels'].tolist())
            preds.extend(final_logit.tolist())
    
    print_classification_metrics(labels, preds)
    print(f"Eval loss: {loss_value / len(eval_loader)}")
    return labels, preds

def save_results(labels, preds, fname='bertqa.json'):
    results_folder = "/content/drive/MyDrive/ruler-master/reports/results/"
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, fname), 'w+') as f:
        json.dump({'labels': labels, 'preds': preds}, f)

model_name = 'bert'
task = 'nonbinary'
split_n = 0
custom_tokens = True
custom_rule_tokens = False
skip_numbers = False
batch_size = 8
augment=True
only_unsafe=False
shuffle_rules=True
permutate_numbers=True
exclude_rules=True
external_datasets=False
n_replicas_per_strategy=1
n_epochs = 5

train, test_stratified, test_n_rules_out, test_n_communities_out = map(prep_for_qa, read_split(task="nonbinary", split_n=0))
all_data = train + test_stratified + test_n_rules_out + test_n_communities_out
max_choices = max(len(all_data[i]['community']['rules']) for i in range(len(all_data)))

model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

if custom_tokens:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[BOR]", "[EOR]", "[BOC]", "[EOC]", "[ANSWER]"]}
    )

if custom_rule_tokens:
    all_rules = set(j[0] for split in (train, test_stratified, test_n_rules_out, test_n_communities_out) 
                    for i in split 
                    for j in i['community']['rules'])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"[RULE {rule_n}]" for rule_n in all_rules]}
    )

model.resize_token_embeddings(len(tokenizer))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"preparing train, len: {len(train)}")
dataset, eval_dataset = prepare_data(train, tokenizer, max_choices, custom_tokens=custom_tokens, custom_rule_tokens=custom_rule_tokens,
                                                skip_numbers=skip_numbers, augment=augment,
                                                only_unsafe=only_unsafe, shuffle_rules=shuffle_rules, permutate_numbers=permutate_numbers,
                                                exclude_rules=exclude_rules,
                                                n_replicas_per_strategy=n_replicas_per_strategy, model_name=model_name, external_datasets=external_datasets)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_sets = dict()
for test_name, test_set in (('test_stratified', test_stratified), ('test_n_rules_out', test_n_rules_out),
                            ('test_n_communities_out', test_n_communities_out)):
    print(f"preparing  {test_name}, len: {len(test_set)}")
    dataset, eval_dataset = prepare_data(data=test_set, tokenizer=tokenizer, max_choices = max_choices,
                                        custom_rule_tokens=custom_rule_tokens,
                                        custom_tokens=custom_tokens, skip_numbers=skip_numbers,
                                        model_name=model_name)
    test_sets[test_name] = dataset, eval_dataset

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

gradient_acc_step = 1
grads = None 

model_save_dir = "/content/drive/MyDrive/ruler-master/models/huggingface"
os.makedirs(model_save_dir, exist_ok=True)

for epoch in range(n_epochs):
    grads = train_step(gradient_acc_step, grads, device, epoch, model, optimizer, train_loader)
    model.save_pretrained(os.path.join(model_save_dir, f'{model_name}qa_onsplit_{epoch}'))
    for test_name, (test_set, eval_dataset) in test_sets.items():
            print(f"testing {test_name}, len: {len(test_set)}")
            labels, preds = eval_step(model, test_set, eval_dataset, tokenizer, device,
                                                         batch_size)
            save_results(labels, preds, fname=f'{model_name}qa_{test_name}.json')









