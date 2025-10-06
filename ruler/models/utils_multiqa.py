import copy
import random
from functools import partial
from hashlib import md5
import pandas as pd 
import string

import datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import BertTokenizerFast, BertForQuestionAnswering

from ruler.features.make_splits import read_split


def classification_metrics(labels, preds):
    report = classification_report(y_true=labels, y_pred=preds, zero_division=0, output_dict=True)
    confusion = confusion_matrix(y_true=labels, y_pred=preds)
    return dict(classification_report=report,
                confusion_matrix=confusion, labels=labels, predictions=preds)


def print_classification_metrics(labels, preds):
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    print(f"Accuracy Score: {accuracy:.4f}")
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_true=labels, y_pred=preds, zero_division=0))
    confusion = confusion_matrix(y_true=labels, y_pred=preds)
    print(confusion)

    return accuracy, f1


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


def char2token_boundaries(answer, offset, sequence_ids):
    # Find the start and end of the context
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # assume start_char[i]>=end_char[i-1]
    idx = context_start
    rule_token_boundaries = dict()
    for rule_n, (start_char, end_char) in sorted(filter(lambda x: x[1] is not None, answer['rule_boundaries'].items()),
                                                 key=lambda x: x[1][0]):

        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_token = idx - 1

        while idx <= context_end and offset[idx][0] <= end_char:
            idx += 1
        end_token = idx - 1

        rule_token_boundaries[rule_n] = start_token, end_token
    for rule_n, _ in filter(lambda x: x[1] is None, answer['rule_boundaries'].items()):
        rule_token_boundaries[rule_n] = [-1, -1]
    return rule_token_boundaries


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
    
    result["labels"] = examples["labels"]
    
    return result


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


if __name__ == '__main__':
    task = 'nonbinary'
    split_n = 0
    train, test, test_n_rules_out, test_n_communities_out = read_split(task=task, split_n=split_n)
    print(f"len train: {len(train)}, test: {len(test)}")
    print(f"data looks like this: \n{train[0]}")

    custom_tokens = True
    custom_rule_tokens = False
    only_unsafe = False
    shuffle_rules = True
    permutate_numbers = True
    exclude_rules = True
    n_replicas_per_strategy = 3
    qa_ = augment_qa(prep_for_qa(train), only_unsafe=only_unsafe, shuffle_rules=shuffle_rules,
                     permutate_numbers=permutate_numbers, exclude_rules=exclude_rules,
                     n_replicas_per_strategy=n_replicas_per_strategy)
    qa = list(
        map(partial(data_to_qa, custom_tokens=custom_tokens, custom_rule_tokens=custom_rule_tokens, skip_numbers=False),
            qa_))
    examples = datasets.Dataset.from_list(qa)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if custom_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[BOR]", "[EOR]", "[BOC]", "[EOC]"]}
        )
    if custom_rule_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"[RULE{rule_n}]" for rule_n in
                                           range(max(len(entry['rules'] for entry in qa)) + 1)]}
        )
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))

    inputs = tokenize(examples, tokenizer)

  
