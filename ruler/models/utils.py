import copy
import random
from functools import partial
from hashlib import md5
import pandas as pd 
import os
import json 

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
            to_return = f"[RULE{rule_n}] " + to_return
        else:
            to_return = f"{rule_n}. " + to_return
    return to_return


def data_to_qa(entry, custom_tokens=True, custom_rule_tokens=False, skip_numbers=False, comment_in_question=False):

    rules = f"""
[BOR]
""" if custom_tokens else ""
    rule_boundaries = dict()
    for rule_n, rule_text in entry['community']['rules']:
        formatted_rule_text = format_rule(rule_n if (not skip_numbers) else None, rule_text,
                                          custom_rule_tokens=custom_rule_tokens)
        rule_boundaries[str(rule_n)] = len(rules), len(rules) + len(formatted_rule_text) + 1
        rules += formatted_rule_text + '\n'
    try:
        answer_start_char, answer_end_char = rule_boundaries[str(entry['applied_rule_n'])]
    except:
        print(entry['community']['rules'], entry['applied_rule_n'])
        raise
    rules = rules[:-1]  # remove final newline
    if custom_tokens:
        rules = f"""{rules}
[EOR]
"""

    # question = "Check if in the following message there is unsafe content according to the specified policy and categories."
    question = entry['community']['actor_id'] if not custom_tokens else f"""[BOQ]
{entry['community']['actor_id']}
[EOQ]"""
    context = rules  #FIXME: the comment itself could be truncated! swap in the context, starting with the comment and then enumerating rules?
    if comment_in_question:
        if custom_tokens:
            question = f"""
[BOC]
{entry['content']}
[EOC]
""" + question
        else:
            question = entry['content'] + '\n' + question
    else:
        if custom_tokens:
            context += f"""
[BOC]
{entry['content']}
[EOC]
    """
        else:
            context += entry['content']
    answers = dict(rule_number=entry['applied_rule_n'],
                   text=entry['applied_rule_text'],
                   removed=entry['removed'],
                   start_char=answer_start_char,
                   end_char=answer_end_char,
                   rule_boundaries=rule_boundaries
                   )

    entry = copy.deepcopy(entry)
    entry['rules'] = {str(i): j for i, j in entry['community'].pop('rules')}
    # print(entry)
    return dict(question=question, context=context, answers=answers, entry=entry,
                id=md5((question + context).encode('utf-8')).hexdigest())


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
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        # max_length=384, # bert base has num_tokens=512
        # truncation="only_second",
        truncation=truncation,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    rule_token_boundaries = list()
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        sequence_ids = inputs.sequence_ids(i)
        current_rule_token_boundaries = char2token_boundaries(answer, offset, sequence_ids)
        start_token, end_token = current_rule_token_boundaries[str(answer['rule_number'])]
        start_positions.append(start_token)
        end_positions.append(end_token)
        rule_token_boundaries.append(current_rule_token_boundaries)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    inputs['rule_token_boundaries'] = rule_token_boundaries
    inputs['answer_rule_number'] = [answer['rule_number'] for answer in answers]
    inputs['answer_removed'] = [answer['removed'] for answer in answers]
    return inputs


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
        remap[0] = 0
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
    rules = entry['community'].pop('rules')
    sampled_idx = random.randint(1, len(rules) - 1)
    sampled_rule = rules.pop(sampled_idx)[0]
    entry['community']['rules'] = rules
    if entry['applied_rule_n'] == sampled_rule:
        entry['applied_rule_n'] = 0
        try:
            entry['applied_rule_text'] = dict(entry['community']['rules'])[0]
        except KeyError: # 0. Safe was removed from the rule set: add it back
            entry['applied_rule_text'] = 'Safe'
            entry['community']['rules'].insert(0, (0, 'Safe'))
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
        entry['community']['rules'] = rules
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

  
