#!/usr/bin/env python
# coding: utf-8
import json
import os
import sys
from collections import Counter
from functools import partial
import random

import datasets
import numpy as np
import torch
import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from transformers import BertTokenizerFast, BertForQuestionAnswering, RobertaForQuestionAnswering, RobertaTokenizer, \
    RobertaTokenizerFast, XLMRobertaForQuestionAnswering, XLMRobertaTokenizerFast, AutoModelForQuestionAnswering, \
    AutoTokenizer

from ruler.data.load_data import load_aegis, load_normvio
from ruler.features.make_splits import read_split
from ruler.models.utils import classification_metrics, print_classification_metrics, data_to_qa, tokenize, \
    augment_qa, prep_for_qa
from ruler.models.grokfast import gradfilter_ema
from torch.utils.data import DataLoader
from evaluate import evaluator


def print_trainable_params(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || Total params: {all_params} || Trainable %: {trainable_params / all_params * 100:.2f}%"
    )


def format_answers(example):
    # print(example)
    return {
        "id": example["id"],
        "answers": {
            "text": [example["answers"]["text"]],  # Wrap in a list
            # Wrap in a list
            "answer_start": [example["answers"]["start_char"]],
        },
    }


def extract_predictions(outputs, batch):
    starts_within_gt = list()
    ends_within_gt = list()

    for s, e, gts, gte in zip(outputs['start_logits'].argmax(1), outputs['end_logits'].argmax(1),
                              batch['start_positions'], batch['end_positions']):
        s, e, gts, gte = s.item(), e.item(), gts.item(), gte.item()
        starts_within_gt.append(gts <= s < gte)
        ends_within_gt.append(gts < e <= gte)

    predicted_rules_by_start = np.full(len(starts_within_gt),
                                       0)  #FIXME: shall we distinguish non-predictions from predictions of "Safe"?
    predicted_rules_by_end = np.full(len(starts_within_gt), 0)
    for rule, rule_boundaries in batch['rule_token_boundaries'].items():
        # print(f'rule {rule}')

        for i, s, e, (rs, re) in zip(range(len(rule_boundaries)), outputs['start_logits'].argmax(1),
                                     outputs['end_logits'].argmax(1), rule_boundaries):
            s, e, rs, re = s.item(), e.item(), rs.item(), re.item()
            if rs <= s < re:
                predicted_rules_by_start[i] = int(rule)
            if rs < e <= re:
                predicted_rules_by_end[i] = int(rule)
    return starts_within_gt, ends_within_gt, predicted_rules_by_start, predicted_rules_by_end


def main(model_name = 'bert', task = 'nonbinary', split_n = 0, custom_tokens = True, custom_rule_tokens = False,
         skip_numbers = False, batch_size = 16, augment=True,
                                                   only_unsafe=False, shuffle_rules=True, permutate_numbers=True,
        exclude_rules=True, external_datasets=False,
        n_replicas_per_strategy=1,
        n_epochs = 10):
    """

    :param model_name: either bert or roberta
    :return: None
    """
    # load data

    train, test_stratified, test_n_rules_out, test_n_communities_out = read_split(task=task, split_n=split_n)
    print(f"len train: {len(train)}, test: {len(test_stratified)}")
    print(f"data looks like this: \n{train[0]}")

    # load models
    #
    # all_rules = set(j  for split in
    #                 (test_stratified,) for i in split for j in i['community']['rules'].keys())
    # print(all_rules)



    if model_name == 'roberta':
        model = XLMRobertaForQuestionAnswering.from_pretrained('xlm-roberta-base')
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", cache_dir='../../models/huggingface')
    else:

        # tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        # model: BertForQuestionAnswering = AutoModelForQuestionAnswering.from_pretrained(
        #     "google/electra-base-discriminator"
        # )

        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir='../../models/huggingface')
    if custom_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[BOR]", "[EOR]", "[BOC]", "[EOC]", "[BOQ]", "[EOQ]"]}
        )
    if custom_rule_tokens:
        # max_rules = max(max(i['community']['rules'].keys()) for split in
        #                 (train, test_stratified, test_n_rules_out, test_n_communities_out) for i in split)
        all_rules = set(j for split in
                        (train, test_stratified, test_n_rules_out, test_n_communities_out) for i in split for j in
                        i['community']['rules'].keys())
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"[RULE{rule_n}]" for rule_n in
                                           all_rules]}
        )
    model.resize_token_embeddings(len(tokenizer))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # prepare train data

    # import random
    # train = random.sample(train, 100)
    print(f"preparing train, len: {len(train)}")
    dataset, formatted_eval_dataset = prepare_data(train, tokenizer, custom_tokens=custom_tokens, custom_rule_tokens=custom_rule_tokens,
                                                   skip_numbers=skip_numbers, augment=augment,
                                                   only_unsafe=only_unsafe, shuffle_rules=shuffle_rules, permutate_numbers=permutate_numbers,
                                                   exclude_rules=exclude_rules,
                                                   n_replicas_per_strategy=n_replicas_per_strategy, model_name=model_name, external_datasets=external_datasets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # prepare test data
    test_sets = dict()
    for test_name, test_set in (('test_stratified', test_stratified), ('test_n_rules_out', test_n_rules_out),
                                ('test_n_communities_out', test_n_communities_out)):
        print(f"preparing  {test_name}, len: {len(test_set)}")
        dataset, formatted_eval_dataset = prepare_data(data=test_set, tokenizer=tokenizer,
                                                       custom_rule_tokens=custom_rule_tokens,
                                                       custom_tokens=custom_tokens, skip_numbers=skip_numbers,
                                                       model_name=model_name)
        test_sets[test_name] = dataset, formatted_eval_dataset

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.01)
    loss = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)


    gradient_acc_step = 1
    grads = None
    # alpha = 0.9
    # lamb = 0.1
    # main loop
    model_save_dir = "../../models/huggingface/"
    os.makedirs(model_save_dir, exist_ok=True)
    for epoch in range(n_epochs):
        grads = train_step(gradient_acc_step, device, epoch, grads, loss, model, optimizer, train_loader)
        model.save_pretrained(os.path.join(model_save_dir, f'{model_name}qa_onsplit_{epoch}'))
        for test_name, (test_set, formatted_eval_dataset) in test_sets.items():
            print(f"testing {test_name}, len: {len(test_set)}")
            results_by_start, results_by_end = eval_step(model, test_set, formatted_eval_dataset, tokenizer, device,
                                                         batch_size)
            save_results(results_by_start, results_by_end, fname=f'{model_name}qa_{test_name}_epoch_{epoch}.json')


def train_step(GRADIENT_ACC_STEPS, device, epoch, grads, loss, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    steps = 0
    for batch in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch} | Loss: {train_loss / len(train_loader)}"
    ):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss_value = loss(outputs["start_logits"], start_positions) + loss(
            outputs["end_logits"], end_positions
        )
        loss_value = loss_value / GRADIENT_ACC_STEPS
        # with apex.amp.scale_loss(loss_value, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss_value.backward()

        if steps % GRADIENT_ACC_STEPS == 0:
            optimizer.step()
            grads = gradfilter_ema(model, grads=grads)
        #
        # optimizer.step()
        train_loss += loss_value.item()
        steps += 1
    print(f"Epoch {epoch} train loss: {train_loss / len(train_loader)}")
    return grads


def eval_model(model_name = 'bert', epoch=10):
    task = 'nonbinary'
    split_n = 0
    train, test_stratified, test_n_rules_out, test_n_communities_out = read_split(task=task, split_n=split_n)

    max_rules = max(max(i['community']['rules'].keys()) for split in
                    (train, test_stratified, test_n_rules_out, test_n_communities_out) for i in split)
    print(f"len train: {len(train)}, test: {len(test_stratified)}")
    print(f"data looks like this: \n{train[0]}")

    custom_tokens = True
    custom_rule_tokens = False
    skip_numbers = False


    if model_name == 'roberta':
        model = XLMRobertaForQuestionAnswering.from_pretrained(f'../../models/huggingface/robertaqa_onsplit_{epoch}')
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", cache_dir='../../models/huggingface')
    else:
        model = BertForQuestionAnswering.from_pretrained(f'../../models/huggingface/bertqa_onsplit_{epoch}')
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir='../../models/huggingface')
    if custom_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[BOR]", "[EOR]", "[BOC]", "[EOC]", "[BOQ]", "[EOQ]"]}
        )
    if custom_rule_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"[RULE{rule_n}]" for rule_n in
                                           range(max_rules + 1)]}
        )
    model.resize_token_embeddings(len(tokenizer))
    batch_size = 48
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for test_name, test_set in (('test_stratified', test_stratified), ('test_n_rules_out', test_n_rules_out),
                                ('test_n_communities_out', test_n_communities_out)):
        print(f"testing {test_name}, len: {len(test_set)}")
        print("label frequency:")
        print('\n'.join(map(lambda x: f"{x[0]}: {x[1]}", sorted(Counter(i['applied_rule_n'] for i in test_set).items()))))
        dataset, formatted_eval_dataset = prepare_data(data=test_set, tokenizer=tokenizer,
                                                       custom_rule_tokens=custom_rule_tokens,
                                                       custom_tokens=custom_tokens, skip_numbers=skip_numbers,
                                                       model_name=model_name)

        results_by_start, results_by_end = eval_step(model, dataset, formatted_eval_dataset, tokenizer, device,
                                                     batch_size)
        save_results(results_by_start, results_by_end, fname=f'{model_name}qa_{test_name}.json')


def eval_step(model, dataset, formatted_eval_dataset, tokenizer, device, batch_size=16):
    task_evaluator = evaluator("question-answering")

    def compute_metrics(pred):
        eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            data=formatted_eval_dataset,
            metric="squad",
            strategy="simple",
            # n_resamples=9999,
            tokenizer=tokenizer,
            squad_v2_format=False,
        )
        return eval_results

    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss = torch.nn.CrossEntropyLoss()

    # Eval Loop
    model = model.to(device)
    model.eval()
    loss_value = 0
    predicted_start_logits = list()
    predicted_end_logits = list()
    true_start_positions = list()
    true_end_positions = list()
    true_labels = list()

    predicted_starts_within_gt = list()
    predicted_ends_within_gt = list()
    predicted_predicted_rules_by_start = list()
    predicted_predicted_rules_by_end = list()

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss_value += loss(outputs["start_logits"], start_positions) + loss(
                outputs["end_logits"], end_positions
            )
            predicted_start_logits.extend(outputs['start_logits'].argmax(1).tolist())
            predicted_end_logits.extend(outputs['end_logits'].argmax(1).tolist())
            true_end_positions.extend(batch['end_positions'].tolist())
            true_start_positions.extend(batch['start_positions'].tolist())
            true_labels.extend(batch['answer_rule_number'].tolist())

            starts_within_gt, ends_within_gt, predicted_rules_by_start, predicted_rules_by_end = extract_predictions(
                outputs, batch)
            predicted_starts_within_gt.extend(starts_within_gt)
            predicted_ends_within_gt.extend(ends_within_gt)
            predicted_predicted_rules_by_start.extend(predicted_rules_by_start)
            predicted_predicted_rules_by_end.extend(predicted_rules_by_end)

    print(f"frac starts within boundary {np.average(predicted_starts_within_gt):.2f}")
    print(f"frac ends within boundary {np.average(predicted_ends_within_gt):.2f}")

    preds = predicted_predicted_rules_by_start
    labels = true_labels
    print_classification_metrics(labels, preds)
    # metrics_by_start = classification_metrics(labels, preds)
    results_by_start = {'predictions': [int(i) for i in preds], 'labels': [int(i) for i in labels]}
    preds = predicted_predicted_rules_by_end
    labels = true_labels
    print_classification_metrics(labels, preds)
    # metrics_by_end = classification_metrics(labels, preds)
    results_by_end = {'predictions': [int(i) for i in preds], 'labels': [int(i) for i in labels]}
    print(f"Eval loss: {loss_value.item() / len(eval_loader)}")

    # eval_results = compute_metrics(outputs)
    # print(eval_results)
    return results_by_start, results_by_end


def save_results(results_by_start, results_by_end, fname='bertqa.json'):
    results_folder = '../../reports/results/'
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, fname), 'w+') as f:
        json.dump({'results_by_start': results_by_start, 'results_by_end': results_by_end}, f)


def prepare_data(data, tokenizer, custom_tokens=True, custom_rule_tokens=False, skip_numbers=False, augment=False,
                 only_unsafe=False, shuffle_rules=True, permutate_numbers=True, exclude_rules=True,
                 n_replicas_per_strategy=1, model_name='bert', external_datasets = False):
    if not augment:
        qa_ = prep_for_qa(data)
    else:
        qa_ = augment_qa(prep_for_qa(data), only_unsafe=only_unsafe, shuffle_rules=shuffle_rules,
                         permutate_numbers=permutate_numbers, exclude_rules=exclude_rules,
                         n_replicas_per_strategy=n_replicas_per_strategy)
        if external_datasets:
            qa_+=prep_for_qa(random.sample(load_aegis(), 5000))
            qa_+=prep_for_qa(random.sample(load_normvio(), 5000))

    qa = list(map(partial(data_to_qa, custom_tokens=custom_tokens, custom_rule_tokens=custom_rule_tokens,
                          skip_numbers=skip_numbers, comment_in_question=(model_name == 'roberta')), qa_))

    # minimize data to move around
    # print(qa[0])
    comms = set(i['entry']['community']['actor_id'] for i in qa)
    comm_dict = {j: str(i) for i, j in enumerate(sorted(comms))}
    for idx, q in enumerate(qa):
        q['id'] = str(idx)
        # q['community_id'] = str(comm_dict[q['entry']['community']['actor_id']])
        q.pop('entry')
    # print(qa[0])
    # sys.exit()
    dataset_qa = datasets.Dataset.from_list(qa)

    # print(tokenizer, model_name)
    # print(qa[0])
    tokenized_dataset = dataset_qa.map(
        partial(tokenize, tokenizer=tokenizer, truncation="only_second") if model_name == 'bert' else partial(tokenize, tokenizer=tokenizer,
                                                                                    truncation=True),
        batched=True,
        remove_columns=dataset_qa.column_names,
        num_proc=4,
    )

    formatted_eval_dataset = dataset_qa.map(format_answers)
    dataset = tokenized_dataset.with_format("torch")
    return dataset, formatted_eval_dataset


if __name__ == '__main__':
    model_name='bert'
    # model_name='roberta'
    main(model_name=model_name, external_datasets=False)
    eval_model(model_name=model_name, epoch=9)

