import json
import os

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

from ruler.features.make_splits import read_split
from ruler.models.utils import classification_metrics, print_classification_metrics


def format_for_dummy(data):
    return pd.DataFrame([(i['content'], i['community']['actor_id'], i['applied_rule_n']) for i in data],
                        columns=['content', 'community', 'applied_rule_n'])


def always_predict_0(X_train, y_train, X_tests):
    dummy_clf = DummyClassifier(strategy="constant", constant=0)
    dummy_clf.fit(X_train, np.zeros_like(y_train))
    return [dummy_clf.predict(X_test) for X_test in X_tests]


def random_predict(X_train, y_train, X_tests):
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_train, y_train)
    return [dummy_clf.predict(X_test) for X_test in X_tests]


def predict_svm(X_train, y_train, X_tests):
    if y_train.nunique() < 2:
        return [np.full(len(X_test), y_train.unique()[0]) for X_test in X_tests]
    pipeline = Pipeline(
        [
            ("vect", HashingVectorizer(n_features=1000)),
            ("clf", LinearSVC()),
        ]
    )
    pipeline.fit(X_train.content, y_train)
    return [pipeline.predict(X_test.content) for X_test in X_tests]


def predict_nb(X_train, y_train, X_tests):
    pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer(max_features=1000)),
            ("clf", ComplementNB()),
        ]
    )
    pipeline.fit(X_train.content, y_train)
    return [pipeline.predict(X_test.content) for X_test in X_tests]


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, label_column, with_community=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_column = label_column
        self.reverse_label_map = dict(enumerate(sorted(dataframe[label_column].unique())))  # make classes contiguous
        self.label_map = {j: i for i, j in self.reverse_label_map.items()}
        self.with_community = with_community

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['content']
        labels = self.label_map[row[self.label_column]]
        if not self.with_community:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        else:
            inputs = self.tokenizer(
                text,
                row.community,
                max_length=self.max_length,
                padding='max_length',
                truncation="only_first",
                return_tensors="pt"
            )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def predict_bert(train_df, test_df, test_n_rules_out_df, test_n_communities_out_df, label_column, max_length=128,
                 batch_size=16, epochs=3,
                 model_name='bert-base-uncased', learning_rate=5e-5, with_community=False):
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='../../models/huggingface')
    train_dataset = CustomDataset(train_df, tokenizer, max_length, label_column, with_community)
    test_dataset = CustomDataset(test_df, tokenizer, max_length, label_column, with_community)
    test_n_rules_out_dataset = CustomDataset(test_n_rules_out_df, tokenizer, max_length, label_column, with_community)
    test_n_communities_out_dataset = CustomDataset(test_n_communities_out_df, tokenizer, max_length, label_column,
                                                   with_community)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_n_rules_out_loader = DataLoader(test_n_rules_out_dataset, batch_size=batch_size, shuffle=False)
    test_n_communities_out_loader = DataLoader(test_n_communities_out_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=train_df[label_column].nunique(),
                                                          cache_dir='../../models/huggingface')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Training loss: {epoch_loss / len(train_loader):.4f}")

    def eval_loop(test_loader):
        # Evaluation loop
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return [train_dataset.reverse_label_map[i] for i in all_preds]

    return [eval_loop(t) for t in (test_loader, test_n_rules_out_loader, test_n_communities_out_loader)]


if __name__ == '__main__':
    task = 'nonbinary'
    split_n = 0
    train, test, test_n_rules_out, test_n_communities_out = read_split(task=task, split_n=split_n)

    train_df, test_df, test_n_rules_out_df, test_n_communities_out_df = format_for_dummy(train), format_for_dummy(
        test), format_for_dummy(test_n_rules_out), format_for_dummy(test_n_communities_out),

    # global baselines
    X_train_global = train_df[[i for i in train_df.columns if i != 'applied_rule_n']]
    y_train_global = train_df['applied_rule_n']
    X_test_global = test_df[[i for i in test_df.columns if i != 'applied_rule_n']]
    y_test_global = test_df['applied_rule_n']
    X_test_n_rules_out_global = test_n_rules_out_df[[i for i in test_n_rules_out_df.columns if i != 'applied_rule_n']]
    y_test_n_rules_out_global = test_n_rules_out_df['applied_rule_n']
    X_test_n_communities_out_global = test_n_communities_out_df[
        [i for i in test_n_communities_out_df.columns if i != 'applied_rule_n']]
    y_test_n_communities_out_global = test_n_communities_out_df['applied_rule_n']

    max_length = 128
    batch_size = 48
    epochs = 10
    learning_rate = 5e-5

    X_tests = [X_test_global, X_test_n_rules_out_global, X_test_n_communities_out_global]
    y_tests = [y_test_global, y_test_n_rules_out_global, y_test_n_communities_out_global]

    results = {
        "Always Predict 0": {
            'predictions': always_predict_0(X_train_global, y_train_global, X_tests),
            'labels': y_tests},
        "Random Predict": {
            'predictions': random_predict(X_train_global, y_train_global, X_tests),
            'labels': y_tests},
        "Naive Bayes": {
            'predictions': predict_nb(X_train_global, y_train_global, X_tests),
            'labels': y_tests},
        "Linear SVM": {
            'predictions': predict_svm(X_train_global, y_train_global, X_tests),
            'labels': y_tests},
        "BERT": {
            'predictions': predict_bert(train_df, test_df, test_n_rules_out_df, test_n_communities_out_df,
                                        'applied_rule_n', max_length=max_length,
                                        batch_size=batch_size, epochs=epochs,
                                        model_name='bert-base-uncased', learning_rate=learning_rate,
                                        with_community=False),
            'labels': y_tests},
    }
    results_folder = '../../reports/results/'
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, 'baselines_global.json'), 'w+') as f:
        json.dump({strategy: {k: [[int(i) for i in vv] for vv in v] for k, v in d.items()} for strategy, d in
                   results.items()}, f)

    # per-community baselines
    results_per_community = {}

    for community in tqdm(train_df['community'].unique(), 'Computing communities'):
        train_group = train_df[train_df['community'] == community]
        test_group = test_df[test_df['community'] == community]

        X_train_community = train_group[[i for i in train_group.columns if i != 'applied_rule_n']]
        y_train_community = train_group['applied_rule_n'].copy()
        X_test_community = test_group[[i for i in test_group.columns if i != 'applied_rule_n']]
        y_test_community = test_group['applied_rule_n'].copy()
        if not len(y_test_community): continue

        X_tests_community = [X_test_community, X_test_n_rules_out_global, X_test_n_communities_out_global]
        y_tests_community = [y_test_community, y_test_n_rules_out_global, y_test_n_communities_out_global]

        results_per_community[community] = {
            "Always Predict 0": {"predictions":
                                     always_predict_0(X_train_community, y_train_community, X_tests_community), 'labels': y_tests_community},
            "Random Predict": {
                "predictions": random_predict(X_train_community, y_train_community, X_tests_community),
                'labels': y_tests_community},
            "Naive Bayes": {
                'predictions': predict_nb(X_train_community, y_train_community, X_tests_community),
                'labels': y_tests_community},
            "Linear SVM": {
                'predictions': predict_svm(X_train_community, y_train_community, X_tests_community),
                'labels': y_tests_community},
        }

    predictions_bert = predict_bert(train_df, test_df, test_n_rules_out_df, test_n_communities_out_df, 'applied_rule_n', max_length=max_length, batch_size=batch_size,
                                    epochs=epochs,
                                    model_name='bert-base-uncased', learning_rate=learning_rate, with_community=True)

    df_bert = pd.DataFrame(dict(preds=predictions_bert[0], community=test_df.community, labels=test_df['applied_rule_n']))
    for community, g in df_bert.groupby('community'):
        if community not in results_per_community: continue
        results_per_community[community]['BERT'] = {'predictions': [g.preds.tolist()]+predictions_bert[1:],
                                                    'labels': [g.labels.tolist()]+y_tests_community[1:]}

    with open(os.path.join(results_folder, 'baselines_per_community.json'), 'w+') as f:
        json.dump(
            {community: {strategy: {k: [[int(i) for i in vv] for vv in v] for k, v in d.items()} for strategy, d in results_.items()}
             for community, results_ in results_per_community.items()}, f)

    # display results
    print("Global Results:")
    for baseline in results:
        print(baseline)
        for labels, preds, test_name in zip(results[baseline]['labels'], results[baseline]['predictions'], ('stratified', 'leave n rules out' ,'leave n communities out')):
            print(f'results for {test_name}')
            print_classification_metrics(labels=labels, preds=preds)

    print("\nAggregated Per-Community Results:")
    for baseline in results:
        print(baseline)
        for idx, test_name in enumerate(('stratified', 'leave n rules out' ,'leave n communities out')):
            labels = np.concat([results_per_community[community][baseline]['labels'][idx] for community in
                                sorted(results_per_community.keys())])
            predictions = np.concat([results_per_community[community][baseline]['predictions'][idx] for community in
                                     sorted(results_per_community.keys())])
            print_classification_metrics(labels=labels, preds=predictions)
