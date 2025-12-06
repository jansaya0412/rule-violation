
import copy
import json
import torch
import string
import os
import tqdm
import pandas as pd
import random
import numpy as np

from functools import partial

from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from ruler.features.make_splits import read_split
from ruler.models.grokfast import gradfilter_ema

class MultipleChoiceDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_choices, custom_tokens=True, custom_rule_tokens=False, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_choices = max_choices
        self.custom_tokens = custom_tokens
        self.custom_rule_tokens = custom_rule_tokens
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # data_to_qa:
        comment = entry['content']
        comm = entry.get("community", {})
        comm_name = comm.get("name", "")

        if self.custom_tokens:
            header = f"[BOC]\n{comm_name}\n[EOC]"
        else:
            header = f"Community: {comm_name}"

        rules = "[BOR]\n" if self.custom_tokens else ""
        for rule_n, rule_text in entry['community']['rules']:
            rules += format_rule(rule_n, rule_text, custom_rule_tokens=self.custom_rule_tokens) + "\n"
        rules = rules.rstrip()
        if self.custom_tokens:
            rules = f"{rules}\n[EOR]"

        choices = []
        for letter, _ in entry['community']['rules']:
            if self.custom_tokens:
                context = f"{header}{rules}\n[ANSWER]\nRule {letter}"
            else:
                context = f"{header}{rules}\nAnswer: Rule {letter}"
            choices.append(context)

        if len(choices) < self.max_choices:
            pad_choice = "[BOC]\n[EOC]\n[BOR]\n[EOR]\n[ANSWER]\n[PAD_RULE]"
            choices += [pad_choice] * (self.max_choices - len(choices))

        label = entry.get("applied_rule_n", None)
        if label is None:
            label_idx = -100
        else:
            letters = [r[0] for r in entry['community']['rules']]
            label_idx = letters.index(entry['applied_rule_n'])

        # tokenize:
        enc = self.tokenizer(
            [comment] * len(choices),
            choices,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(label_idx, dtype=torch.long),
            "num_choices": len(choices)
        }

        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
            
        return item


#я тут конвертирую цифры в буквы. надо уточнить у профа, может лучше оставить в виде цифры?
def prep_for_qa(data):
    to_return = list()
    for entry in data:
        rules = list(sorted(entry['community']['rules'].items()))
        entry = copy.deepcopy(entry)
        lettered_rules = [(letter, text) for letter, (_, text) in zip(string.ascii_uppercase, rules)] 
        #we are changing the format of rules from numeric to alphabetic
        #(0, Safe) -> (A, Safe)

        entry['community']['rules'] = lettered_rules
        for letter, text in lettered_rules:
            if text.strip() == entry['applied_rule_text'].strip():
                entry['applied_rule_n'] = letter
        # we formatting "applied_rule_n" from rule text to alphabet
        #applied_rule_text = "Be civil" -> applied_rule_text = "B"

        entry['reason'] = f"Rule {entry['applied_rule_n']}"
        #adding new field "reason": "Rule B"
        to_return.append(entry)
    return to_return

def format_rule(rule_n, rule_text, custom_rule_tokens=False):
    to_return = rule_text
    if rule_n is not None:
        #if None, it returns only text without prefix
        #this is possible if we turn skip_numbers= True to remove some rules' prefixes

        if custom_rule_tokens:
            to_return = f"[RULE {rule_n}] " + to_return
            #if true, then ('B', 'Be civil') → "[RULE B] Be civil"
        else:
            to_return = f"{rule_n}. " + to_return
            #if false, then ('B', 'Be civil') → "B. Be civil"
    return to_return

def train_step(gradient_acc_step, grads, device, model, optimizer, scheduler, loss_fn , train_loader, epoch):
    model.train()
    train_loss = 0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}| Loss: {train_loss / len(train_loader)}"): 
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"]

        outputs = model(**inputs)

        logits = outputs.logits #это уверенность модели для каждого варианта
        loss = loss_fn(logits, batch["labels"])

        loss = loss / gradient_acc_step
        loss.backward()

        steps += 1
        if (steps % gradient_acc_step) == 0:
            grads = gradfilter_ema(model, grads=grads) if grads is not None else gradfilter_ema(model, grads=None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)


        train_loss += loss.item() * gradient_acc_step
        avg_loss = train_loss / steps
        tqdm.tqdm.write(f"step {steps} | loss: {avg_loss:.4f}", end="\r")

    print(f"Epoch {epoch} | train_loss: {avg_loss:.4f}")
    return grads

def print_classification_metrics(labels, preds):
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    print(f"Accuracy Score: {accuracy:.4f}")
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_true=labels, y_pred=preds, zero_division=0))
    confusion = confusion_matrix(y_true=labels, y_pred=preds)
    print(confusion)

    return accuracy, f1, confusion

def eval_step(model, dataset, device, epoch, loss_fn, test_name, batch_size=16, lr=5e-6, wd=0.01):
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    loss_value = 0
    preds = list()
    labels = list()

    evaluation_folder = '/content/drive/MyDrive/rule-violation-main/reports/evaluation'
    #evaluation_folder = '../../reports/evaluation/'
    os.makedirs(evaluation_folder, exist_ok=True)

    for batch in eval_loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
             for k, v in batch.items()}
        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]

            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])

            final_logit = logits.argmax(dim=1)

            loss_value += loss.item()
            labels.extend(batch['labels'].tolist())
            preds.extend(final_logit.tolist())

    accuracy, f1, confusion = print_classification_metrics(labels, preds)
    avg_loss = loss_value / len(eval_loader)
    print(f"Eval loss: {avg_loss}")

    row_metrics = pd.DataFrame([{
        "epoch" : epoch,
        "lr": lr, 
        "wd": wd,
        "test_name" : test_name,
        "accuracy": accuracy,
        "f1": f1,
        "loss": avg_loss
    }])
    pd.DataFrame(confusion).to_csv(os.path.join(evaluation_folder, f"confmat_{test_name}_e{epoch}_lr{lr}_wd{wd}.csv"), index=False)

    results_path = os.path.join(evaluation_folder, "results3_log.csv")
    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        df_all = pd.concat([df_existing, row_metrics], ignore_index=True)
    else:
        df_all = row_metrics

    df_all.to_csv(results_path, index=False)
    print(f"Updated metrics saved to {results_path}")

    return labels, preds

def save_results(labels, preds, fname='bertqa.json'):
    #results_folder = '../../reports/results/'
    results_folder = '/content/drive/MyDrive/rule-violation-main/reports/results'
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, fname), 'w+') as f:
        json.dump({'labels': labels, 'preds': preds}, f)


torch.cuda.empty_cache()
model_name = 'bert'
task = 'nonbinary'
split_n = 0
custom_tokens = True
custom_rule_tokens = True
batch_size = 4
n_replicas_per_strategy = 1
n_epochs = 10
lr = 1e-5
wd = 5e-4

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train, test_stratified, test_n_rules_out, test_n_communities_out = map(prep_for_qa, read_split(task="nonbinary", split_n=0))
all_data = train + test_stratified + test_n_rules_out + test_n_communities_out
max_choices = max(len(all_data[i]['community']['rules']) for i in range(len(all_data))) 

model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased') 
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') 

if custom_tokens:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[BOR]", "[EOR]", "[BOC]", "[EOC]", "[ANSWER]", "[PAD_RULE]"]}
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
dataset = MultipleChoiceDataset(
            data=train,
            tokenizer=tokenizer,
            max_choices=max_choices,
            custom_tokens=custom_tokens,
            custom_rule_tokens=custom_rule_tokens,
            max_length=256)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_sets = dict()
for test_name, test_set in (('test_stratified', test_stratified), ('test_n_rules_out', test_n_rules_out),
                            ('test_n_communities_out', test_n_communities_out)):
    print(f"preparing  {test_name}, len: {len(test_set)}")
    dataset = MultipleChoiceDataset(
            data=test_set,
            tokenizer=tokenizer,
            max_choices=max_choices,
            custom_tokens=custom_tokens,
            custom_rule_tokens=custom_rule_tokens,
            max_length=256)

    test_sets[test_name] = dataset

no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
param_groups = [
    {
        "params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": wd
    },
    {
        "params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0
    },
]

#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
optimizer = torch.optim.AdamW(param_groups, lr=lr)

num_training_steps = len(train_loader) * n_epochs
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
loss_fn = torch.nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

gradient_acc_step = 1
grads = None 

model_save_dir = '/content/drive/MyDrive/rule-violation-main/models/huggingface'
#model_save_dir = "../../models/huggingface"
os.makedirs(model_save_dir, exist_ok=True)

for epoch in range(n_epochs):
    grads = train_step(gradient_acc_step, grads, device, model, optimizer, scheduler, loss_fn, train_loader, epoch)
    model_path = os.path.join(model_save_dir, f'{model_name}_lr{lr}_wd{wd}_epoch{epoch}')
    model.save_pretrained(model_path)
    torch.save(optimizer.state_dict(), os.path.join(model_path, "optimizer.pt"))
    
    for test_name, test_set in test_sets.items():
        print(f"testing {test_name}, len: {len(test_set)}")
        labels, preds = eval_step(
            model, test_set, device,
            epoch, loss_fn, test_name, batch_size, lr = lr, wd = wd
        )
        save_results(labels, preds, fname=f'{model_name}_{test_name}_lr{lr}_wd{wd}_epoch{epoch}.json')


