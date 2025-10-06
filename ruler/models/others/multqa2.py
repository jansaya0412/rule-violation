
import copy
import json
import torch
import string
import datasets
import os
import tqdm
import pandas as pd

from functools import partial

from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from ruler.features.make_splits import read_split
from ruler.models.grokfast import gradfilter_ema


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

def data_to_qa(entry, max_choices, custom_tokens=True, custom_rule_tokens=False):

    rules = "[BOR]\n" if custom_tokens else ""

    for rule_n, rule_text in entry['community']['rules']:
        rules += format_rule(rule_n, rule_text, custom_rule_tokens=custom_rule_tokens) + '\n'
    rules = rules.rstrip()  # убираем последнюю \n
    #created list of formatted rules

    if custom_tokens:
        rules = f"{rules}\n[EOR]"
        #added ending token 

    comment = entry['content']
    
    choices = []
    correct_idx = None

    for idx, (letter, _) in enumerate(entry['community']['rules']):
        if custom_tokens:
            context = (
                f"{rules}\n"
                f"[ANSWER]\n"
                f"Rule {letter}"
            )
        #adding content part 

        else:
            context = f"""{rules}\nAnswer: Rule {letter}"""

        choices.append(context) 
        #each element of choices has the list of rules
        #examples["choices"][i] == [
        #"[BOR]\nA. Safe\nB. Be civil\nC. No misinformation\nD. Posts should be memes\nE. No bots, spam or self-promotion\n[EOR]\n[BOC]\nThis wasn't a debate, Clyde...\n[EOC]\n[Answer]\nRule A",
        #"[BOR]\nA. Safe\nB. Be civil\nC. No misinformation\nD. Posts should be memes\nE. No bots, spam or self-promotion\n[EOR]\n[BOC]\nThis wasn't a debate, Clyde...\n[EOC]\n[Answer]\nRule B",
        #"[BOR]\nA. Safe\nB. Be civil\nC. No misinformation\nD. Posts should be memes\nE. No bots, spam or self-promotion\n[EOR]\n[BOC]\nThis wasn't a debate, Clyde...\n[EOC]\n[Answer]\nRule C",
    
        if letter == entry['applied_rule_n']:
            correct_idx = idx
        #looking for correct index
    if correct_idx is None:
        correct_idx = -100  # или скипаем пример


    if len(choices) < max_choices:
        padding_needed = max_choices - len(choices)
        pad_choice = "[BOR]\n[EOR]\n[ANSWER]\n[PAD_RULE]"
        choices += [pad_choice] * padding_needed
    #filling up with paddings if number of choices is not enough
    return {
        "id": entry.get("ap_id", entry.get("modlog_id", "no_id")),
        "comment": comment,
        "choices": choices,
        "labels": correct_idx
    }
    #this is the structure for multiple choice qa.
    #it is the structure that tokenize and DataLoader accepts

def tokenize(examples, tokenizer, truncation='only_second'):  
    #converts text to digits and adapts to tensor of [batch, num_choices, seq_len] form that is need for AutoModelForMultipleChoice

    num_choices = len(examples["choices"][0]) 
    #due to the padding, each list contains the same number of rules. therefore it is enough to take the length of the first element only

    #sentences are question/rules+context+answer pair
    first_sentence = []
    second_sentence = []
    for comment, choices in zip(examples["comment"], examples["choices"]):
        for choice in choices:
            first_sentence.append(comment)   # первое предложение = comment
            second_sentence.append(choice)


    inputs = tokenizer(
        first_sentence,
        second_sentence,
        truncation=truncation,
        padding="max_length",
        max_length=384,
    )
    #calling HuggingFace tokenizer to make digit tensors for the model
    #inputs = {"input_ids":       [[101, 102, 2009, ... , 102, 0, 0, ...],  ... N раз],
    #"attention_mask":  [[1,   1,   1,   ... , 1,   0, 0, ...],   ... N],
    #"token_type_ids":  [[0,   0,   1,   ... , 1,   0, 0, ...],   ... N]  # только для BERT}

    result = {k: [inputs[k][i:i + num_choices] for i in range(0, len(inputs[k]), num_choices)]
              for k in inputs}
    #before: inputs["input_ids"] = [s0A, s0B, s0C, s1A, s1B, s1C]
    #after: result["input_ids"] = [[s0A, s0B, s0C], [s1A, s1B, s1C]]
    
    result["labels"] = examples["labels"]
    return result

def prepare_data(data, tokenizer, max_choices, custom_tokens=True, custom_rule_tokens=False, model_name='bert'):

    qa = list(map(partial(data_to_qa, max_choices = max_choices, custom_tokens=custom_tokens, custom_rule_tokens=custom_rule_tokens), data))

    for idx, q in enumerate(qa):
            q['id'] = str(idx)
    #to guarantee unique ids

    dataset_qa = datasets.Dataset.from_list(qa)
    #converting qa dictionary to Hugging Face Datasets

    tokenized_dataset = dataset_qa.map(
            partial(tokenize, tokenizer=tokenizer, truncation="only_second") if model_name == 'bert'
            else partial(tokenize, tokenizer=tokenizer, truncation=True),
            batched=True,
            remove_columns=dataset_qa.column_names,
            num_proc=4,
        )

    dataset = tokenized_dataset.with_format("torch")
    #making it pytorch formatted, so we can feed it into DataLoader
    return dataset, dataset_qa

def train_step(gradient_acc_step, grads, device, model, optimizer, loss_fn , train_loader, epoch):
    model.train()
    train_loss = 0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}| Loss: {train_loss / len(train_loader)}"): #going through batches
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) 
        logits = outputs.logits #это уверенность модели для каждого варианта
        loss = loss_fn(logits, batch["labels"])

        loss = loss / gradient_acc_step
        loss.backward()

        steps += 1
        if (steps % gradient_acc_step) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #so model doesnt bomb
            optimizer.step()
            grads = gradfilter_ema(model, grads=grads) if grads is not None else gradfilter_ema(model, grads=None)
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

    return accuracy, f1

def eval_step(model, dataset, formatted_eval_dataset, tokenizer, device, epoch, loss_fn, test_name, batch_size=16):
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    loss_value = 0
    preds = list()
    labels = list()

    evaluation_folder = '../../reports/evaluation/'
    os.makedirs(evaluation_folder, exist_ok=True)

    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(outputs.logits, batch['labels'])

            final_logit = logits.argmax(dim=1)

            loss_value += loss.item()
            labels.extend(batch['labels'].tolist())
            preds.extend(final_logit.tolist())

    accuracy, f1 = print_classification_metrics(labels, preds)
    avg_loss = loss_value / len(eval_loader)
    print(f"Eval loss: {avg_loss}")

    row_metrics = pd.DataFrame([{
        "epoch" : epoch,
        "test_name" : test_name,
        "accuracy": accuracy,
        "f1": f1,
        "loss": avg_loss
    }])

    results_path = os.path.join(evaluation_folder, "results2_log.csv")
    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        df_all = pd.concat([df_existing, row_metrics], ignore_index=True)
    else:
        df_all = row_metrics

    df_all.to_csv(results_path, index=False)
    print(f"Updated metrics saved to {results_path}")

    return labels, preds

def save_results(labels, preds, fname='bertqa.json'):
    results_folder = '../../reports/results/'
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, fname), 'w+') as f:
        json.dump({'labels': labels, 'preds': preds}, f)


torch.cuda.empty_cache()
model_name = 'bert'
task = 'nonbinary'
split_n = 0
custom_tokens = True
custom_rule_tokens = True
batch_size = 8
n_replicas_per_strategy = 1
n_epochs = 10

train, test_stratified, test_n_rules_out, test_n_communities_out = map(prep_for_qa, read_split(task="nonbinary", split_n=0))
all_data = train + test_stratified + test_n_rules_out + test_n_communities_out
max_choices = max(len(all_data[i]['community']['rules']) for i in range(len(all_data))) 

model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased') #model baseline
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


model.resize_token_embeddings(len(tokenizer)) #synchronising our model's dictionary with tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"preparing train, len: {len(train)}")
dataset, eval_dataset = prepare_data(train, tokenizer, max_choices, custom_tokens=custom_tokens, custom_rule_tokens=custom_rule_tokens, 
                                     model_name=model_name)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_sets = dict()
for test_name, test_set in (('test_stratified', test_stratified), ('test_n_rules_out', test_n_rules_out),
                            ('test_n_communities_out', test_n_communities_out)):
    print(f"preparing  {test_name}, len: {len(test_set)}")
    dataset, eval_dataset = prepare_data(data=test_set, tokenizer=tokenizer, max_choices = max_choices,
                                        custom_rule_tokens=custom_rule_tokens,
                                        custom_tokens=custom_tokens,
                                        model_name=model_name)
    test_sets[test_name] = dataset, eval_dataset

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

gradient_acc_step = 1
grads = None 

model_save_dir = "../../models/huggingface"
os.makedirs(model_save_dir, exist_ok=True)

for epoch in range(n_epochs):
    grads = train_step(gradient_acc_step, grads, device, model, optimizer,loss_fn, train_loader, epoch)
    model_path = os.path.join(model_save_dir, f'{model_name}qa_onsplit_{epoch}')
    model.save_pretrained(model_path)
    torch.save(optimizer.state_dict(), os.path.join(model_path, "optimizer.pt"))
    
    for test_name, (test_set, eval_dataset) in test_sets.items():
        print(f"testing {test_name}, len: {len(test_set)}")
        labels, preds = eval_step(model, test_set, eval_dataset, tokenizer, device,
                                                        epoch, loss_fn, test_name, batch_size)
        save_results(labels, preds, fname=f'{model_name}qa_{test_name}_{epoch}.json')

