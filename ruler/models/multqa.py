import json
from transformers import AutoTokenizer, AutoModelForMultipleChoice, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import copy
import random
import re

class RuleQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        context = example["content"] + " [SEP] " + example["community"]["name"]
        choices = list(example["community"]["rules"].values())
        label = choices.index(example['applied_rule_text'])

        # Tokenize all choices paired with the same context
        encodings = self.tokenizer(
            [context] * len(choices),
            choices,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"],  # shape: (num_choices, seq_len)
            "attention_mask": encodings["attention_mask"],  # shape: (num_choices, seq_len)
            "label": label,
            "num_choices": len(choices)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")}

def shuffled_rules(example, only_unsafe):
    example = copy.deepcopy(example)
    rules = list(example['community']['rules'].values())
    if only_unsafe:
        random.shuffle(rules)
    else:
        safe_rule = rules[-1]          
        unsafe_rules = rules[:-1]    
        random.shuffle(unsafe_rules)
        rules = unsafe_rules + [safe_rule]
    example["community"]["rules"] = {str(i): v for i, v in enumerate(rules)}
    return example

def exclude_rules(example, min_choices=2):
    example  = copy.deepcopy(example)
    rules_items = list(example["community"]["rules"].items())  # list of (k,v)
    rule_texts = [v for (k, v) in rules_items]
    applied = example["applied_rule_text"]

    if applied not in rule_texts:
        return example

    # индексы, которые нельзя удалять (index of applied)
    applied_idx = rule_texts.index(applied)
    candidate_indices = [i for i in range(len(rule_texts)) if i != applied_idx]

    # сколько оставить: минимум min_choices, максимум all
    max_removable = max(0, len(candidate_indices) - (min_choices - 1))
    if max_removable <= 0:
        return example

    # число удаляемых элементов: случайно от 0 до max_removable
    num_remove = random.randint(1, max_removable)
    remove_indices = set(random.sample(candidate_indices, num_remove)) if num_remove > 0 else set()

    new_rule_texts = [rule_texts[i] for i in range(len(rule_texts)) if i not in remove_indices]

    # rebuild dict
    new_rules = {str(i): v for i, v in enumerate(new_rule_texts)}
    example["community"]["rules"] = new_rules
    return example

def permutate_numbers(example):
    example = copy.deepcopy(example)
    text = example["content"]

    def replace_number(m):
        num = m.group(0)
        if len(num) <= 2:
            # небольшое изменение
            delta = random.choice([-1, 0, +1])
            new = str(max(0, int(num) + delta))
            return new
        else:
            # в длинных числах перемешать цифры
            digits = list(num)
            random.shuffle(digits)
            if digits[0] == "0":     # избегаем ведущего нуля   
                digits[0], digits[-1] = digits[-1], digits[0]
            return "".join(digits)

    example["content"] = re.sub(r"\d+", replace_number, text)
    return example

def augment_qa(data,
               only_unsafe=True,
               do_shuffle=True,
               do_permutate_numbers=True,
               do_exclude_rules=True,
               n_replicas_per_strategy=3):
    augmented = []
    seen = set()  # для удаления дубликатов, хранит (content, tuple(rules_values))

    def fingerprint(ex):
        # fingerprint по полю content и порядку правил
        content = ex.get("content", "")
        rules_vals = tuple(ex["community"]["rules"].values())
        return (content, rules_vals)

    for entry in data:
        # добавляем оригинал в seen, чтобы не добавить его снова
        seen.add(fingerprint(entry))

        for _ in range(n_replicas_per_strategy):
            if do_shuffle:
                aug = shuffled_rules(entry, only_unsafe=only_unsafe)
                fp = fingerprint(aug)
                # проверяем, что applied_rule_text остался в вариантах
                if entry["applied_rule_text"] in list(aug["community"]["rules"].values()) and fp not in seen:
                    augmented.append(aug)
                    seen.add(fp)

            if do_permutate_numbers:
                aug = permutate_numbers(entry)
                fp = fingerprint(aug)
                if fp not in seen:
                    augmented.append(aug)
                    seen.add(fp)

            if do_exclude_rules:
                aug = exclude_rules(entry, min_choices=2)
                fp = fingerprint(aug)
                if entry["applied_rule_text"] in list(aug["community"]["rules"].values()) and fp not in seen:
                    augmented.append(aug)
                    seen.add(fp)

    print(f"augment_qa: original={len(data)}, augmented_added={len(augmented)}, total={len(data)+len(augmented)}")
    return data + augmented

#паддит всe инпуты до максимальной длины
#формирует батч правильной формы
class VariableChoiceCollator:
    def __init__(self, max_num_choices):
        self.max_num_choices = max_num_choices

    def __call__(self, features):
        max_choices = self.max_num_choices
        max_len = features[0]["input_ids"].shape[1]  #берем только первый пример, потому что во всех примерах длина = 128

        def pad_tensor(tensor, target_shape):
            pad_size = (target_shape[0] - tensor.shape[0], 0)
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size[0]), value=0)

        input_ids = torch.stack([pad_tensor(f["input_ids"], (max_choices, max_len)) for f in features])
        attention_mask = torch.stack([pad_tensor(f["attention_mask"], (max_choices, max_len)) for f in features])
        labels = torch.tensor([f["label"] for f in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def get_max_num_choices(datasets):
    return max(len(example["community"]["rules"]) for dataset in datasets for example in dataset)

def label_index(example):
        choices = list(example["community"]["rules"].values())
        return choices.index(example["applied_rule_text"])

if __name__ == '__main__':
    random.seed(42)
    os.environ["WANDB_DISABLED"] = "true"

    tests = ['test_n_rules_out.json', 'test_n_communities_out.json', 'test_stratified.json']
    modes = ['shuffle', 'permute', 'exclude', 'all']

    for mode in modes:
        if mode == "shuffle":
            FLAGS = {"do_shuffle": True, "do_permutate_numbers": False, "do_exclude_rules": False}
        elif mode == "permute":
            FLAGS = {"do_shuffle": False, "do_permutate_numbers": True, "do_exclude_rules": False}
        elif mode == "exclude":
            FLAGS = {"do_shuffle": False, "do_permutate_numbers": False, "do_exclude_rules": True}
        else:
            FLAGS = {"do_shuffle": True, "do_permutate_numbers": True, "do_exclude_rules": True}


        print(f"\n\n=== RUN MODE: {mode} | FLAGS: {FLAGS} ===\n")

        for test in tests:
            #interim_path = r'F:\PycharmProjects\ruler\data\interim'
            interim_path = r'/content/drive/MyDrive/rule-violation-main/data/interim'
            split_path = f'{interim_path}/splits/nonbinary/0'

            # Load data
            train_path = f'{split_path}/train.json'
            #eval_path = f'{split_path}/dev.json'
            test_path = f'{split_path}/{test}'
            with open(train_path) as f:
                train_data = json.load(f)
            with open(test_path) as f:
                test_data = json.load(f)
            
            train_data = augment_qa(train_data, only_unsafe=True, do_shuffle=FLAGS["do_shuffle"], do_permutate_numbers=FLAGS["do_permutate_numbers"], do_exclude_rules=FLAGS["do_exclude_rules"], n_replicas_per_strategy=1)
            train_data, eval_data = train_test_split(train_data, test_size=0.15, random_state=42)
            max_num_choices = get_max_num_choices([train_data, eval_data, test_data])

            # Load tokenizer and model
            # model_name = "distilbert-base-uncased"
            # model_name = "bert-large-uncased"
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMultipleChoice.from_pretrained(model_name)

            # print(train_data[0])
            # Prepare datasets
            train_dataset = RuleQADataset(train_data, tokenizer)
            eval_dataset = RuleQADataset(eval_data, tokenizer)

            # print(train_dataset[0])

            # Set training args
            training_args = TrainingArguments(
                output_dir="./results",
                eval_strategy="epoch",
                learning_rate=1e-5,
                save_strategy="no",           
                load_best_model_at_end=False,      
                metric_for_best_model="eval_f1",  
                greater_is_better=True,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                num_train_epochs=5,
                weight_decay=0.03,
                fp16=True,
                logging_dir="./logs",
                logging_steps=10,
            )

            #мы создаем просто обьект с алгоритмом сборки батчей. он внутри по сути пустой и не взаимодействует с никакими данными, пока его не вызовут
            data_collator = VariableChoiceCollator(max_num_choices=max_num_choices)

            # DataLoader:
            # -> вызывает dataset.__getitem__(i)  (токенизация внутри)
            # -> получает features = [sample1, sample2, ...]  (уже tensors)
            # -> вызывает data_collator(features)
            # -> collator делает padding/stack -> batch tensors -> модель

            # Trainer setup
            #вот тут создается train\eval DataLoader
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                data_collator=data_collator
            )

            # Trainer создаёт DataLoader.
            # DataLoader получает collate_fn = data_collator.
            # DataLoader собирает очередной батч:
            # вызывает dataset.__getitem__() несколько раз из train/eval dataset
            # получает список samples:

            # Train
            trainer.train()

            #log_df = pd.DataFrame(trainer.state.log_history)
            #eval_log_df = log_df[[c for c in log_df.columns if c.startswith("eval_") or c == "epoch"]]
            #eval_log_df.to_csv("./results/epoch_5e-6_01_rules.csv", index=False)
            #print("✅ Per-epoch metrics saved to ./results/epoch_metrics_2e-05_0001.csv")
            #%%

            test_dataset = RuleQADataset(test_data, tokenizer)
            preds_output = trainer.predict(test_dataset)
            preds = preds_output.predictions.argmax(axis=-1)
            labels = preds_output.label_ids
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")
            print(accuracy, f1)
            row_metrics = pd.DataFrame([{
                "test": test,
                "do_shuffle" : FLAGS["do_shuffle"],
                "do_permutate_numbers" : FLAGS["do_permutate_numbers"],
                "do_exclude_rules" : FLAGS["do_exclude_rules"],
                "only_unsafe" : True,
                "lr": 1e-05, 
                "wd": 0.03,
                "accuracy": accuracy,
                "f1": f1
            }])
            results_path = "./results/augmentations.csv"
            if os.path.exists(results_path):
                df_existing = pd.read_csv(results_path)
                df_all = pd.concat([df_existing, row_metrics], ignore_index=True)
            else:
                df_all = row_metrics

            df_all.to_csv(results_path, index=False)
            print(f"Updated metrics saved to {results_path}")
