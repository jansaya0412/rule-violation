from collections import defaultdict
import random
import datasets
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, InputExample
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from ruler.features.make_splits import read_split
from sentence_transformers import losses

from ruler.models.utils import print_classification_metrics

# dataset construction strategy
# sample community
# partition appliced rules
# random assign negatives

if __name__ == '__main__':
    task = 'nonbinary'
    split_n = 0
    train, test, test_n_rules_out, test_n_communities_out = read_split(task=task, split_n=split_n)
    # Load the model
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # print(train[0], device)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    dataset_qa = [InputExample(texts=[i['content'],
                                                       i['community']['rules'][i['applied_rule_n']]] + [
        v for k, v in i['community']['rules'].items() if k != i['applied_rule_n']], label=1) for i in train]

    comments_to_rules = defaultdict(lambda:defaultdict(list))
    for i in train:
        comments_to_rules[i['community']['actor_id']][i['applied_rule_text']].append(i['content'])
    # dataset_qa = list()
    for community, rules in comments_to_rules.items():
        rule_texts = list(rules.keys())
        if len(rule_texts) == 1:
            continue
        for i in range(len(rule_texts)):
            for positive in rules[rule_texts[i]]:
                for _ in range(1):
                    j = random.sample(range(len(rule_texts)-1), 1)[0]
                    if j==i:
                        j+=1
                    dataset_qa.append(InputExample(texts=[rule_texts[i], positive, random.sample(rules[rule_texts[j]], 1)[0]]))

    # dataset_qa = [InputExample(texts=[i['content'],
    #                                                    i['community']['rules'][i['applied_rule_n']]] + [
    #     v for k, v in i['community']['rules'].items() if k != i['applied_rule_n']], label=1) for i in train]
    print(dataset_qa[0], len(dataset_qa))
    # import random
    # dataset_qa = random.sample(dataset_qa, 1000)
    train_loader = DataLoader(dataset_qa, batch_size=1, shuffle=True)
    model.fit(train_objectives=[(train_loader, train_loss)], epochs=10)
    model.save('../../models/huggingface/multi-qa-minilm')

    for test_name, test_set in zip(('stratified', 'leave n rules out' ,'leave n communities out'), (test, test_n_rules_out, test_n_communities_out)):
        print(test_name)
        predictions = list()
        labels = list()
        for entry in tqdm(test_set):

            rule_ns, rule_texts = zip(*entry['community']['rules'].items())
            applied_rule_n = entry['applied_rule_n']

            query = entry['content'] #+ '\n Which of the following rules applies to this message?'

            # Encode query and documents
            query_emb = model.encode(query)
            doc_emb = model.encode(rule_texts)

            # Compute dot score between query and all document embeddings
            scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
            predictions.append(rule_ns[np.argmax(scores)])
            labels.append(applied_rule_n)
        print_classification_metrics(labels=labels, preds=predictions)
