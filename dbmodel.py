import torch
from transformers import DistilBertForTokenClassification
from transformers import Trainer, TrainingArguments,  AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset as HFDataset
from nltk.corpus import wordnet
import random
import re
import numpy as np
from sklearn.model_selection import train_test_split
import random 
import nltk
from nltk.corpus import wordnet
from train_data import UTRAIN_DATA

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report



label_map = {
    'O': 0,
    'B-TASK': 1,
    'I-TASK': 2,
    'B-DEADLINE': 3,
    'I-DEADLINE': 4,
    'B-RECIPIENT': 5,
    'I-RECIPIENT': 6,
}
id2label = {v: k for k, v in label_map.items()}


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def augment_data(examples, num_augmented_examples=2):
    augmented_data = []
    for text, annotations in examples:
        # Original example
        augmented_data.append((text, annotations))
        
        for _ in range(num_augmented_examples):
            
            # Entity swapping
            new_text, new_annotations = entity_swapping(text, annotations)
            augmented_data.append((new_text, new_annotations))
            
            new_text, new_annotations = entity_replacement(text, annotations)
            augmented_data.append((new_text, new_annotations))
    
    return augmented_data

def entity_replacement(text, annotations):
    recipients = ["CEO","Manager","HR department","development team","CFO", "CTO", "sales department","team", "department", "Aarav", "Bhavna", "Chirag", "Deepa", "Esha", "Farhan", "Gauri", "Harish", "Isha", "Jay", "Kiran", "Lakshmi", "Manish", "Nikita", "Omkar", "Pallavi", "Rakesh", "Sonal", "Tarun", "Usha", "Vikas", "Yamini", "Zara","Akhil","Arun","Manoj","Nikhil","Ajay","Hari","Anjali","Ravi","Neha","Karan","Ritu","Diya"]
    tasks = ["draft the new policy document", "send a mail","send an email","send the report","submit the report","complete the market analysis", "prepare the project timeline", "review the client feedback", "update the inventory records", "finalize the event budget", "send the project summary", "organize the client meeting", "complete the software testing", "review the annual performance", 
             "Submit the final draft of the strategic plan ","Draft the new policy document","Send the meeting minutes","finalize the product design", "update the client on the project progress", "submit the final project plan", "prepare the quarterly budget", "organize the team workshop", "review the new project proposal", "finalize the marketing campaign", "prepare the annual financial report", "send the updated project timeline", "review the sales performance", "finalize the event logistics", "update the client on the project status", "prepare the quarterly performance report", "organize the annual team meeting"]
    deadlines = ["agreed date","by 5th January", "by 10th February", "by 15th March", "by 20th April", "by 25th May", "by 30th June", "by 5th July", "by 10th August", "by 15th September", "by 20th October", "by 25th November", "by 30th December", "before monday", "before tuesday", "before next week", "before next month", "before next year", "before tomorrow", "on July 12th before 12pm", "on friday before 4pm", "on monday before 10am", "on thursday before 2pm","after the holidays","after the weekend","after the week", "before the week ends"]

    new_text = text
    new_entities = []
    offset = 0

    for start, end, label in sorted(annotations['entities'], key=lambda x: x[0]):
        if label == "RECIPIENT":
            new_entity = random.choice(recipients)
        elif label == "TASK":
            new_entity = random.choice(tasks)
        elif label == "DEADLINE":
            new_entity = random.choice(deadlines)
        else:
            new_entity = text[start:end]

        new_text = new_text[:start+offset] + new_entity + new_text[end+offset:]
        new_start = start + offset
        new_end = new_start + len(new_entity)
        new_entities.append((new_start, new_end, label))
        offset += len(new_entity) - (end - start)

    return new_text, {'entities': new_entities}


def entity_swapping(text, annotations):
    words = text.split()
    entities = annotations['entities']
    
    if len(entities) < 2:
        return text, annotations
    
    entity1, entity2 = random.sample(entities, 2)
    start1, end1, label1 = entity1
    start2, end2, label2 = entity2
    
    # Swap entities
    words[start1:end1], words[start2:end2] = words[start2:end2], words[start1:end1]
    
    # Update entity positions
    len1 = end1 - start1
    len2 = end2 - start2
    
    new_entities = []
    for start, end, label in entities:
        if (start, end, label) == entity1:
            new_entities.append((start2, start2 + len1, label))
        elif (start, end, label) == entity2:
            new_entities.append((start1, start1 + len2, label))
        else:
            new_entities.append((start, end, label))
    
    new_text = ' '.join(words)
    new_annotations = {'entities': new_entities}
    
    return new_text, new_annotations

TRAIN_DATA = augment_data(UTRAIN_DATA)
train_data, val_data = train_test_split(TRAIN_DATA, test_size=0.2, random_state=42)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, return_tensors="pt")
model = AutoModelForTokenClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=len(label_map),
    id2label=id2label,
    label2id=label_map
)
def preprocess_data(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,  # Adjust this based on your data
        is_split_into_words=False
    )

    labels = []
    for i, annotations in enumerate(examples["entities"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(0)  # O (Outside) label
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        # Now, let's add the actual labels
        for annotation in annotations:
            start, end, label = annotation['start'], annotation['end'], annotation['label']
            start_token = tokenized_inputs.char_to_token(i, start)
            end_token = tokenized_inputs.char_to_token(i, end - 1)
            if start_token is None or end_token is None:
                continue
            for token in range(start_token, end_token + 1):
                if token >= len(label_ids):
                    break
                if token == start_token:
                    label_ids[token] = label_map[f"B-{label}"]
                else:
                    label_ids[token] = label_map[f"I-{label}"]

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    
    # Add per-class metrics
    report = classification_report(true_labels, true_predictions, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Skip 'accuracy' which is not a dict
            for metric, value in metrics.items():
                results[f"{label}_{metric}"] = value

    return results

label_list = list(label_map.keys())
# Process the training data
def format_entities(entities):
    return [{'start': start, 'end': end, 'label': label} for start, end, label in entities]

train_dataset = HFDataset.from_dict({
    "text": [example[0] for example in TRAIN_DATA],
    "entities": [format_entities(example[1]['entities']) for example in TRAIN_DATA]
})

# If you have a separate validation set, create it similarly
val_dataset = HFDataset.from_dict({
    "text": [example[0] for example in val_data],
    "entities": [format_entities(example[1]['entities']) for example in val_data]
})

# Apply preprocessing
tokenized_train_dataset = train_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Running tokenizer on train dataset",
)

tokenized_val_dataset = val_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Running tokenizer on validation dataset",
)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,  # If you have a validation set
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
try:
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    print("Trainer state:")
    print(trainer.state)
    print("\nModel:")
    print(model)
    print("\nDataset sample:")
  

print("Evaluating the model...")
evaluation_results = trainer.evaluate()
print(f"Evaluation results: {evaluation_results}")
model_path = "./ner_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")





