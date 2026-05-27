# %%
import json
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

# %%
CLIENT = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="none")

# %%
def call_llm(system_prompt, user_query, response_model=None):
    api_params = {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": 8192,
        "temperature": 0.2,
    }

    if response_model:
        api_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
            },
        }

    response = CLIENT.chat.completions.create(**api_params)
    return response.choices[0].message.content

# %% [markdown]
# ## Dataset creation
# 
# dataset format: 
# 
# [entity, sentence] -> label
# 
# It will predict the label given an entity and the sentence it occurs in. each entity can have different labels when existing in different sentences. from the text we will predict the labels of all entities and merge all the labels of a single entity in a list


with open("classifier_data/dataset.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# %%
def format_input(entity, sentence):
    return f"Entity: {entity} [SEP] Sentence: {sentence}"

# %%
label_list = [
    'Time','Activity','Location','Food','Object','Text',
    'Medical_Concept','Sanskrit_text','Social_Group_&_Role',
    'Phenomenon','Concept','Event','Primordial_Element',
    'Geographical_Feature','Emotions','Body_part','Living_Being',
    'Celestial_Entity','Mythical_Entity'
]

label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

# %%
X = [format_input(d["entity"], d["sentence"]) for d in dataset]
y = [label2id[d["label"]] for d in dataset]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# %%
counts = Counter(y_train)
total = len(y_train)

class_weights = torch.tensor([
    total / counts[i] for i in range(len(label_list))
], dtype=torch.float)

# %%
# %%
encoder = SentenceTransformer("krutrim-ai-labs/Vyakyarth")

X_train_emb = encoder.encode(X_train, convert_to_numpy=True, show_progress_bar=True)
X_val_emb   = encoder.encode(X_val, convert_to_numpy=True, show_progress_bar=True)
X_test_emb  = encoder.encode(X_test, convert_to_numpy=True, show_progress_bar=True)

# %%
# %%
X_train_t = torch.tensor(X_train_emb, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_val_t = torch.tensor(X_val_emb, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

X_test_t = torch.tensor(X_test_emb, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# %%
# %%
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)



# %% [markdown]
# ## FineTuning MuRIL

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

# %%
class MurilDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "google/muril-base-cased",
    num_labels=len(label_list)
)

# %%
from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# %%
for param in model.base_model.parameters():
    param.requires_grad = False

# %% [markdown]
# ### stage 1

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./muril_stage1",
    learning_rate=5e-4,   # higher since encoder frozen
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True
)

# %%
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=MurilDataset(X_train, y_train),
    eval_dataset=MurilDataset(X_val, y_val)
)

trainer.train()

from sklearn.metrics import classification_report

preds = trainer.predict(MurilDataset(X_test, y_test))
y_pred = preds.predictions.argmax(axis=1)

print(classification_report(y_test, y_pred, target_names=label_list))
