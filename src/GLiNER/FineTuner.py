import json
import random
import os
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print("GLiNER FineTuner")

train_path = "data.json"

with open(train_path, "r") as f:
    data = json.load(f)

print('Dataset size:', len(data))

print(data[1])  # Display the first item in the dataset to understand its structure

print(data[1]["tokenized_text"][17])
print(data[1]["tokenized_text"][18])
print(data[1]["tokenized_text"][19])


random.shuffle(data)
print('Dataset is shuffled...')

train_dataset = data[:int(len(data)*0.9)]
test_dataset = data[int(len(data)*0.9):]

print('Dataset is splitted...')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = GLiNER.from_pretrained("urchade/gliner_small")

# use it for better performance, it mimics original implementation but it's less memory efficient
data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

# Optional: compile model for faster training
model.to(device)
print("done")

# calculate number of epochs
num_steps = 500
batch_size = 8
data_size = len(train_dataset)
num_batches = data_size // batch_size
num_epochs = max(1, num_steps // num_batches)

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear", #cosine
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    dataloader_num_workers=0,
    use_cpu=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)

trainer.train()

trained_model = GLiNER.from_pretrained("models/checkpoint-100", load_tokenizer=True)

text = "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time."

# Labels for entity prediction
labels = ["Person", "Award"] # for v2.1 use capital case for better performance

# Perform entity prediction
entities = trained_model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
