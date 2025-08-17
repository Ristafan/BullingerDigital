import json
import random
import os
import torch
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def fine_tune(train_path, model_name):
    print("GLiNER FineTuner")

    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    print(data[0])  # Display the first item in the dataset to understand its structure

    random.shuffle(data)
    print('Dataset is shuffled...')

    train_dataset = data[:int(len(data)*0.9)]
    test_dataset = data[int(len(data)*0.9):]

    print('Dataset is splitted...')

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)

    model = GLiNER.from_pretrained(model_name)

    # use it for better performance, it mimics original implementation, but it's less memory efficient
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # Optional: compile model for faster training
    # model.to(device)
    # print("model to device done")

    # calculate number of epochs
    num_steps = 500
    batch_size = 4
    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
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

    print("Start Training...")
    trainer.train()

    # Save the trained model
    trainer.save_model("models/trained_model.pth")


if __name__ == "__main__":
    train_path = "training.json"  # Path to your training data
    model_name = "urchade/gliner_multi-v2.1"

    fine_tune(train_path, model_name)
