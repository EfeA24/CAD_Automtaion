import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

SEED = 42
MAX_LENGTH = 128

TR_PATH = r"C:\Users\DELL\Desktop\Automation\Translarion\Datasets\KDE4.en-tr.tr"
EN_PATH = r"C:\Users\DELL\Desktop\Automation\Translarion\Datasets\KDE4.en-tr.en"

CHECKPOINT = "Helsinki-NLP/opus-mt-tr-en"
OUTPUT_DIR = "./TurkishToEnglishModel"
FINAL_MODEL_DIR = "./TurkishToEnglishModel/final"

with open(TR_PATH, encoding="utf-8") as tr_file:
    tr_sentences = tr_file.read().splitlines()

with open(EN_PATH, encoding="utf-8") as en_file:
    en_sentences = en_file.read().splitlines()

assert len(tr_sentences) == len(en_sentences)

data = Dataset.from_dict({
    "id": list(range(len(tr_sentences))),
    "translation": [
        {
            "tr": tr,
            "en": en
        }
        for tr, en in zip(tr_sentences, en_sentences)
    ]
})

train_temp = data.train_test_split(
    test_size=0.2,
    seed=SEED
)

val_test = train_temp["test"].train_test_split(
    test_size=0.5,
    seed=SEED
)

split = {
    "train": train_temp["train"],
    "validation": val_test["train"],
    "test": val_test["test"]
}

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

model = AutoModelForSeq2SeqLM.from_pretrained(
    CHECKPOINT
)


def preprocess_function(examples):
    translations = examples["translation"]

    inputs = [
        item["tr"]
        for item in translations
    ]

    targets = [
        item["en"]
        for item in translations
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_LENGTH,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


tokenized_train = split["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=split["train"].column_names
)

tokenized_validation = split["validation"].map(
    preprocess_function,
    batched=True,
    remove_columns=split["validation"].column_names
)

tokenized_test = split["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=split["test"].column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=30,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    seed=SEED,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

test_results = trainer.evaluate(
    eval_dataset=tokenized_test,
    metric_key_prefix="test"
)

print(f"Final Test Loss: {test_results['test_loss']:.4f}")

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)