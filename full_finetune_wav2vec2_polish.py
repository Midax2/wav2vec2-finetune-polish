import json
import torch
import torch.onnx
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# --------- 1. Przygotowanie danych ---------
data = [
    {"audio": "data/audio/zako.wav", "text": "projekt na zako"},
    {"audio": "data/audio/dzien_dobry.wav", "text": "dzień dobry"}
]

dataset = Dataset.from_list(data)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# --------- 2. Stworzenie tokenizera dla polskiego ---------
vocab = list("abcdefghijklmnopqrstuvwxyząćęłńóśźż '")
vocab_dict = {v: k for k, v in enumerate(vocab)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

with open("vocab.json", "w") as f:
    json.dump(vocab_dict, f)

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
tokenizer.save_pretrained("tokenizer/")

# --------- 3. Przygotowanie modelu i procesora ---------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", tokenizer=tokenizer)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", vocab_size=len(tokenizer))

# --------- 4. Preprocessing danych ---------
def prepare_batch(batch):
    audio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(batch["text"], return_tensors="pt", padding=True).input_ids
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    batch["labels"] = labels[0]
    return batch

train_dataset = dataset.map(prepare_batch, remove_columns=["audio", "text"])

# --------- 5. Data collator ---------
def data_collator(batch):
    return {
        "input_values": torch.stack([f["input_values"] for f in batch]),
        "attention_mask": torch.stack([f["attention_mask"] for f in batch]),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        ),
    }

# --------- 6. Trening ---------
training_args = TrainingArguments(
    output_dir="./wav2vec2-pl",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_steps=10,
    logging_steps=5,
    learning_rate=1e-4,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=processor,
)

trainer.train()
trainer.save_model("wav2vec2-pl")
processor.save_pretrained("wav2vec2-pl")

# --------- 7. Eksport modelu do ONNX ---------
dummy_input = torch.randn(1, 16000)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={"input_values": {1: "length"}},
)

# --------- 8. Konwersja ONNX -> TensorFlow -> .h5 ---------
onnx_model = onnx.load("model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model")

# TensorFlow load i save do .h5
loaded = tf.keras.models.load_model("tf_model", compile=False)
loaded.save("final_model_polish.h5")

print("Model zapisany jako final_model_polish.h5")
