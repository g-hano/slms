import os
import yaml
import glob
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, normalizers
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Load config
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

HF_WRITE_TOKEN = config.get("HF_WRITE_TOKEN", "")
HF_READ_TOKEN = config.get("HF_READ_TOKEN", "")

tokenizer_config = config.get("data_pipeline", {}).get("tokenizer_training", {})

RAW_DATA_PATH = tokenizer_config.get("raw_data_path", "D:/fineweb2-train/CC-MAIN-2025-26")
OUTPUT_DIR = tokenizer_config.get("output_dir", "./custom_tokenizer")
VOCAB_SIZE = tokenizer_config.get("vocab_size", 32768)
SAMPLE_COUNT = tokenizer_config.get("sample_count", 1000000)

def get_training_corpus():
    data_files = glob.glob(os.path.join(RAW_DATA_PATH, "**/*.parquet"), recursive=True)
    
    if not data_files:
        raise FileNotFoundError(f"Data not found: {RAW_DATA_PATH}")
        
    print(f"📂 Found {len(data_files)} files")
    print(f"🚀 Streaming with first {SAMPLE_COUNT} rows...")

    dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    
    iterator = iter(dataset)
    for i in range(SAMPLE_COUNT):
        try:
            item = next(iterator)
            text = item.get("text", item.get("content", ""))
            yield text
            
        except StopIteration:
            break

def create_chat_template():
    with open("./chat_template.txt", "r", encoding="utf-8") as f:
        return f.read()

def train_tokenizer():
    MAX_LEN = 1024
    MIN_FREQ = 10
    global OUTPUT_DIR
    OUTPUT_DIR = OUTPUT_DIR+f"-{MAX_LEN}"
    tokenizer = Tokenizer(models.BPE())

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])
    
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
    
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        min_frequency=MIN_FREQ,
        show_progress=True
    )
    
    print("\n🔨 Training tokenizer (BPE)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    print("\n💾 Converting to HuggingFace format...")
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=MAX_LEN,
        bos_token="<|im_start|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|endoftext|>",
        additional_special_tokens=special_tokens
    )
    
    hf_tokenizer.chat_template = create_chat_template()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    hf_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Tokenizer Saved: {os.path.abspath(OUTPUT_DIR)}")
    
    return hf_tokenizer

def test_tokenizer(tokenizer):
    print("\n🧪 Testing...")
    text = "Hello! Can you explain how Spiking Neural Networks work?"
    
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("--- Generated Prompt ---")
    print(prompt)
    print("--------------------------")
    
    tokens = tokenizer.encode(prompt)
    print(f"Token Count: {len(tokens)}")
    print(f"Decoded: {tokenizer.decode(tokens)}")

if __name__ == "__main__":
    trained_tokenizer = train_tokenizer()
    test_tokenizer(trained_tokenizer)