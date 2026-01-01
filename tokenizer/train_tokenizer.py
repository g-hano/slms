import os
import glob
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, normalizers
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

RAW_DATA_PATH = "D:/fineweb2-train/CC-MAIN-2025-26" 
OUTPUT_DIR = "./custom_tokenizer"
VOCAB_SIZE = 32768
SAMPLE_COUNT = 1_000_000

def get_training_corpus():
    data_files = glob.glob(os.path.join(RAW_DATA_PATH, "**/*.parquet"), recursive=True)
    
    if not data_files:
        raise FileNotFoundError(f"Veri bulunamadı: {RAW_DATA_PATH}")
        
    print(f"📂 Bulunan dosya sayısı: {len(data_files)}")
    print(f"🚀 Streaming ile ilk {SAMPLE_COUNT} satır okunuyor...")

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
    return """{%- if messages[0].role != 'system' %}
{{- '<|im_start|>system\nYou are a helpful AI assistant powered by Spiking Neural Networks (SNNs), created by Cihan Yalçın. You are an advanced SpikingLLM designed to provide accurate, helpful, and concise responses in English. You utilize biologically-inspired neuron models for energy-efficient processing.<|im_end|>\n' }}
{%- endif %}

{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    
    {%- if message.role == "system" %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""

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
    
    print("\n🔨 Tokenizer eğitiliyor (BPE)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    print("\n💾 HuggingFace formatına dönüştürülüyor...")
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
    print(f"✅ Tokenizer başarıyla kaydedildi: {os.path.abspath(OUTPUT_DIR)}")
    
    return hf_tokenizer

def test_tokenizer(tokenizer):
    print("\n🧪 Test Ediliyor...")
    text = "Hello! Can you explain how Spiking Neural Networks work?"
    
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("--- Oluşturulan Prompt ---")
    print(prompt)
    print("--------------------------")
    
    tokens = tokenizer.encode(prompt)
    print(f"Token Sayısı: {len(tokens)}")
    print(f"Geri Dönüş (Decode): {tokenizer.decode(tokens)}")

if __name__ == "__main__":
    trained_tokenizer = train_tokenizer()
    test_tokenizer(trained_tokenizer)