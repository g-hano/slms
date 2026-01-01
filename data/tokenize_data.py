import os
import glob
import math
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_ID = "./custom_tokenizer"
INPUT_DIR = "D:/fineweb2-train/CC-MAIN-2025-26"
OUTPUT_BASE = "D:/fineweb2-packed"
FILE_EXTENSION = "*.parquet"

phases = [
    {"name": "phase1_256",  "seq_len": 256,  "range": (0.0, 0.20)}, 
    {"name": "phase2_1024", "seq_len": 1024, "range": (0.20, 0.50)},
    {"name": "phase3_2048", "seq_len": 2048, "range": (0.50, 1.00)}
]

def get_all_files(directory, extension):
    search_path = os.path.join(directory, "**", extension)
    files = sorted(glob.glob(search_path, recursive=True))
    return files

def group_texts(examples, seq_len):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= seq_len:
        total_length = (total_length // seq_len) * seq_len
    result = {
        k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
        for k, t in concatenated_examples.items()
    }
    return result

def process_phase(phase_config, file_list, tokenizer):
    phase_name = phase_config["name"]
    seq_len = phase_config["seq_len"]
    output_dir = os.path.join(OUTPUT_BASE, phase_name)
    
    print(f"\n>>> İşleniyor: {phase_name} | Seq Len: {seq_len} | Dosya Sayısı: {len(file_list)}")
    
    if not file_list:
        print(f"UYARI: {phase_name} için dosya bulunamadı.")
        return

    try:
        data_format = "parquet" if file_list[0].endswith(".parquet") else "json"
        raw_dataset = load_dataset(data_format, data_files=file_list, split="train")

        def tokenize_function(examples):
            text_column = "text" if "text" in examples else "content"
            return tokenizer(examples[text_column], truncation=False, return_special_tokens_mask=True)

        print(f"   Tokenizing ({phase_name})...")
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            num_proc=os.cpu_count()
        )

        print(f"   Packing into chunks of {seq_len}...")
        packed_dataset = tokenized_dataset.map(
            lambda examples: group_texts(examples, seq_len),
            batched=True,
            num_proc=os.cpu_count()
        )

        print(f"   Saving to {output_dir}...")
        packed_dataset.save_to_disk(output_dir)
        print(f"✔ {phase_name} tamamlandı. Kayıt sayısı: {len(packed_dataset)}")

    except Exception as e:
        print(f"HATA: {phase_name} işlenirken bir sorun oluştu: {e}")

def main():
    if not os.path.exists(OUTPUT_BASE):
        os.makedirs(OUTPUT_BASE)
    
    print(f"Tokenizer yükleniyor: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_files = get_all_files(INPUT_DIR, FILE_EXTENSION)
    total_files = len(all_files)
    print(f"Toplam bulunan dosya: {total_files}")

    if total_files == 0:
        print("Hata: Belirtilen klasörde dosya bulunamadı. Uzantıyı veya yolu kontrol et.")
        return

    for phase in phases:
        start_pct, end_pct = phase["range"]
        start_idx = math.floor(start_pct * total_files)
        end_idx = math.floor(end_pct * total_files)
        phase_files = all_files[start_idx:end_idx]
        process_phase(phase, phase_files, tokenizer)

    print("\n--- TÜM İŞLEMLER TAMAMLANDI ---")

if __name__ == "__main__":
    main()