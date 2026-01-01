import os
from glob import glob
from datasets import load_dataset

# 1. Yollar
input_dir = "D:/fineweb2/data/data/CC-MAIN-2025-26" 
output_dir = "D:/fineweb2-train/CC-MAIN-2025-26"

os.makedirs(output_dir, exist_ok=True)

# 2. Dosyaları Bul
files = glob(os.path.join(input_dir, "**/*.parquet"), recursive=True)
print(f"Toplam {len(files)} dosya bulundu. İşlem başlıyor...")

cols_to_remove = ['id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count']

# 3. İşlem Döngüsü
for file_path in files:
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    
    # Dosya zaten varsa atla (Yeniden oluşturmak için eski klasörü silmelisiniz!)
    if os.path.exists(output_path):
        print(f"Zaten mevcut, atlanıyor: {file_name}")
        continue

    try:
        # Dataseti yükle
        ds = load_dataset("parquet", data_files=file_path, split="train")
        
        # A) Gereksiz sütunları sil
        existing_cols = [col for col in cols_to_remove if col in ds.column_names]
        ds_clean = ds.remove_columns(existing_cols)
        
        # B) BOZUK VERİ TEMİZLİĞİ (YENİ EKLENEN KISIM)
        # Text sütunu None olanları veya boş string olanları filtrele
        if "text" in ds_clean.column_names:
            initial_count = len(ds_clean)
            
            ds_clean = ds_clean.filter(
                lambda x: x["text"] is not None and len(x["text"].strip()) > 0
            )
            
            final_count = len(ds_clean)
            dropped = initial_count - final_count
            if dropped > 0:
                print(f"  -> {file_name}: {dropped} adet bozuk/boş satır temizlendi.")
        
        # Temizlenmiş veriyi kaydet
        ds_clean.to_parquet(output_path)        
        print(f"Tamamlandı: {file_name}")
        
    except Exception as e:
        print(f"HATA oluştu ({file_name}): {e}")

print("Tüm işlemler bitti.")