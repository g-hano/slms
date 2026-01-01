import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from tqdm import tqdm
import math

# Senin modellerin
from models.transformer.transformer import RegularLLM, SpikingLLM

# --- OPTIMIZATION FLAGS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class TrainingConfig:
    DATA_PATH = "D:/fineweb2-packed"
    OUTPUT_DIR = "./checkpoints_streaming"
    MODEL_ID = "./custom_tokenizer-1024"
    MODEL_TYPE = "regular"
    
    # Model
    d_model = 768
    n_heads = 12
    n_kv_heads = 4
    num_layers = 12
    vocab_size = None

    phase_steps = {
        "phase1_256": 5000, 
        #"phase2_1024": 3000, 
        #"phase3_2048": 1500
    }
    
    batch_sizes = {"phase1_256": 64, "phase2_1024": 16, "phase3_2048": 8}
    grad_accumulation_steps = 4
    
    # Streaming Ayarları
    shuffle_buffer = 10000
    val_samples = 200
    
    # Optimizer
    muon_lr = 0.02
    adam_lr = 0.0006 
    weight_decay = 0.01
    warmup_ratio = 0.05
    max_grad_norm = 1.0
    
    save_steps = 1000
    log_steps = 10
    eval_steps = 200
    
    phases = ["phase1_256", "phase2_1024", "phase3_2048"]

def get_grouped_params(model):
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 2 and "token_emb" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params

def evaluate_streaming(model, val_batch_list, device):
    """
    Streaming modunda sabit bir validation listesi üzerinden test yapar.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for input_ids in val_batch_list:
            input_ids = input_ids.to(device)
            logits, _ = model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            losses.append(loss.item())
            
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

def main():
    config = TrainingConfig()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accumulation_steps,
        log_with="wandb",
        mixed_precision="bf16"
    )
    
    if accelerator.is_main_process:
        if not os.path.exists(config.OUTPUT_DIR): os.makedirs(config.OUTPUT_DIR)
        accelerator.init_trackers("streaming_muon", config={k:v for k,v in TrainingConfig.__dict__.items() if not k.startswith('__')})

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    vocab_size = tokenizer.vocab_size
    
    if config.MODEL_TYPE == "regular":
        model = RegularLLM(vocab_size, config.d_model, config.n_heads, config.n_kv_heads, config.num_layers, dtype=torch.bfloat16)
    else:
        model = SpikingLLM(vocab_size, config.d_model, config.n_heads, config.n_kv_heads, config.num_layers, dtype=torch.bfloat16)
    print("Model Parametre Sayısı:", model.get_num_params())
    
    muon_params_list, adam_params_list = get_grouped_params(model)
    
    try:
        optim_muon = torch.optim.Muon(muon_params_list, lr=config.muon_lr, momentum=0.95, nesterov=True, ns_steps=5)
    except AttributeError:
        if accelerator.is_main_process: print("UYARI: torch.optim.Muon bulunamadı! SGD fallback kullanılıyor.")
        optim_muon = torch.optim.SGD(muon_params_list, lr=config.muon_lr, momentum=0.95)

    optim_adam = torch.optim.AdamW(adam_params_list, lr=config.adam_lr, weight_decay=config.weight_decay, fused=True)

    model, optim_muon, optim_adam = accelerator.prepare(model, optim_muon, optim_adam)

    global_step = 0
    
    for phase_name in config.phases:
        phase_path = os.path.join(config.DATA_PATH, phase_name)
        if not os.path.exists(phase_path): continue
        
        arrow_files = glob.glob(f"{phase_path}/**/data-*.arrow", recursive=True)
        
        if not arrow_files:
            if accelerator.is_main_process: 
                print(f"UYARI: {phase_name} içinde 'data-*.arrow' dosyası bulunamadı!")
            continue
            
        if accelerator.is_main_process: 
            print(f"\n>>> FAZ: {phase_name} | Streaming Mode | Dosya Sayısı: {len(arrow_files)}")
            
        dataset = load_dataset("arrow", data_files=arrow_files, streaming=True, split="train")
        
        dataset = dataset.with_format("torch")
        
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer)
        
        batch_size = config.batch_sizes.get(phase_name, 8)
        max_steps = config.phase_steps.get(phase_name, 1000)
        
        ds_iter = iter(dataset)
        val_data = []

        if accelerator.is_main_process: print("Validasyon seti ayrılıyor...")
        try:
            for _ in range(config.val_samples):
                item = next(ds_iter)
                if "input_ids" in item:
                    val_data.append(item["input_ids"])
        except StopIteration:
            print("Veri seti validasyon için yetersiz!")
            continue
            
        val_batches = []
        for i in range(0, len(val_data), batch_size):
            batch_list = val_data[i:i+batch_size]
            if batch_list:
                val_batches.append(torch.stack(batch_list))
                
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2,
        )
        
        train_dataloader, optim_muon, optim_adam = accelerator.prepare(train_dataloader, optim_muon, optim_adam)
        
        # Scheduler
        sched_muon = get_cosine_schedule_with_warmup(optim_muon, int(max_steps*config.warmup_ratio), max_steps)
        sched_adam = get_cosine_schedule_with_warmup(optim_adam, int(max_steps*config.warmup_ratio), max_steps)
        
        model.train()
        progress_bar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process, desc=f"{phase_name}")
        
        train_iterator = iter(train_dataloader)
        steps_in_phase = 0
        best_val_loss = float('inf')
        
        while steps_in_phase < max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
                
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                
                logits, _ = model(input_ids)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = nn.CrossEntropyLoss()(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optim_muon.step()
                optim_adam.step()
                sched_muon.step()
                sched_adam.step()
                
                optim_muon.zero_grad(set_to_none=True)
                optim_adam.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                steps_in_phase += 1
                
                current_loss = loss.item()
                current_lr = sched_muon.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}", 
                    "lr": f"{current_lr:.6f}"
                })

                if steps_in_phase % config.log_steps == 0:
                    accelerator.log({"loss": current_loss, "lr": current_lr}, step=global_step)
                
                if steps_in_phase % config.eval_steps == 0:
                    val_loss = evaluate_streaming(model, val_batches, accelerator.device)
                    accelerator.log({"val_loss": val_loss}, step=global_step)
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}", 
                        "val_loss": f"{val_loss:.4f}",
                        "lr": f"{current_lr:.6f}"
                    })

                    if accelerator.is_main_process:
                        print(f" [Step {steps_in_phase}] Val: {val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            accelerator.save_state(os.path.join(config.OUTPUT_DIR, f"best_{phase_name}"))

        del train_dataloader, dataset, train_iterator, val_batches
        torch.cuda.empty_cache()
        accelerator.free_memory()

    accelerator.end_training()
    if accelerator.is_main_process: print("Eğitim Tamamlandı.")

if __name__ == "__main__":
    main()