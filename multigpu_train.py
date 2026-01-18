import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from tqdm import tqdm

from models.transformer.transformer import RegularLLM, SpikingLLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

checkpoint_dir = "streaming_hf_checkpoints"
phase2_path = os.path.join(checkpoint_dir, "best_phase2_1024")
phase1_path = os.path.join(checkpoint_dir, "best_phase1_256")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Training Script")
    
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training")
    
    parser.add_argument("--muon_lr", type=float, help="Override Muon learning rate")
    parser.add_argument("--adam_lr", type=float, help="Override AdamW learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size globally (for all phases)")
    
    args = parser.parse_args()
    return args

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
    model.eval()
    losses = []
    with torch.no_grad():
        for input_ids in val_batch_list:
            input_ids = input_ids.to(device)
            logits, _ = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.output_dir: config['output_dir'] = args.output_dir
    if args.muon_lr: config['optimizer']['muon_lr'] = args.muon_lr
    if args.adam_lr: config['optimizer']['adam_lr'] = args.adam_lr

    accelerator = Accelerator(
        gradient_accumulation_steps=config['optimizer']['grad_accumulation_steps'],
        log_with="wandb",
        mixed_precision="bf16"
    )
    set_seed(config['seed'])

    if accelerator.is_main_process:
        os.makedirs(config['output_dir'], exist_ok=True)
        accelerator.init_trackers(config['project_name'], config=config)
        print(f"🔧 Config yüklendi: {args.config}")

    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_id'])
    vocab_size = tokenizer.vocab_size
    model_type_ = config['model']['type']
    print(f"Model Type: {model_type_}")

    resume_from = None
    if os.path.isdir(phase2_path):
        resume_from = phase2_path
        print(f"Found Phase 2 checkpoint at {resume_from}. Loading...")
    elif os.path.isdir(phase1_path):
        resume_from = phase1_path
        print(f"Found Phase 1 checkpoint at {resume_from}. Loading...")
    else:
        print("No existing phase1 or phase2 checkpoints found. Initializing new model.")
    
    model_cls = RegularLLM if model_type_ == "regular" else SpikingLLM
    
    model = model_cls(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_kv_heads=config['model']['n_kv_heads'],
        num_layers=config['model']['num_layers'],
        max_seq_len=config['model']['max_seq_len'],
        dtype=torch.bfloat16
    )
    
    if model_type_ == "regular":
         model.lm_head.weight = model.token_emb.weight

    if resume_from:
    try:
        # Look for model.safetensors (preferred) or pytorch_model.bin
        weight_file = None
        if os.path.exists(os.path.join(resume_from, "model.safetensors")):
            from safetensors.torch import load_file
            weight_file = os.path.join(resume_from, "model.safetensors")
            state_dict = load_file(weight_file)
        elif os.path.exists(os.path.join(resume_from, "pytorch_model.bin")):
            weight_file = os.path.join(resume_from, "pytorch_model.bin")
            state_dict = torch.load(weight_file, map_location="cpu")
        
        if weight_file:
            # Load weights into the manually initialized model
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded weights from {weight_file}")
            if missing:
                print(f"Warning: Missing keys in state dict: {missing}")
            if unexpected:
                print(f"Warning: Unexpected keys in state dict: {unexpected}")
        else:
            print(f"Warning: Checkpoint directory {resume_from} exists but contains no recognized weight files.")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    muon_params_list, adam_params_list = get_grouped_params(model)
    
    try:
        optim_muon = torch.optim.Muon(
            muon_params_list, 
            lr=config['optimizer']['muon_lr'], 
            momentum=0.95, nesterov=True, ns_steps=5
        )
    except AttributeError:
        if accelerator.is_main_process: print("⚠️ Native Muon yok, SGD fallback.")
        optim_muon = torch.optim.SGD(muon_params_list, lr=config['optimizer']['muon_lr'], momentum=0.95)

    optim_adam = torch.optim.AdamW(
        adam_params_list, 
        lr=config['optimizer']['adam_lr'], 
        weight_decay=config['optimizer']['weight_decay'], 
        fused=True
    )

    model, optim_muon, optim_adam = accelerator.prepare(model, optim_muon, optim_adam)
    if accelerator.is_main_process and model_type_ == "regular":
        print("🚀 Compiling model...")
        # DDP paketlenmiş modeli compile ediyoruz
        model = torch.compile(model)
        
    global_step = 0
    
    phase_names = list(config['phases'].keys())
    
    for phase_name in phase_names:
        phase_cfg = config['phases'][phase_name]
        repo_id = phase_cfg['repo_id']
        max_steps = phase_cfg['steps']
        batch_size = args.batch_size if args.batch_size else phase_cfg['batch_size']
        
        if accelerator.is_main_process: 
            print(f"\n>>> FAZ: {phase_name} | Repo: {repo_id} | Steps: {max_steps}")

        try:
            dataset = load_dataset(repo_id, split="train", streaming=True)
            dataset = dataset.with_format("torch")
        except Exception as e:
            if accelerator.is_main_process: print(f"❌ Dataset yüklenemedi: {e}")
            continue

        dataset = dataset.shuffle(buffer_size=config['streaming']['shuffle_buffer'], seed=config['seed'])

        ds_iter = iter(dataset)
        val_data = []
        try:
            for _ in range(config['streaming']['val_samples']):
                item = next(ds_iter)
                if "input_ids" in item: val_data.append(item["input_ids"])
        except: pass
            
        val_batches = [torch.stack(val_data[i:i+batch_size]) for i in range(0, len(val_data), batch_size) if val_data[i:i+batch_size]]
        
        train_dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=config['streaming']['num_workers'],
            pin_memory=True
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        
        sched_muon = get_cosine_schedule_with_warmup(optim_muon, int(max_steps * config['optimizer']['warmup_ratio']), max_steps)
        sched_adam = get_cosine_schedule_with_warmup(optim_adam, int(max_steps * config['optimizer']['warmup_ratio']), max_steps)
        sched_muon, sched_adam = accelerator.prepare(sched_muon, sched_adam)

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
            
            if "input_ids" not in batch:
                if "text" in batch: 
                    pass 
                continue

            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                with accelerator.autocast():
                    logits, _ = model(input_ids)
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    loss = nn.CrossEntropyLoss()(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['optimizer']['max_grad_norm'])
                    
                    optim_muon.step()
                    optim_adam.step()
                    sched_muon.step()
                    sched_adam.step()
                    
                    optim_muon.zero_grad(set_to_none=True)
                    optim_adam.zero_grad(set_to_none=True)
                    
                    progress_bar.update(1)
                    global_step += 1
                    steps_in_phase += 1
                    
                    current_loss = loss.item()
                    current_lr = sched_muon.get_last_lr()[0]
                    progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{current_lr:.6f}"})

                    if steps_in_phase % config['logging']['log_steps'] == 0:
                        accelerator.log({"loss": current_loss, "lr": current_lr}, step=global_step)
                    
                    if steps_in_phase % config['logging']['eval_steps'] == 0:
                        val_loss = evaluate_streaming(model, val_batches, accelerator.device)
                        accelerator.log({"val_loss": val_loss}, step=global_step)
                        progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
                        
                        if accelerator.is_main_process:
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                save_path = os.path.join(config['output_dir'], f"best_{phase_name}")
                                
                                # 1. Tam durumu kaydetmek için (Resume için):
                                accelerator.save_state(save_path)
                                
                                # 2. SADECE model ağırlıklarını kaydetmek için (Daha güvenli):
                                # Modeli unwrap yapıyoruz (DDP ve Compile katmanlarını soyuyoruz)
                                unwrapped_model = accelerator.unwrap_model(model)
                                
                                # Eğer modeliniz bir HuggingFace modeli değilse standart torch.save kullanabilirsiniz:
                                # torch.compile yapılmış modelin orijinal halini almak için ._orig_mod kullanılır
                                model_to_save = unwrapped_model._orig_mod if hasattr(unwrapped_model, "_orig_mod") else unwrapped_model
                                
                                torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                                print(f"💾 Model ağırlıkları ve state başarıyla kaydedildi: {save_path}")

        accelerator.wait_for_everyone()
        del train_dataloader, dataset, train_iterator, val_batches
        torch.cuda.empty_cache()
        accelerator.free_memory()

    accelerator.end_training()
    if accelerator.is_main_process: print("🏁 Eğitim Başarıyla Tamamlandı.")

if __name__ == "__main__":
    main()
