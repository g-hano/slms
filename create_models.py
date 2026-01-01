import torch
import os
from src.models.transformer.transformer import RegularLLM, SpikingLLM
from safetensors.torch import save_model
class TrainingConfig:
    DATA_PATH = "D:/fineweb2-packed"
    OUTPUT_DIR = "./random_models"
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

config = TrainingConfig()

# Save models locally
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

regular = RegularLLM(32768, config.d_model, config.n_heads, config.n_kv_heads, config.num_layers, dtype=torch.bfloat16)
regular_path = os.path.join(config.OUTPUT_DIR, "regular_model.pt")
torch.save(regular.state_dict(), regular_path)
regular_path = os.path.join(config.OUTPUT_DIR, "regular_model.safetensors")
save_model(regular, regular_path)
print(f"Regular model saved to {regular_path}")
del regular

spiking = SpikingLLM(32768, config.d_model, config.n_heads, config.n_kv_heads, config.num_layers, dtype=torch.bfloat16)
spiking_path = os.path.join(config.OUTPUT_DIR, "spiking_model.pt")
torch.save(spiking.state_dict(), spiking_path)
spiking_path = os.path.join(config.OUTPUT_DIR, "spiking_model.safetensors")
save_model(spiking, spiking_path)
print(f"Spiking model saved to {spiking_path}")