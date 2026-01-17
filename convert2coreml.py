import torch
import coremltools as ct
from safetensors.torch import load_file
from src.models.transformer.transformer import SpikingLLM, RegularLLM
import snntorch as snn
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
        "phase2_1024": 3000, 
        "phase3_2048": 1500
    }
    
    batch_sizes = {"phase1_256": 64, "phase2_1024": 16, "phase3_2048": 8}
    grad_accumulation_steps = 4
    
    # Streaming Settings
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
# --- SETTINGS ---
MODEL_PATH = r"C:\Users\Cihan\Desktop\snn\random_models\spiking_model.safetensors" # or regular.safetensors
MODEL_TYPE = "spiking" # or "regular"
OUTPUT_NAME = "spiking"   # Output file name

def coreml_friendly_init_leaky(self):
    return None 
def coreml_friendly_forward(self, input_, mem=None):    
    if mem is None:
        mem = torch.zeros_like(input_)
    print(f"{input_.shape=} | {mem.shape=}")
    mem = self.beta * mem + input_
    
    spk = (mem > self.threshold).float()
    
    mem = mem - (spk * self.threshold)
    
    return spk, mem

print("🔧 snntorch.Leaky CoreML friendly...")
snn.Leaky.forward = coreml_friendly_forward
snn.Leaky.init_leaky = coreml_friendly_init_leaky

print("🏗️ Model skeleton created...")
if MODEL_TYPE == "spiking":
    model = SpikingLLM(
        vocab_size=32768, d_model=config.d_model, n_heads=config.n_heads, n_kv_heads=config.n_kv_heads, num_layers=config.num_layers,
        max_seq_len=2048, num_steps=3, dtype=torch.float32, device="cpu"
    )
else:
    model = RegularLLM(
        vocab_size=32768, d_model=config.d_model, n_heads=config.n_heads, n_kv_heads=config.n_kv_heads, num_layers=config.num_layers,
        max_seq_len=2048, dtype=torch.float32, device="cpu"
    )

print(f"📥 Weights loading: {MODEL_PATH}")
if MODEL_PATH.endswith(".safetensors"):
    state_dict = load_file(MODEL_PATH)
else:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

model.load_state_dict(state_dict, strict=False)
model.eval()
example_input_ids = torch.randint(0, 32000, (1, 1024)).long() # Batch=1, Seq=1024

print("🕵️ PyTorch Tracing started...")
class CoreMLWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):   
        logits, _ = self.model(input_ids)
        return logits

wrapped_model = CoreMLWrapper(model)
traced_model = torch.jit.trace(wrapped_model, example_input_ids, check_trace=False)

print("🍎 Converting to CoreML...")
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=example_input_ids.shape, dtype=int)
    ],
    outputs=[
        ct.TensorType(name="logits")
    ],
    minimum_deployment_target=ct.target.iOS18, # iOS 18+
    compute_units=ct.ComputeUnit.ALL # NPU + GPU + CPU
)

save_path = f"{OUTPUT_NAME}.mlpackage"
mlmodel.save(save_path)
print(f"✅ Success! Model saved to: {save_path}")
print("📱 Drag and drop this file to your Xcode project.")