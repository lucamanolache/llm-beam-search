from transformers import AutoModel, AutoTokenizer, pipeline

model_name = "Qwen/Qwen2.5-Math-PRM-7B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()
