import torch
import gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset as HFDataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# 1. PURGE CORRUPTED MODEL FROM GPU MEMORY

try:
    del llm_model
    del peft_model
    del trainer
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()


# 2. RELOAD A CLEAN 4-BIT BASE MODEL

print("Loading clean base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
llm_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", 
    quantization_config=bnb_config, 
    device_map="auto"
)
llm_model.config.use_cache = False # Required for gradient checkpointing


# 3. FORMAT THE DATASET

def format_qa_to_chat(row):
    ctx = combined_context(row['prompt'])    
    user_msg = (
        f"Context:\n{ctx}\n\n"
        f"Question: {row['prompt']}\n"
        f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\n"
        f"D: {row['D']}\nE: {row['E']}\n\n"
        "Answer with only the single best option letter."
    )
    
    messages = [
        {"role": "system", "content": "You are an expert scientist answering multiple-choice questions precisely."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": row['answer']}
    ]
    
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

hf_dataset = HFDataset.from_pandas(fit_df)
hf_formatted_dataset = hf_dataset.map(format_qa_to_chat, remove_columns=hf_dataset.column_names)


# 4. DEFINE LORA CONFIG

peft_config = LoraConfig(
    r=16,               
    lora_alpha=32,      
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,  
    bias="none",        
    task_type="CAUSAL_LM" 
)


# 5. TRAINER CONFIGURATION & EXECUTION

training_args = SFTConfig(
    output_dir="./qwen_qlora_output",
    per_device_train_batch_size=2,      
    gradient_accumulation_steps=8,      
    learning_rate=5e-4,                 
    num_train_epochs=1,                 
    optim="paged_adamw_8bit",           
    fp16=True,                          
    logging_steps=10,
    save_strategy="epoch",
    dataset_text_field="text",          
    max_length=512,
    packing=True,
    
    gradient_checkpointing=True,  # Let the Trainer handle memory checkpointing safely!
    report_to="wandb"
)


trainer = SFTTrainer(
    model=llm_model,               
    train_dataset=hf_formatted_dataset,
    args=training_args,
    peft_config=peft_config,       
)

print("Starting QLoRA training...")
trainer.train()


from peft import PeftModel

# 1. Save the tiny adapter weights
trainer.model.save_pretrained("./qwen_trained_adapter")
print("Adapter saved successfully!")

# 2. To use it in your ensemble, wrap the existing base llm_model
fine_tuned_llm = PeftModel.from_pretrained(llm_model, "./qwen_trained_adapter")
fine_tuned_llm.eval()

# Now, update your existing llm_score_row function to use fine_tuned_llm instead of llm_model
@torch.no_grad()
def llm_score_row_tuned(prompt, options_text, context):
    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {prompt}\n"
        f"A: {options_text['A']}\nB: {options_text['B']}\nC: {options_text['C']}\n"
        f"D: {options_text['D']}\nE: {options_text['E']}\n\n"
        "Answer with only the single best option letter."
    )
    messages = [
        {"role": "system", "content": "You are an expert scientist answering multiple-choice questions precisely."},
        {"role": "user", "content": user_msg},
    ]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(text, return_tensors="pt").to(fine_tuned_llm.device)
    
    # Generate using the FINE-TUNED model
    out = fine_tuned_llm(**inputs) 
    logits = out.logits[0, -1, option_ids]
    return torch.softmax(logits.float(), dim=0).cpu().numpy()
