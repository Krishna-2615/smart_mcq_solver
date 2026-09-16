import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

CE_BACKBONE = "microsoft/deberta-v3-small"   ### replace with any model of your choice

class CrossEncoder(nn.Module):
    def __init__(self, model_name=CE_BACKBONE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls).squeeze(-1)

ce_tokenizer = AutoTokenizer.from_pretrained(CE_BACKBONE)
ce_model = CrossEncoder().to(DEVICE)
ce_model=ce_model.float()





def build_ce_examples(df, context_fn, neg_per_pos=4):
    """1 positive (correct option) + neg_per_pos negatives per question."""
    text_a_list, text_b_list, labels = [], [], []
    for _, row in df.iterrows():
        ctx = context_fn(row["prompt"])
        text_a = f"Context: {ctx}\nQuestion: {row['prompt']}"
        correct = row["answer"]
        wrong_opts = [o for o in OPTIONS if o != correct]
        random.shuffle(wrong_opts)
        chosen = [correct] + wrong_opts[:neg_per_pos]
        for opt in chosen:
            text_a_list.append(text_a)
            text_b_list.append(f"Option: {row[opt]}")
            labels.append(1.0 if opt == correct else 0.0)
    return text_a_list, text_b_list, labels

train_a, train_b, train_y = build_ce_examples(fit_df, combined_context, neg_per_pos=2)
print(f"CE training examples: {len(train_a)}")


from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, a, b, y):
        self.a, self.b, self.y = a, b, y
    def __len__(self):
        return len(self.a)
    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]

def collate(batch):
    a, b, y = zip(*batch)
    enc = ce_tokenizer(list(a), list(b), padding=True, truncation=True,
                        max_length=384, return_tensors="pt")
    return enc, torch.tensor(y, dtype=torch.float)

train_loader = DataLoader(PairDataset(train_a, train_b, train_y),
                           batch_size=16, shuffle=True, collate_fn=collate)

optimizer = torch.optim.AdamW(ce_model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

EPOCHS = 1
ce_model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for enc, y in train_loader:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        y = y.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = ce_model(**enc)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        ce_avg_loss=(total_loss/len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {ce_avg_loss:.4f}")
    wandb.log({"CE_Epoch": epoch + 1, "CE_avg_loss": ce_avg_loss})

ce_model.eval()
torch.save(ce_model.state_dict(), "ce_model.pt")
print("Saved trained cross-encoder to ce_model.pt")


@torch.no_grad()
def ce_score_row(prompt, options_text, context):
    text_a = f"Context: {context}\nQuestion: {prompt}"
    text_b_list = [f"Option: {options_text[o]}" for o in OPTIONS]
    enc = ce_tokenizer([text_a] * 5, text_b_list, padding=True, truncation=True,
                        max_length=384, return_tensors="pt").to(DEVICE)
    logits = ce_model(**enc)
    return torch.softmax(logits, dim=0).cpu().numpy()

import math
from collections import Counter

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"

def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

# Build vocab ONLY from fit_df (+ the knowledge txt files) — never from val_df/test_df,
# so the from-scratch model can't leak validation/test vocabulary either.
counter = Counter()
for _, row in fit_df.iterrows():
    counter.update(simple_tokenize(row["prompt"]))
    for o in OPTIONS:
        counter.update(simple_tokenize(str(row[o])))
for doc in kb_docs:
    counter.update(simple_tokenize(doc.page_content))

MIN_FREQ = 2
vocab = [PAD, UNK, SEP] + [w for w, c in counter.most_common() if c >= MIN_FREQ]
stoi = {w: i for i, w in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
print(f"From-scratch vocab size: {VOCAB_SIZE}")

def encode_pair(text_a, text_b, max_len=256):
    toks = simple_tokenize(text_a) + [SEP] + simple_tokenize(text_b)
    ids = [stoi.get(t, stoi[UNK]) for t in toks][:max_len]
    return ids

def pad_batch(list_of_ids, max_len=None):
    max_len = max_len or max(len(x) for x in list_of_ids)
    padded = torch.full((len(list_of_ids), max_len), stoi[PAD], dtype=torch.long)
    mask = torch.zeros((len(list_of_ids), max_len), dtype=torch.bool)
    for i, ids in enumerate(list_of_ids):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, :len(ids)] = True
    return padded, mask






class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ScratchTransformerScorer(nn.Module):
    """Randomly-initialized Transformer encoder trained from scratch (no pretrained weights)."""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos_enc = PositionalEncoding(d_model) 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        # TransformerEncoder expects True = ignore, so invert the boolean mask
        x = self.transformer(x, src_key_padding_mask=~attention_mask)
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)  # mean-pool valid tokens
        pooled = self.dropout(pooled)
        return self.classifier(pooled).squeeze(-1)

scratch_model = ScratchTransformerScorer(VOCAB_SIZE).to(DEVICE)
n_params = sum(p.numel() for p in scratch_model.parameters())
print(f"From-scratch model parameters: {n_params:,} (all randomly initialized)")


class ScratchPairDataset(Dataset):
    def __init__(self, a, b, y):
        self.a, self.b, self.y = a, b, y
    def __len__(self):
        return len(self.a)
    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]

def scratch_collate(batch):
    a, b, y = zip(*batch)
    ids_list = [encode_pair(ta, tb) for ta, tb in zip(a, b)]
    padded, mask = pad_batch(ids_list)
    return padded, mask, torch.tensor(y, dtype=torch.float)

# Re-use the same (text_a, text_b, label) triples built earlier for the cross-encoder
scratch_loader = DataLoader(ScratchPairDataset(train_a, train_b, train_y),
                             batch_size=16, shuffle=True, collate_fn=scratch_collate)

scratch_optimizer = torch.optim.AdamW(scratch_model.parameters(), lr=1e-3)
scratch_loss_fn = nn.BCEWithLogitsLoss()

SCRATCH_EPOCHS = 6  # from-scratch models need more epochs than fine-tuning does
scratch_model.train()
for epoch in range(SCRATCH_EPOCHS):
    total_loss = 0.0
    for input_ids, attn_mask, y in scratch_loader:
        input_ids, attn_mask, y = input_ids.to(DEVICE), attn_mask.to(DEVICE), y.to(DEVICE)
        scratch_optimizer.zero_grad()
        logits = scratch_model(input_ids, attn_mask)
        loss = scratch_loss_fn(logits, y)
        loss.backward()
        scratch_optimizer.step()
        total_loss += loss.item()
        scratch_avg_loss=total_loss/len(scratch_loader)
    print(f"[from-scratch] Epoch {epoch+1}/{SCRATCH_EPOCHS} — avg_loss: {scratch_avg_loss:.4f}")
    wandb.log({"Scratch_Epoch": epoch + 1, "Scratch_avg_loss": scratch_avg_loss})
scratch_model.eval()
#save model weights to output directory
torch.save({"state_dict": scratch_model.state_dict(), "stoi": stoi}, "scratch_model.pt") 
print("Saved from-scratch model to scratch_model.pt")



@torch.no_grad()
def scratch_score_row(prompt, options_text, context):
    text_a = f"Context: {context}\nQuestion: {prompt}"
    text_b_list = [f"Option: {options_text[o]}" for o in OPTIONS]
    ids_list = [encode_pair(text_a, tb) for tb in text_b_list]
    padded, mask = pad_batch(ids_list)
    padded, mask = padded.to(DEVICE), mask.to(DEVICE)
    logits = scratch_model(padded, mask)
    return torch.softmax(logits, dim=0).cpu().numpy()



from transformers import AutoModelForCausalLM, BitsAndBytesConfig

LLM_ID = "Qwen/Qwen2.5-3B-Instruct"   # drop to "Qwen/Qwen2.5-3B-Instruct" if VRAM/time is tight

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_ID, quantization_config=bnb_config, device_map="auto"
)


llm_model.eval()
option_ids = [llm_tokenizer.encode(o, add_special_tokens=False)[0] for o in OPTIONS]

@torch.no_grad()
def llm_score_row(prompt, options_text, context):
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
    inputs = llm_tokenizer(text, return_tensors="pt").to(llm_model.device)
    out = llm_model(**inputs)
    logits = out.logits[0, -1, option_ids]
    return torch.softmax(logits.float(), dim=0).cpu().numpy()

