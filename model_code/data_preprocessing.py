import os, re, glob, random, shutil
import numpy as np
import pandas as pd
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_CSV = "training_data_path"
TEST_CSV  = "test_data_path"
TXT_GLOB  = "rag_knowledge_base_txt_file_path"  
OPTIONS = ["A", "B", "C", "D", "E"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | GPUs: {torch.cuda.device_count()}") #check no of gpus integrated 

train_df = pd.read_csv(TRAIN_CSV) 
test_df  = pd.read_csv(TEST_CSV)

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val   = max(200, int(0.1 * len(train_df)))
val_df  = train_df.iloc[:n_val].reset_index(drop=True) # taking 200 or 10% of data for validation
fit_df  = train_df.iloc[n_val:].reset_index(drop=True)

txt_files = glob.glob(TXT_GLOB) #concatanates all .txt files in basic_physics_text daatset

print(f"fit_df (KB + CE training): {len(fit_df)}")
print(f"val_df (held out): {len(val_df)}")
print(f"test_df: {len(test_df)}")
print(f"knowledge .txt files found: {len(txt_files)}")


'''Creating the chunks from the knowledge base dataset in langchain document object format¶'''

from langchain_core.documents import Document
#this function extracts the title ,unique terms and concept paragraph from text documents in our private dataset
def parse_structured_txt(path): #path of .txt files in basic_physics_text
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else os.path.basename(path)
    terms = re.findall(r"\*\s+\*\*(.+?)\*\*:\s*(.+)", text)
    concept_match = re.search(r"## Core Concepts\s*(.+)", text, re.DOTALL)
    concept_text = concept_match.group(1).strip() if concept_match else text #detailed paragraph from.txt
    return title, terms, concept_text

kb_docs = []
all_terms = {}  # unique lower(term) that is (original_term, definition, topic_title)

for path in txt_files:
    title, terms, concept_text = parse_structured_txt(path)
# loop to create and append a document object of term and its definition
    for term, definition in terms:
        kb_docs.append(Document(
            page_content=f"{term}: {definition.strip()}",
            metadata={"source": title, "type": "definition", "term": term.strip()},
        ))
        key = term.strip().lower()
        if key not in all_terms:
            all_terms[key] = (term.strip(), definition.strip(), title)

    for para in [p.strip() for p in concept_text.split("\n\n") if p.strip()]:
        kb_docs.append(Document(
            page_content=para,
            metadata={"source": title, "type": "concept"},
        ))

print(f"KB chunks from txt files: {len(kb_docs)} | unique defined terms: {len(all_terms)}")

for _, row in fit_df.iterrows():
    correct_text = row[row["answer"]] #correct answer string
    kb_docs.append(Document(
        page_content=f"Question: {row['prompt']}\nAnswer: {correct_text}",
        metadata={"source": "train_qa", "type": "qa"},
    ))

print(f"Total KB chunks (incl. train QA): {len(kb_docs)}")


