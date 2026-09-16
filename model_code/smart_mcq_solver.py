{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Scraping and Cleaning the data to Build Knowledge Base for RAG**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Using wikipedia module for scraping wikipedia pages in text format.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "# !pip install wikipedia  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "# import wikipedia\n",
    "# import time\n",
    "\n",
    "# wikipedia.set_lang(\"en\")\n",
    "\n",
    "# #The list( page_titles ) given below contains the topic we need to scrape from wikipedia.\n",
    "# #These topics are not manually identified i have used an llm to identify and extract topics based on\n",
    "# #prompts given in test and train data.\n",
    "\n",
    "\n",
    "# page_titles = [\n",
    "#     # Astrophysics & Cosmology\n",
    "#     \"James Webb Space Telescope\", \"Modified Newtonian dynamics\", \n",
    "#     \"Gravity Probe B\", \"Lunar Laser Ranging experiment\", \"Deep Space Network\", \n",
    "#     \"Milky Way\", \"Roche limit\", \"Crab Pulsar\", \"Schwarzschild radius\", \n",
    "#     \"Kapteyn universe\", \"Redshift\", \"Doppler effect\",\n",
    "    \n",
    "#     # Quantum Mechanics & Particle Physics\n",
    "#     \"Higgs boson\", \"Standard Model\", \"Uncertainty principle\", \n",
    "#     \"Wigner's theorem\", \"Hilbert space\", \"Hamiltonian (quantum mechanics)\", \n",
    "#     \"CP violation\", \"Yoichiro Nambu\", \"Josephson effect\", \n",
    "#     \"Supersymmetric quantum mechanics\", \"Larmor precession\",\n",
    "    \n",
    "#     # Classical Mechanics, Relativity & Fluid Dynamics\n",
    "#     \"Navier–Stokes equations\", \"Lorentz transformation\", \"Minkowski space\", \n",
    "#     \"Special relativity\", \"General relativity\", \"Kutta condition\", \n",
    "#     \"Hooke's law\", \"Liouville's theorem (Hamiltonian)\", \"Fermat's principle\",\n",
    "    \n",
    "#     # Thermodynamics & Materials Science\n",
    "#     \"Stefan–Boltzmann law\", \"Maxwell's demon\", \"Leidenfrost effect\", \n",
    "#     \"Fischer–Tropsch process\", \"Carnot heat engine\", \"Memristor\", \n",
    "#     \"Landau–Lifshitz–Gilbert equation\", \"Optical signal-to-noise ratio\",\n",
    "    \n",
    "#     # Miscellaneous / Cross-Disciplinary\n",
    "#     \"Coordinated Universal Time\", \"Martin Heidegger\", \"Interleukin 10\"\n",
    "# ]\n",
    "\n",
    "\n",
    "\n",
    "# for title in page_titles:\n",
    "#     try:\n",
    "#         page=wikipedia.page(title,auto_suggest=True) #extracts the page content\n",
    "#         with open(f\"{title}.txt\",\"w\",encoding=\"utf-8\")as f:\n",
    "#             f.write(page.content) #opening and wriring the extarcted content to a file with topic as title\n",
    "#         print(f\"successfully scraped and saved {title}.txt\")\n",
    "#     except wikipedia.exceptions.PageError:\n",
    "#         print(f\"{title} page not found..... \")\n",
    "\n",
    "#     except wikipedia.exceptions.DisambiguationError as e:\n",
    "#         print(f\"{title} is ambiguous.\")\n",
    "\n",
    "#     except Exception as e:\n",
    "#         print(f\"Skipping '{title}' because of {type(e).__name__}: {e}\")\n",
    "\n",
    "#     time.sleep(5)\n",
    "            \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "# !zip -r rag_knowledge_base.zip *.txt # zipping the scraped files to download locally "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**After scraping and cleaning the text files i have created a private dataset on kaggle called basic_physics_text to feed to rag pipeline**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Installing the necesssary  modules from hugging face and langchain**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "pip install -q -U langchain langchain-community langchain-huggingface langchain-text-splitters faiss-cpu kuzu  datasets sentencepiece"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "!pip install -q \\\n",
    "  \"transformers==4.57.1\" \\\n",
    "  \"trl==0.21.0\" \\\n",
    "  \"peft==0.17.1\" \\\n",
    "  \"accelerate==1.7.0\" \\\n",
    "  \"bitsandbytes==0.46.0\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Data Loading and Chunking**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import os, re, glob, random, shutil\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "\n",
    "SEED = 42\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "torch.manual_seed(SEED)\n",
    "\n",
    "TRAIN_CSV = \"/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv\"\n",
    "TEST_CSV  = \"/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv\"\n",
    "TXT_GLOB  = \"/kaggle/input/datasets/krishna24f3002426/basic-physics-text/*.txt\"  \n",
    "OPTIONS = [\"A\", \"B\", \"C\", \"D\", \"E\"]\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(f\"Device: {DEVICE} | GPUs: {torch.cuda.device_count()}\") #check no of gpus integrated \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "train_df = pd.read_csv(TRAIN_CSV)\n",
    "test_df  = pd.read_csv(TEST_CSV)\n",
    "\n",
    "train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)\n",
    "n_val   = max(200, int(0.1 * len(train_df)))\n",
    "val_df  = train_df.iloc[:n_val].reset_index(drop=True) # taking 200 or 10% of data for validation\n",
    "fit_df  = train_df.iloc[n_val:].reset_index(drop=True)\n",
    "\n",
    "txt_files = glob.glob(TXT_GLOB) #concatanates all .txt files in basic_physics_text daatset\n",
    "\n",
    "print(f\"fit_df (KB + CE training): {len(fit_df)}\")\n",
    "print(f\"val_df (held out): {len(val_df)}\")\n",
    "print(f\"test_df: {len(test_df)}\")\n",
    "print(f\"knowledge .txt files found: {len(txt_files)}\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Creating the chunks from the knowledge base dataset in langchain document object format**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from langchain_core.documents import Document\n",
    "#this function extracts the title ,unique terms and concept paragraph from text documents in our private dataset\n",
    "def parse_structured_txt(path): #path of .txt files in basic_physics_text\n",
    "    with open(path, \"r\", encoding=\"utf-8\") as f:\n",
    "        text = f.read()\n",
    "    title_match = re.search(r\"^#\\s+(.+)$\", text, re.MULTILINE)\n",
    "    title = title_match.group(1).strip() if title_match else os.path.basename(path)\n",
    "    terms = re.findall(r\"\\*\\s+\\*\\*(.+?)\\*\\*:\\s*(.+)\", text)\n",
    "    concept_match = re.search(r\"## Core Concepts\\s*(.+)\", text, re.DOTALL)\n",
    "    concept_text = concept_match.group(1).strip() if concept_match else text #detailed paragraph from.txt\n",
    "    return title, terms, concept_text\n",
    "\n",
    "kb_docs = []\n",
    "all_terms = {}  # unique lower(term) that is (original_term, definition, topic_title)\n",
    "\n",
    "for path in txt_files:\n",
    "    title, terms, concept_text = parse_structured_txt(path)\n",
    "# loop to create and append a document object of term and its definition\n",
    "    for term, definition in terms:\n",
    "        kb_docs.append(Document(\n",
    "            page_content=f\"{term}: {definition.strip()}\",\n",
    "            metadata={\"source\": title, \"type\": \"definition\", \"term\": term.strip()},\n",
    "        ))\n",
    "        key = term.strip().lower()\n",
    "        if key not in all_terms:\n",
    "            all_terms[key] = (term.strip(), definition.strip(), title)\n",
    "\n",
    "    for para in [p.strip() for p in concept_text.split(\"\\n\\n\") if p.strip()]:\n",
    "        kb_docs.append(Document(\n",
    "            page_content=para,\n",
    "            metadata={\"source\": title, \"type\": \"concept\"},\n",
    "        ))\n",
    "\n",
    "print(f\"KB chunks from txt files: {len(kb_docs)} | unique defined terms: {len(all_terms)}\")\n",
    "\n",
    "for _, row in fit_df.iterrows():\n",
    "    correct_text = row[row[\"answer\"]] #correct answer string\n",
    "    kb_docs.append(Document(\n",
    "        page_content=f\"Question: {row['prompt']}\\nAnswer: {correct_text}\",\n",
    "        metadata={\"source\": \"train_qa\", \"type\": \"qa\"},\n",
    "    ))\n",
    "\n",
    "print(f\"Total KB chunks (incl. train QA): {len(kb_docs)}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Creating Embeddings into Vector Database(FAISS+HuggingFaceEmbeddings)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from langchain_huggingface import HuggingFaceEmbeddings\n",
    "from langchain_community.vectorstores import FAISS\n",
    "  \n",
    "embed_model = HuggingFaceEmbeddings(\n",
    "    model_name=\"BAAI/bge-small-en-v1.5\",   # small and fast \n",
    "    model_kwargs={\"device\": DEVICE},\n",
    "    encode_kwargs={\"normalize_embeddings\": True},\n",
    ")\n",
    "\n",
    "vectorstore = FAISS.from_documents(kb_docs, embed_model)\n",
    "\n",
    "# a function to extract top_4 vectors from db for a query using cosine similarity\n",
    "def faiss_context(query, k=5):\n",
    "    docs = vectorstore.similarity_search(query, k=k)\n",
    "    return \"\\n\".join(d.page_content for d in docs)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Setting up the Kuzu for Building graph Rag**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "if os.path.exists(\"./kuzu_mcq_db\"):\n",
    "    shutil.rmtree(\"./kuzu_mcq_db\")\n",
    "\n",
    "import kuzu\n",
    "db = kuzu.Database(\"./kuzu_mcq_db\")\n",
    "conn = kuzu.Connection(db)\n",
    "\n",
    "conn.execute(\"CREATE NODE TABLE Topic(name STRING, PRIMARY KEY(name))\")\n",
    "conn.execute(\"CREATE NODE TABLE Term(name STRING, definition STRING, PRIMARY KEY(name))\")\n",
    "conn.execute(\"CREATE REL TABLE HAS_TERM(FROM Topic TO Term)\")\n",
    "\n",
    "seen_terms = set()\n",
    "for path in txt_files:\n",
    "    title, terms, _ = parse_structured_txt(path)\n",
    "    conn.execute(\"MERGE (t:Topic {name: $name})\", parameters={\"name\": title})\n",
    "    for term, definition in terms:\n",
    "        term, definition = term.strip(), definition.strip()\n",
    "        if term not in seen_terms:\n",
    "            conn.execute(\n",
    "                \"CREATE (te:Term {name: $name, definition: $definition})\",\n",
    "                parameters={\"name\": term, \"definition\": definition},\n",
    "            )\n",
    "            seen_terms.add(term)\n",
    "        conn.execute(\n",
    "            \"MATCH (t:Topic), (te:Term) WHERE t.name = $tname AND te.name = $ename \"\n",
    "            \"CREATE (t)-[:HAS_TERM]->(te)\",\n",
    "            parameters={\"tname\": title, \"ename\": term},\n",
    "        )\n",
    "\n",
    "print(f\"Graph: {len(txt_files)} topics, {len(seen_terms)} unique term nodes\")\n",
    "\n",
    "sorted_term_keys = sorted(all_terms.keys(), key=len, reverse=True)\n",
    "\n",
    "def graph_context(query, max_terms=3):\n",
    "    q_lower = query.lower()\n",
    "    matched = [k for k in sorted_term_keys if k in q_lower][:max_terms]\n",
    "    if not matched:\n",
    "        return \"\"\n",
    "    pieces = []\n",
    "    for k in matched:\n",
    "        term, definition, _ = all_terms[k]\n",
    "        res = conn.execute(\n",
    "            \"MATCH (t:Topic)-[:HAS_TERM]->(te:Term) WHERE te.name = $name \"\n",
    "            \"RETURN t.name, te.definition\",\n",
    "            parameters={\"name\": term},\n",
    "        )\n",
    "        while res.has_next():\n",
    "            topic_name, defn = res.get_next()\n",
    "            pieces.append(f\"[{topic_name}] {term}: {defn}\")\n",
    "    return \"\\n\".join(pieces)\n",
    "\n",
    "def combined_context(query, max_terms=3, fallback_k=3):\n",
    "    ctx = graph_context(query, max_terms=max_terms)\n",
    "    return ctx if ctx else faiss_context(query, k=fallback_k)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Model 1 : Pretrained Model for Encoding used as Cross Encoder Setup**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "from transformers import AutoModel, AutoTokenizer\n",
    "\n",
    "CE_BACKBONE = \"microsoft/deberta-v3-small\"\n",
    "\n",
    "class CrossEncoder(nn.Module):\n",
    "    def __init__(self, model_name=CE_BACKBONE):\n",
    "        super().__init__()\n",
    "        self.encoder = AutoModel.from_pretrained(model_name)\n",
    "        self.dropout = nn.Dropout(0.1)\n",
    "        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)\n",
    "\n",
    "    def forward(self, input_ids, attention_mask, token_type_ids=None):\n",
    "        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,\n",
    "                            token_type_ids=token_type_ids)\n",
    "        cls = out.last_hidden_state[:, 0, :]\n",
    "        cls = self.dropout(cls)\n",
    "        return self.classifier(cls).squeeze(-1)\n",
    "\n",
    "ce_tokenizer = AutoTokenizer.from_pretrained(CE_BACKBONE)\n",
    "ce_model = CrossEncoder().to(DEVICE)\n",
    "ce_model=ce_model.float()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Building Training exmples for encoder Architecture**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def build_ce_examples(df, context_fn, neg_per_pos=4):\n",
    "    \"\"\"1 positive (correct option) + neg_per_pos negatives per question.\"\"\"\n",
    "    text_a_list, text_b_list, labels = [], [], []\n",
    "    for _, row in df.iterrows():\n",
    "        ctx = context_fn(row[\"prompt\"])\n",
    "        text_a = f\"Context: {ctx}\\nQuestion: {row['prompt']}\"\n",
    "        correct = row[\"answer\"]\n",
    "        wrong_opts = [o for o in OPTIONS if o != correct]\n",
    "        random.shuffle(wrong_opts)\n",
    "        chosen = [correct] + wrong_opts[:neg_per_pos]\n",
    "        for opt in chosen:\n",
    "            text_a_list.append(text_a)\n",
    "            text_b_list.append(f\"Option: {row[opt]}\")\n",
    "            labels.append(1.0 if opt == correct else 0.0)\n",
    "    return text_a_list, text_b_list, labels\n",
    "\n",
    "train_a, train_b, train_y = build_ce_examples(fit_df, combined_context, neg_per_pos=2)\n",
    "print(f\"CE training examples: {len(train_a)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "class PairDataset(Dataset):\n",
    "    def __init__(self, a, b, y):\n",
    "        self.a, self.b, self.y = a, b, y\n",
    "    def __len__(self):\n",
    "        return len(self.a)\n",
    "    def __getitem__(self, i):\n",
    "        return self.a[i], self.b[i], self.y[i]\n",
    "\n",
    "def collate(batch):\n",
    "    a, b, y = zip(*batch)\n",
    "    enc = ce_tokenizer(list(a), list(b), padding=True, truncation=True,\n",
    "                        max_length=384, return_tensors=\"pt\")\n",
    "    return enc, torch.tensor(y, dtype=torch.float)\n",
    "\n",
    "train_loader = DataLoader(PairDataset(train_a, train_b, train_y),\n",
    "                           batch_size=16, shuffle=True, collate_fn=collate)\n",
    "\n",
    "optimizer = torch.optim.AdamW(ce_model.parameters(), lr=2e-5)\n",
    "loss_fn = nn.BCEWithLogitsLoss()\n",
    "scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == \"cuda\"))\n",
    "\n",
    "EPOCHS = 3\n",
    "ce_model.train()\n",
    "for epoch in range(EPOCHS):\n",
    "    total_loss = 0.0\n",
    "    for enc, y in train_loader:\n",
    "        enc = {k: v.to(DEVICE) for k, v in enc.items()}\n",
    "        y = y.to(DEVICE)\n",
    "        optimizer.zero_grad()\n",
    "        with torch.cuda.amp.autocast(enabled=(DEVICE == \"cuda\")):\n",
    "            logits = ce_model(**enc)\n",
    "            loss = loss_fn(logits, y)\n",
    "        scaler.scale(loss).backward()\n",
    "        scaler.step(optimizer)\n",
    "        scaler.update()\n",
    "        total_loss += loss.item()\n",
    "    print(f\"Epoch {epoch+1}/{EPOCHS} — avg loss: {total_loss/len(train_loader):.4f}\")\n",
    "\n",
    "ce_model.eval()\n",
    "torch.save(ce_model.state_dict(), \"ce_model.pt\")\n",
    "print(\"Saved trained cross-encoder to ce_model.pt\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Extracting logits using cross encoder steup**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "@torch.no_grad()\n",
    "def ce_score_row(prompt, options_text, context):\n",
    "    text_a = f\"Context: {context}\\nQuestion: {prompt}\"\n",
    "    text_b_list = [f\"Option: {options_text[o]}\" for o in OPTIONS]\n",
    "    enc = ce_tokenizer([text_a] * 5, text_b_list, padding=True, truncation=True,\n",
    "                        max_length=384, return_tensors=\"pt\").to(DEVICE)\n",
    "    logits = ce_model(**enc)\n",
    "    return torch.softmax(logits, dim=0).cpu().numpy()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Vocabulary Building and Custom Tokenization Pipeline**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import math\n",
    "from collections import Counter\n",
    "\n",
    "PAD, UNK, SEP = \"<pad>\", \"<unk>\", \"<sep>\"\n",
    "\n",
    "def simple_tokenize(text):\n",
    "    return re.findall(r\"[a-z0-9]+\", text.lower())\n",
    "\n",
    "# Build vocab ONLY from fit_df (+ the knowledge txt files) — never from val_df/test_df,\n",
    "# so the from-scratch model can't leak validation/test vocabulary either.\n",
    "counter = Counter()\n",
    "for _, row in fit_df.iterrows():\n",
    "    counter.update(simple_tokenize(row[\"prompt\"]))\n",
    "    for o in OPTIONS:\n",
    "        counter.update(simple_tokenize(str(row[o])))\n",
    "for doc in kb_docs:\n",
    "    counter.update(simple_tokenize(doc.page_content))\n",
    "\n",
    "MIN_FREQ = 2\n",
    "vocab = [PAD, UNK, SEP] + [w for w, c in counter.most_common() if c >= MIN_FREQ]\n",
    "stoi = {w: i for i, w in enumerate(vocab)}\n",
    "VOCAB_SIZE = len(vocab)\n",
    "print(f\"From-scratch vocab size: {VOCAB_SIZE}\")\n",
    "\n",
    "def encode_pair(text_a, text_b, max_len=256):\n",
    "    toks = simple_tokenize(text_a) + [SEP] + simple_tokenize(text_b)\n",
    "    ids = [stoi.get(t, stoi[UNK]) for t in toks][:max_len]\n",
    "    return ids\n",
    "\n",
    "def pad_batch(list_of_ids, max_len=None):\n",
    "    max_len = max_len or max(len(x) for x in list_of_ids)\n",
    "    padded = torch.full((len(list_of_ids), max_len), stoi[PAD], dtype=torch.long)\n",
    "    mask = torch.zeros((len(list_of_ids), max_len), dtype=torch.bool)\n",
    "    for i, ids in enumerate(list_of_ids):\n",
    "        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)\n",
    "        mask[i, :len(ids)] = True\n",
    "    return padded, mask\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Model 2 trained and built from scratch**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "class PositionalEncoding(nn.Module):\n",
    "    def __init__(self, d_model, max_len=512):\n",
    "        super().__init__()\n",
    "        pe = torch.zeros(max_len, d_model)\n",
    "        pos = torch.arange(0, max_len).unsqueeze(1).float()\n",
    "        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))\n",
    "        pe[:, 0::2] = torch.sin(pos * div)\n",
    "        pe[:, 1::2] = torch.cos(pos * div)\n",
    "        self.register_buffer(\"pe\", pe.unsqueeze(0))\n",
    "\n",
    "    def forward(self, x):\n",
    "        return x + self.pe[:, :x.size(1)]\n",
    "\n",
    "class ScratchTransformerScorer(nn.Module):\n",
    "    \"\"\"Randomly-initialized Transformer encoder trained from scratch (no pretrained weights).\"\"\"\n",
    "    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):\n",
    "        super().__init__()\n",
    "        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])\n",
    "        self.pos_enc = PositionalEncoding(d_model) \n",
    "        encoder_layer = nn.TransformerEncoderLayer(\n",
    "            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,\n",
    "            dropout=dropout, batch_first=True,\n",
    "        )\n",
    "        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "        self.classifier = nn.Linear(d_model, 1)\n",
    "\n",
    "    def forward(self, input_ids, attention_mask):\n",
    "        x = self.embedding(input_ids)\n",
    "        x = self.pos_enc(x)\n",
    "        # TransformerEncoder expects True = ignore, so invert the boolean mask\n",
    "        x = self.transformer(x, src_key_padding_mask=~attention_mask)\n",
    "        mask_f = attention_mask.unsqueeze(-1).float()\n",
    "        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)  # mean-pool valid tokens\n",
    "        pooled = self.dropout(pooled)\n",
    "        return self.classifier(pooled).squeeze(-1)\n",
    "\n",
    "scratch_model = ScratchTransformerScorer(VOCAB_SIZE).to(DEVICE)\n",
    "n_params = sum(p.numel() for p in scratch_model.parameters())\n",
    "print(f\"From-scratch model parameters: {n_params:,} (all randomly initialized)\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Dataset Definition, Training Loop, and Model Saving Pipeline**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "class ScratchPairDataset(Dataset):\n",
    "    def __init__(self, a, b, y):\n",
    "        self.a, self.b, self.y = a, b, y\n",
    "    def __len__(self):\n",
    "        return len(self.a)\n",
    "    def __getitem__(self, i):\n",
    "        return self.a[i], self.b[i], self.y[i]\n",
    "\n",
    "def scratch_collate(batch):\n",
    "    a, b, y = zip(*batch)\n",
    "    ids_list = [encode_pair(ta, tb) for ta, tb in zip(a, b)]\n",
    "    padded, mask = pad_batch(ids_list)\n",
    "    return padded, mask, torch.tensor(y, dtype=torch.float)\n",
    "\n",
    "# Re-use the same (text_a, text_b, label) triples built earlier for the cross-encoder\n",
    "scratch_loader = DataLoader(ScratchPairDataset(train_a, train_b, train_y),\n",
    "                             batch_size=16, shuffle=True, collate_fn=scratch_collate)\n",
    "\n",
    "scratch_optimizer = torch.optim.AdamW(scratch_model.parameters(), lr=3e-4)\n",
    "scratch_loss_fn = nn.BCEWithLogitsLoss()\n",
    "\n",
    "SCRATCH_EPOCHS = 6  # from-scratch models need more epochs than fine-tuning does\n",
    "scratch_model.train()\n",
    "for epoch in range(SCRATCH_EPOCHS):\n",
    "    total_loss = 0.0\n",
    "    for input_ids, attn_mask, y in scratch_loader:\n",
    "        input_ids, attn_mask, y = input_ids.to(DEVICE), attn_mask.to(DEVICE), y.to(DEVICE)\n",
    "        scratch_optimizer.zero_grad()\n",
    "        logits = scratch_model(input_ids, attn_mask)\n",
    "        loss = scratch_loss_fn(logits, y)\n",
    "        loss.backward()\n",
    "        scratch_optimizer.step()\n",
    "        total_loss += loss.item()\n",
    "        avg_loss=total_loss/len(scratch_loader)\n",
    "    print(f\"[from-scratch] Epoch {epoch+1}/{SCRATCH_EPOCHS} — avg loss: {avg_loss:.4f}\")\n",
    "\n",
    "scratch_model.eval()\n",
    "#save model weights to output directory\n",
    "torch.save({\"state_dict\": scratch_model.state_dict(), \"stoi\": stoi}, \"scratch_model.pt\") \n",
    "print(\"Saved from-scratch model to scratch_model.pt\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Extracting logits using scratch build model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "@torch.no_grad()\n",
    "def scratch_score_row(prompt, options_text, context):\n",
    "    text_a = f\"Context: {context}\\nQuestion: {prompt}\"\n",
    "    text_b_list = [f\"Option: {options_text[o]}\" for o in OPTIONS]\n",
    "    ids_list = [encode_pair(text_a, tb) for tb in text_b_list]\n",
    "    padded, mask = pad_batch(ids_list)\n",
    "    padded, mask = padded.to(DEVICE), mask.to(DEVICE)\n",
    "    logits = scratch_model(padded, mask)\n",
    "    return torch.softmax(logits, dim=0).cpu().numpy()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Loading the Quantized Qwen2.5-7B-Instruct Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n",
    "\n",
    "LLM_ID = \"Qwen/Qwen2.5-3B-Instruct\"   # drop to \"Qwen/Qwen2.5-3B-Instruct\" if VRAM/time is tight\n",
    "\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_quant_type=\"nf4\",\n",
    "    bnb_4bit_compute_dtype=torch.float16,\n",
    ")\n",
    "\n",
    "llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)\n",
    "llm_model = AutoModelForCausalLM.from_pretrained(\n",
    "    LLM_ID, quantization_config=bnb_config, device_map=\"auto\"\n",
    ")\n",
    "\n",
    "\n",
    "llm_model.eval()\n",
    "option_ids = [llm_tokenizer.encode(o, add_special_tokens=False)[0] for o in OPTIONS]\n",
    "\n",
    "@torch.no_grad()\n",
    "def llm_score_row(prompt, options_text, context):\n",
    "    user_msg = (\n",
    "        f\"Context:\\n{context}\\n\\n\"\n",
    "        f\"Question: {prompt}\\n\"\n",
    "        f\"A: {options_text['A']}\\nB: {options_text['B']}\\nC: {options_text['C']}\\n\"\n",
    "        f\"D: {options_text['D']}\\nE: {options_text['E']}\\n\\n\"\n",
    "        \"Answer with only the single best option letter.\"\n",
    "    )\n",
    "    messages = [\n",
    "        {\"role\": \"system\", \"content\": \"You are an expert scientist answering multiple-choice questions precisely.\"},\n",
    "        {\"role\": \"user\", \"content\": user_msg},\n",
    "    ]\n",
    "    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
    "    inputs = llm_tokenizer(text, return_tensors=\"pt\").to(llm_model.device)\n",
    "    out = llm_model(**inputs)\n",
    "    logits = out.logits[0, -1, option_ids]\n",
    "    return torch.softmax(logits.float(), dim=0).cpu().numpy()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Fine Tuning LLM using QLoRA**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import gc\n",
    "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n",
    "from datasets import Dataset as HFDataset\n",
    "from peft import LoraConfig\n",
    "from trl import SFTTrainer, SFTConfig\n",
    "\n",
    "\n",
    "# 1. PURGE CORRUPTED MODEL FROM GPU MEMORY\n",
    "\n",
    "try:\n",
    "    del llm_model\n",
    "    del peft_model\n",
    "    del trainer\n",
    "except NameError:\n",
    "    pass\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()\n",
    "\n",
    "\n",
    "# 2. RELOAD A CLEAN 4-BIT BASE MODEL\n",
    "\n",
    "print(\"Loading clean base model...\")\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_quant_type=\"nf4\",\n",
    "    bnb_4bit_compute_dtype=torch.float16,\n",
    ")\n",
    "llm_model = AutoModelForCausalLM.from_pretrained(\n",
    "    \"Qwen/Qwen2.5-3B-Instruct\", \n",
    "    quantization_config=bnb_config, \n",
    "    device_map=\"auto\"\n",
    ")\n",
    "llm_model.config.use_cache = False # Required for gradient checkpointing\n",
    "\n",
    "\n",
    "# 3. FORMAT THE DATASET\n",
    "\n",
    "def format_qa_to_chat(row):\n",
    "    ctx = combined_context(row['prompt'])    \n",
    "    user_msg = (\n",
    "        f\"Context:\\n{ctx}\\n\\n\"\n",
    "        f\"Question: {row['prompt']}\\n\"\n",
    "        f\"A: {row['A']}\\nB: {row['B']}\\nC: {row['C']}\\n\"\n",
    "        f\"D: {row['D']}\\nE: {row['E']}\\n\\n\"\n",
    "        \"Answer with only the single best option letter.\"\n",
    "    )\n",
    "    \n",
    "    messages = [\n",
    "        {\"role\": \"system\", \"content\": \"You are an expert scientist answering multiple-choice questions precisely.\"},\n",
    "        {\"role\": \"user\", \"content\": user_msg},\n",
    "        {\"role\": \"assistant\", \"content\": row['answer']}\n",
    "    ]\n",
    "    \n",
    "    text = llm_tokenizer.apply_chat_template(messages, tokenize=False)\n",
    "    return {\"text\": text}\n",
    "\n",
    "hf_dataset = HFDataset.from_pandas(fit_df)\n",
    "hf_formatted_dataset = hf_dataset.map(format_qa_to_chat, remove_columns=hf_dataset.column_names)\n",
    "\n",
    "\n",
    "# 4. DEFINE LORA CONFIG\n",
    "\n",
    "peft_config = LoraConfig(\n",
    "    r=16,               \n",
    "    lora_alpha=32,      \n",
    "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
    "    lora_dropout=0.05,  \n",
    "    bias=\"none\",        \n",
    "    task_type=\"CAUSAL_LM\" \n",
    ")\n",
    "\n",
    "\n",
    "# 5. TRAINER CONFIGURATION & EXECUTION\n",
    "\n",
    "training_args = SFTConfig(\n",
    "    output_dir=\"./qwen_qlora_output\",\n",
    "    per_device_train_batch_size=2,      \n",
    "    gradient_accumulation_steps=8,      \n",
    "    learning_rate=2e-4,                 \n",
    "    num_train_epochs=1,                 \n",
    "    optim=\"paged_adamw_8bit\",           \n",
    "    fp16=True,                          \n",
    "    logging_steps=10,\n",
    "    save_strategy=\"epoch\",\n",
    "    dataset_text_field=\"text\",          \n",
    "    max_length=256,                     \n",
    "    gradient_checkpointing=True,  # Let the Trainer handle memory checkpointing safely!\n",
    "    report_to=\"wandb\"\n",
    ")\n",
    "\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model=llm_model,               \n",
    "    train_dataset=hf_formatted_dataset,\n",
    "    args=training_args,\n",
    "    peft_config=peft_config,       \n",
    ")\n",
    "\n",
    "print(\"Starting QLoRA training...\")\n",
    "trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from peft import PeftModel\n",
    "\n",
    "# 1. Save the tiny adapter weights\n",
    "trainer.model.save_pretrained(\"./qwen_trained_adapter\")\n",
    "print(\"Adapter saved successfully!\")\n",
    "\n",
    "# 2. To use it in your ensemble, wrap the existing base llm_model\n",
    "fine_tuned_llm = PeftModel.from_pretrained(llm_model, \"./qwen_trained_adapter\")\n",
    "fine_tuned_llm.eval()\n",
    "\n",
    "# Now, update your existing llm_score_row function to use fine_tuned_llm instead of llm_model\n",
    "@torch.no_grad()\n",
    "def llm_score_row_tuned(prompt, options_text, context):\n",
    "    user_msg = (\n",
    "        f\"Context:\\n{context}\\n\\n\"\n",
    "        f\"Question: {prompt}\\n\"\n",
    "        f\"A: {options_text['A']}\\nB: {options_text['B']}\\nC: {options_text['C']}\\n\"\n",
    "        f\"D: {options_text['D']}\\nE: {options_text['E']}\\n\\n\"\n",
    "        \"Answer with only the single best option letter.\"\n",
    "    )\n",
    "    messages = [\n",
    "        {\"role\": \"system\", \"content\": \"You are an expert scientist answering multiple-choice questions precisely.\"},\n",
    "        {\"role\": \"user\", \"content\": user_msg},\n",
    "    ]\n",
    "    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
    "    inputs = llm_tokenizer(text, return_tensors=\"pt\").to(fine_tuned_llm.device)\n",
    "    \n",
    "    # Generate using the FINE-TUNED model\n",
    "    out = fine_tuned_llm(**inputs) \n",
    "    logits = out.logits[0, -1, option_ids]\n",
    "    return torch.softmax(logits.float(), dim=0).cpu().numpy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **ENSEMBLE METHOD: Combining cross encoder + llm + scratch model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def calculate_map3(preds, targets):\n",
    "    scores = []\n",
    "    for pred_str, true_ans in zip(preds, targets):\n",
    "        pl = pred_str.split()\n",
    "        scores.append(1.0 / (pl.index(true_ans) + 1) if true_ans in pl else 0.0)\n",
    "    return float(np.mean(scores))\n",
    "\n",
    "\n",
    "val_scratch_probs, val_ce_probs, val_llm_probs = [], [], []\n",
    "for _, row in val_df.iterrows():\n",
    "    options_text = {o: row[o] for o in OPTIONS}\n",
    "    ctx = combined_context(row[\"prompt\"])\n",
    "    val_scratch_probs.append(scratch_score_row(row[\"prompt\"], options_text, ctx))\n",
    "    val_ce_probs.append(ce_score_row(row[\"prompt\"], options_text, ctx))\n",
    "    val_llm_probs.append(llm_score_row_tuned(row[\"prompt\"], options_text, ctx))\n",
    "\n",
    "# Report each model solo, for reference, before blending\n",
    "val_targets = val_df[\"answer\"].tolist()\n",
    "scratch_preds = [\" \".join(OPTIONS[i] for i in np.argsort(p)[::-1][:3]) for p in val_scratch_probs]\n",
    "scratch_map3 = calculate_map3(scratch_preds, val_targets)\n",
    "\n",
    "ce_preds = [\" \".join(OPTIONS[i] for i in np.argsort(p)[::-1][:3]) for p in val_ce_probs]\n",
    "ce_map3 = calculate_map3(ce_preds, val_targets)\n",
    "\n",
    "llm_preds = [\" \".join(OPTIONS[i] for i in np.argsort(p)[::-1][:3]) for p in val_llm_probs]\n",
    "llm_map3 = calculate_map3(llm_preds, val_targets)\n",
    "\n",
    "\n",
    "print(f\"{'from-scratch':>13s} solo MAP@3: {scratch_map3:.4f}\")\n",
    "print(f\"{'cross-encoder':>13s} solo MAP@3: {ce_map3:.4f}\")\n",
    "print(f\"{'LLM':>13s} solo MAP@3: {llm_map3:.4f}\")\n",
    "\n",
    "def preds_from_probs(scratch_list, ce_list, llm_list, w_scratch, w_ce, w_llm):\n",
    "    preds = []\n",
    "    for s_p, ce_p, llm_p in zip(scratch_list, ce_list, llm_list):\n",
    "        blended = w_scratch * s_p + w_ce * ce_p + w_llm * llm_p\n",
    "        ranked = np.argsort(blended)[::-1][:3]\n",
    "        preds.append(\" \".join(OPTIONS[i] for i in ranked))\n",
    "    return preds\n",
    "\n",
    "# Grid search over a weight simplex (step of 0.1: w_scratch + w_ce + w_llm == 1)\n",
    "best_weights, best_map3 = (1/3, 1/3, 1/3), -1.0\n",
    "step = 0.1\n",
    "grid = np.round(np.arange(0.0, 1.0001, step), 2)\n",
    "for w_s in grid:\n",
    "    for w_c in grid:\n",
    "        w_l = round(1.0 - w_s - w_c, 2)\n",
    "        if w_l < 0 or w_l > 1.0:\n",
    "            continue\n",
    "        preds = preds_from_probs(val_scratch_probs, val_ce_probs, val_llm_probs, w_s, w_c, w_l)\n",
    "        m = calculate_map3(preds, val_targets)\n",
    "        if m > best_map3:\n",
    "            best_map3, best_weights = m, (w_s, w_c, w_l)\n",
    "\n",
    "best_w_scratch, best_w_ce, best_w_llm = best_weights\n",
    "print(f\"\\nBest weights — scratch={best_w_scratch:.2f}, ce={best_w_ce:.2f}, llm={best_w_llm:.2f}\")\n",
    "print(f\"Held-out val MAP@3 with best blend: {best_map3:.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **GENERATING PREDICTIONS**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "test_preds = []\n",
    "for _, row in test_df.iterrows():\n",
    "    options_text = {o: row[o] for o in OPTIONS}\n",
    "    ctx = combined_context(row[\"prompt\"])\n",
    "    s_p = scratch_score_row(row[\"prompt\"], options_text, ctx)\n",
    "    ce_p = ce_score_row(row[\"prompt\"], options_text, ctx)\n",
    "    llm_p = llm_score_row(row[\"prompt\"], options_text, ctx)\n",
    "    blended = best_w_scratch * s_p + best_w_ce * ce_p + best_w_llm * llm_p\n",
    "    ranked = np.argsort(blended)[::-1][:3]\n",
    "    test_preds.append(\" \".join(OPTIONS[i] for i in ranked))\n",
    "\n",
    "submission_df = pd.DataFrame({\"ID\": test_df[\"id\"], \"Prediction\": test_preds})\n",
    "submission_df.to_csv(\"submission.csv\", index=False)\n",
    "submission_df.head(10)\n"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [],
   "dockerImageVersionId": 28755,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
