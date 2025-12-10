# Text-Summarization-Application

A full-stack text-summarization system built using FastAPI (backend), HuggingFace Transformers, and a Gradio UI for accessible interaction.
Supports both online (API-based) and offline (local models) usage.

Text-Summarization-Application/
│
├── backend/
│   ├── main.py
│   ├── app.py
│   ├── summarizer.py
│   ├── vector_store.py
│   ├── nltk_data/                   # offline nltk data
│   ├── local_models/                # local Models (bart-large-cnn, all-MiniLM-L6-v2)
│   ├── packages/                    # <<-- local wheel/tar.gz packages for backend
│   ├── Dockerfile
│   └── requirements.txt (optional)
│
├── ui/
│   ├── app.py
│   ├── packages/                    # <<-- local wheel/tar.gz packages for ui
│   ├── Dockerfile
│   └── requirements.txt (optional)
│
├── docker-compose.yml
└── README.md


## 1 Preparation machine (Linux / macOS)

Open a terminal and run these steps.

1 Create a working directory
export WORKDIR=~/offline-prep
mkdir -p $WORKDIR
cd $WORKDIR

2 Create wheels for backend & ui packages (preferred: wheels)

From the project root of the repo on the internet machine, run:

# Backend wheels (example packages, adapt to your requirements.txt)
mkdir -p $WORKDIR/backend/packages
python3 -m pip download -d $WORKDIR/backend/packages \
  -r /path/to/your/repo/backend/requirements.txt

# UI wheels
mkdir -p $WORKDIR/ui/packages
python3 -m pip download -d $WORKDIR/ui/packages \
  -r /path/to/your/repo/ui/requirements.txt


## 2 Download Hugging Face models into a local folder

Set up environment variables to put HF cache in WORKDIR/hf_cache so we can copy it later:

export HF_HOME=$WORKDIR/hf_cache   # where HF will store files
export TRANSFORMERS_CACHE=$HF_HOME # transformers compatibility
mkdir -p $HF_HOME


Download the models (this may take a while):

python3 - <<'PY'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sentence_transformers import SentenceTransformer
import os

hf_cache = os.environ.get("HF_HOME")
print("HF_HOME:", hf_cache)

# BART (tokenizer + model)
AutoTokenizer.from_pretrained('facebook/bart-large-cnn', cache_dir=hf_cache)
AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn', cache_dir=hf_cache)

# MiniLM (for SentenceTransformer, we will save to local folder later)
st = SentenceTransformer('all-MiniLM-L6-v2')
# Save SentenceTransformer in repo-style folder:
st.save(os.path.join(os.getcwd(), 'all-MiniLM-L6-v2_local'))
print("Saved SentenceTransformer to ./all-MiniLM-L6-v2_local")
PY


## 3 create repo-style local_models
mkdir -p $WORKDIR/local_models
# copy BART snapshot from HF cache to local_models/bart-large-cnn
# find BART snapshot directory inside $HF_HOME/models--facebook--bart-large-cnn
cp -r $HF_HOME/models--facebook--bart-large-cnn/* $WORKDIR/local_models/bart-large-cnn/

# copy your saved SBERT folder
cp -r $WORKDIR/all-MiniLM-L6-v2_local $WORKDIR/local_models/all-MiniLM-L6-v2

## 4 Download NLTK data

python3 - <<'PY'
import nltk
nltk.download('punkt', download_dir='/tmp/nltk_download')
print("done")
PY

# move it into our package
mkdir -p $WORKDIR/backend/nltk_data
cp -r /tmp/nltk_download $WORKDIR/backend/nltk_data

## 5 Collect everything into the project structure

# adjust /path/to/repo to where you will put files later
mkdir -p $WORKDIR/repo_copy/backend
mkdir -p $WORKDIR/repo_copy/ui

# backend files
cp -r $WORKDIR/backend/packages $WORKDIR/repo_copy/backend/packages
cp -r $WORKDIR/local_models $WORKDIR/repo_copy/backend/local_models
cp -r $WORKDIR/backend/nltk_data $WORKDIR/repo_copy/backend/nltk_data

# ui files
cp -r $WORKDIR/ui/packages $WORKDIR/repo_copy/ui/packages
