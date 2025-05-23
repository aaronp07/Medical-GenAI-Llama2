# Medical Chatbot - Llama2

Technical Requirement:-
    1. Programming Language -> Python
    2. LangChain / LlamaIndex -> Generative AI Framework
    3. Frontend / Web App -> Flask
    4. LLM -> Meta llama 2
    5. Vector Db -> Pinecone

### Step 1 - Create Project Environment
```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### Step 2 - Install the requirements.txt
```bash
pip install -r ./requirements.txt
```

### Step 3 - Commit in GitHub


### Step 4 - Create a '.env' file in the root directory and add your Pinecone credential

```
PINECONE_API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

### Step 5 - Download the Quantize Model from the link provided in model folder and the model in the model directory

```
## Donload the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q2_K.bin

## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

```

### Step 6 - Create Project Structure

```bash
python  .\template.py

```

```Chatbot HTML UI
URL:    https://codepen.io/Anthony-Hirt/pen/raaoXma
```

