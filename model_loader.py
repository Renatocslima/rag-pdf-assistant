import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def carregar_modelo_llm(nome_modelo="meta-llama/Meta-Llama-3.1-8B"):
    """
    Carrega o modelo de linguagem e o tokenizador da Hugging Face.
    Detecta automaticamente se há GPU com suporte a CUDA disponível.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo detectado: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    
    if device == "cuda":
        modelo = AutoModelForCausalLM.from_pretrained(
            nome_modelo,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        modelo = AutoModelForCausalLM.from_pretrained(
            nome_modelo,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
    return modelo, tokenizer

def gerar_resposta(modelo, tokenizer, pergunta, contexto, max_new_tokens=150):
    """Gera uma resposta baseada na pergunta e no contexto do PDF."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = f"Contexto do documento:\n{contexto[:2000]}\n\nPergunta: {pergunta}\n\nResposta baseada no documento:"
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        saida = modelo.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    resposta = tokenizer.decode(saida[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return resposta.strip()
