from fastapi import FastAPI, Request
from transformers import GPT2Tokenizer, GPT2Model
import torch

app = FastAPI()

from transformers import GPT2Tokenizer, GPT2LMHeadModel, MarianMTModel, MarianTokenizer


# ------------------ 1 ------------------
# Carregar o modelo GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


@app.post("/generate_text")
async def generate_text(request: Request):
    try:
        # Receber a frase de entrada como JSON
        data = await request.json()
        input_text = data["input_text"]

        # Utilizar a biblioteca transformers para gerar um texto de saída
        inputs = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )

        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # Retornar o texto gerado em uma resposta HTTP
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        return {"error": str(e)}

# ------------------ 2 ------------------

# Carregar o modelo de tradução
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.post("/translate")
async def translate(request: Request):
    # Receber o texto em inglês via requisição HTTP
    data = await request.json()
    text_en = data["text"]

    # Traduzir o texto para o francês utilizando o modelo de tradução
    inputs = tokenizer.encode_plus(
        text_en,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=512,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Retornar o texto traduzido em uma resposta JSON
    text_fr = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": text_fr}


