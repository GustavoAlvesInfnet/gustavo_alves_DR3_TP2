from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import FakeListLLM

# Configuração do Fake LLM
# As respostas são retornadas sequencialmente a cada chamada.
responses = [
    "Olá! Como posso ajudá-lo hoje?",
    "Eu sou um chatbot Fake LLM.",
    "Eu respondo perguntas para simular um chatbot básico.",
    "Até logo! Foi bom falar com você.",
    "Desculpe, não entendi sua pergunta.",
]

# Fake LLM com respostas pré-definidas
fake_llm = FakeListLLM(responses=responses)

# Inicializando FastAPI
app = FastAPI()

# Modelo de Entrada
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Gera uma resposta usando o FakeLLM
        response = fake_llm(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    


    # ------------------ 2 ------------------


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Inicializar FastAPI
app = FastAPI()

# Baixar e carregar o modelo GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Modelo de entrada da API
class TextRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_text(request: TextRequest):
    try:
        # Tokenizar o texto de entrada
        inputs = tokenizer.encode(request.text, return_tensors="pt", truncation=True, max_length=50)
        
        # Gerar texto com o GPT-2
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True)
        
        # Decodificar o texto gerado
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# ------------------ 3 ------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# Inicializando FastAPI
app = FastAPI()

# Configurando o modelo HuggingFace com pipeline
translation_pipeline = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")

# Configurando o LLM do LangChain
llm = HuggingFacePipeline(pipeline=translation_pipeline)

# Template de prompt para LangChain
prompt = PromptTemplate(
    input_variables=["text"],
    template="{text}"  # Para tradução, o texto é usado diretamente.
)

# Criando a cadeia LangChain
translation_chain = LLMChain(prompt=prompt, llm=llm)

# Modelo de entrada da API
class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Usando a cadeia LangChain para traduzir
        translated_text = translation_chain.run(text=request.text)
        return {"translated_text": translated_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


