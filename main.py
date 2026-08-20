import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

class ChatRequest(BaseModel):
    query: str
    context: str

def search_web(query: str):
    try:
        print(f"🔍 Searching for: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            print(f"✅ Found {len(results)} results")
            return results
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

@app.get("/")
async def root():
    return {"message": "Aria Backend is running!"}

@app.get("/ping")
async def ping():
    return {"status": "alive"}

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"📨 Received query: {request.query}")
    
    results = search_web(request.query)
    print(f"📦 Results from search: {len(results)}")
    
    if results:
        search_context = "\n\n⚠️ CRITICAL: You MUST use the following web search results to answer. DO NOT use your training data.\n\n"
        search_context += "=== WEB SEARCH RESULTS ===\n"
        for i, r in enumerate(results, 1):
            search_context += f"\n--- SOURCE {i} ---\n"
            search_context += f"URL: {r.get('href', 'Unknown')}\n"
            search_context += f"CONTENT: {r.get('body', '')}\n"
        
        full_prompt = request.context + "\n\n" + search_context + "\n\nBased ONLY on the search results above, provide a specific, factual answer."
    else:
        full_prompt = request.context + "\n\n(No search results found. Answer based on your knowledge.)"

    print(f"📤 Sending to DeepSeek...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You MUST base your answer ONLY on the provided web search results. Never say you don't have access to real-time information when search results are provided."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.1
    )
    print(f"✅ Response received")
    return {"success": True, "content": response.choices[0].message.content}
