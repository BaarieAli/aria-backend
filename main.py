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
    query: str       # User's question — used for web search
    context: str     # Full prompt with personality + history — used for DeepSeek

def search_web(query: str):
    try:
        print(f"🔍 Searching for: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            print(f"✅ Found {len(results)} results")
            if results:
                print(f"📄 First result: {results[0].get('body', '')[:100]}...")
            return results
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"📨 Received query: {request.query}")
    
    # 1. Search the web using the user's query
    results = search_web(request.query)
    print(f"📦 Results from search: {len(results)}")
    
    # 2. Inject search results into the full context
    if results:
        search_context = "\n\n⚠️ CRITICAL INSTRUCTION: You MUST use the following web search results to answer the user's question. DO NOT use your training data. ONLY use the information provided below.\n\n"
        search_context += "=== WEB SEARCH RESULTS ===\n"
        for i, r in enumerate(results, 1):
            search_context += f"\n--- SOURCE {i} ---\n"
            search_context += f"URL: {r.get('href', 'Unknown')}\n"
            search_context += f"CONTENT: {r.get('body', '')}\n"
        
        full_prompt = request.context + "\n\n" + search_context + "\n\nBased ONLY on the search results above, provide a specific, factual answer. If the search results contain news, report that news directly. Do not say you don't have access to real-time information."
    else:
        full_prompt = request.context + "\n\n(No search results were found. Answer based on your knowledge.)"

    print(f"📤 Sending to DeepSeek with prompt length: {len(full_prompt)} characters")
    
    # 3. Call DeepSeek with the full context
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. You MUST base your answer ONLY on the provided web search results. If the search results contain specific news, report that news exactly as it appears. Never say you don't have access to real-time information when search results are provided."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.1  # Very low temperature = more factual, less creative
    )
    print(f"✅ Response received")
    return {"success": True, "content": response.choices[0].message.content}