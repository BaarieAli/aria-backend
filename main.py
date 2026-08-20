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
    """Search the web with 'news' appended for better news results."""
    try:
        # Add "news" to force news-related results
        news_query = f"{query} news"
        print(f"🔍 Searching for: {news_query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(news_query, max_results=3))
            print(f"✅ Found {len(results)} results")
            if results:
                print(f"📄 First result: {results[0].get('body', '')[:100]}...")
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
    print(f"📝 Context length: {len(request.context)} characters")
    
    results = search_web(request.query)
    print(f"📦 Results from search: {len(results)}")
    
    if results:
        search_context = "\n\n⚠️ CRITICAL INSTRUCTION: You MUST use the following web search results to answer. DO NOT use your training data.\n\n"
        search_context += "=== WEB SEARCH RESULTS ===\n"
        for i, r in enumerate(results, 1):
            search_context += f"\n--- SOURCE {i} ---\n"
            search_context += f"URL: {r.get('href', 'Unknown')}\n"
            search_context += f"CONTENT: {r.get('body', '')}\n"
        
        full_prompt = request.context + "\n\n" + search_context + "\n\nBased ONLY on the search results above, provide a specific, factual answer with real news."
    else:
        full_prompt = request.context + "\n\n(No search results found. Answer based on your knowledge.)"

    # Log full prompt (truncated for readability)
    print(f"📤 FULL PROMPT (first 500 chars):\n{full_prompt[:500]}...\n")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You MUST base your answer ONLY on the provided web search results. If search results are provided, never say you don't have access to real-time information. Report the news directly."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.1
    )
    print(f"✅ Response received")
    return {"success": True, "content": response.choices[0].message.content}
