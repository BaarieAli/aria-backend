import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow all origins (for development — restrict later for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

class ChatRequest(BaseModel):
    query: str       # User's question
    context: str     # Full prompt with personality + history

# ── Smart Search Decision Engine ──────────────────────────────
def should_search(query: str) -> bool:
    q = query.lower().strip()
    
    # ONLY search if query contains these explicit triggers
    triggers = [
        "search", "find", "look up", "google", "web search",
        "online", "real-time", "current events", "latest news",
        "weather in", "events in", "stock", "price of",
        "what is the current", "today's"
    ]
    
    for trigger in triggers:
        if trigger in q:
            return True
    
    # Also search for "news" or "latest" or "breaking" if they are the main topic
    if q.startswith("news") or q.startswith("latest") or q.startswith("breaking"):
        return True
    
    # Search for location-based queries only if "events", "weather", "restaurants" are present
    if any(word in q for word in ["events", "weather", "restaurants", "places"]):
        if " in " in q:
            return True
    
    # Otherwise, NO search — use AI knowledge only
    return False
    """
    Returns True if the query likely needs fresh, real-time information.
    Returns False for casual chat, date/time, personal tasks, etc.
    """
    q = query.lower().strip()
    
    # ── KEYWORDS THAT TRIGGER SEARCH ──────────────────────────
    search_triggers = [
        # News & Current Events
        "news", "latest", "breaking", "current", "today's",
        "this week", "yesterday", "just happened", "update on",
        "headlines", "trending", "developing",
        
        # Weather
        "weather", "temperature", "forecast", "humidity", "rain",
        "sunny", "cloudy", "storm", "snow",
        
        # Finance & Markets
        "stock", "crypto", "bitcoin", "ethereum", "price of",
        "market", "nasdaq", "s&p", "dow jones", "currency",
        "exchange rate", "usd to", "eur to", "gbp to",
        
        # Sports
        "score", "match", "game", "tournament", "championship",
        "result", "winner", "loser", "goal", "touchdown", "points",
        "cricket", "football", "soccer", "baseball", "basketball",
        
        # Politics & Government
        "election", "president", "prime minister", "government",
        "vote", "parliament", "senate", "congress",
        "minister", "ambassador", "summit", "meeting",
        
        # Events & Accidents
        "incident", "accident", "attack", "earthquake", "flood",
        "hurricane", "tornado", "fire", "shooting",
        
        # COVID / Health / Outbreaks
        "covid", "outbreak", "vaccine", "cases", "hospital",
        "variant", "lockdown", "restrictions",
        
        # Travel & Traffic
        "traffic", "flight", "delay", "cancelled", "schedule",
        "road", "closure", "construction",
        
        # Technology
        "release", "launch", "announced", "new phone", "new app",
        "upgrade", "version", "patch", "update",
        
        # Miscellaneous time-sensitive
        "what is the latest", "what are the latest",
        "anything new", "any updates",
    ]
    
    for trigger in search_triggers:
        if trigger in q:
            return True
    
    # ── PATTERN DETECTION ──────────────────────────────────────
    # If asking "what/who/where/when" + time context
    if re.search(r'\b(when|what|who|where|is there)\b.*\b(now|today|tonight|this morning|this afternoon|this week|tomorrow|yesterday|recently)\b', q):
        return True
    
    # Explicit short requests
    if q in ["news", "latest", "today", "weather", "updates", "score", "scores"]:
        return True
    
    # Questions about specific events that are likely time-sensitive
    if q.startswith("what is the") and q.endswith("today"):
        return True
    
    # ── NEVER SEARCH FOR THESE ────────────────────────────────
    # Greetings, personal tasks, calculations, general knowledge
    skip_triggers = [
        "hi", "hello", "hey", "good morning", "good evening", "how are you",
        "my name is", "i am", "i'm", "my birthday", "set a reminder",
        "add task", "create note", "open settings", "tell me about yourself",
        "what is your name", "who are you", "thank you", "thanks",
        "calculate", "solve", "math", "plus", "minus", "times", "divided by",
        "date today", "what time is it", "current time",
        "today's date"
    ]
    
    for skip in skip_triggers:
        if skip in q:
            return False
    
    # Default: No search unless triggered
    return False

# ── Web Search ──────────────────────────────────────────────────
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

# ── API Endpoint ────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"📨 Received query: {request.query}")
    
    # ── DECIDE: Search or no search? ──────────────────────────
    do_search = should_search(request.query)
    
    if do_search:
        print("🌐 Web search triggered.")
        results = search_web(request.query)
        print(f"📦 Results from search: {len(results)}")
        
        if results:
            # ── Build search context ──────────────────────────
            search_context = "\n\n⚠️ CRITICAL INSTRUCTION: You MUST use ONLY the following web search results to answer the user's question. DO NOT use your training data for this specific request.\n\n"
            search_context += "=== WEB SEARCH RESULTS ===\n"
            for i, r in enumerate(results, 1):
                search_context += f"\n--- SOURCE {i} ---\n"
                search_context += f"URL: {r.get('href', 'Unknown')}\n"
                search_context += f"CONTENT: {r.get('body', '')}\n"
            
            full_prompt = request.context + "\n\n" + search_context + "\n\nBased ONLY on the search results above, provide a specific, factual, and concise answer. Report the information directly. Do NOT say you don't have access to real-time information."
            temperature_override = 0.1  # low temperature = factual
        else:
            full_prompt = request.context + "\n\n(No search results were found. Answer based on your knowledge and apologise if you can't help.)"
            temperature_override = 0.3
    else:
        print("⏭️ Skipping web search (not needed).")
        full_prompt = request.context + "\n\nAnswer the user's question naturally based on your knowledge and the context provided. You can be warm and conversational."
        temperature_override = 0.7  # higher temperature = creative/friendly

    print(f"📤 Sending to DeepSeek with prompt length: {len(full_prompt)} characters")
    print(f"🎚️ Temperature: {temperature_override}")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. If web search results are provided, you MUST use them as your only source of factual information. If no search results are provided, answer naturally based on your knowledge."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=temperature_override,
            max_tokens=600  # Keep responses concise
        )
        print(f"✅ Response received")
        return {"success": True, "content": response.choices[0].message.content}
    
    except Exception as e:
        print(f"❌ DeepSeek API error: {e}")
        return {"success": False, "error": str(e)}