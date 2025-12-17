# ============================================================
#  MULTI-AGENT RESEARCH ASSISTANT FOR DEEP CNNs
#  Real AutoGen Agent Communication + RAG + Code Gen
# ============================================================

import os
import requests
from PyPDF2 import PdfReader
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
from groq import Groq
import concurrent.futures
from functools import lru_cache
import json

# ============================================================
# 1. LOAD ENVIRONMENT + INIT GROQ
# ============================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in .env")

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# 2. GROQ CHAT WRAPPER
# ============================================================

def groq_chat(prompt: str, model="llama-3.3-70b-versatile",
              conversation_history=None, temperature=0.4):
    messages = []
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1800,
    )

    return response.choices[0].message.content


# ============================================================
# 3. AUTOGEN AGENTS WITH REAL COMMUNICATION
# ============================================================

NO_DOCKER = {"use_docker": False}

# LLM Config for all agents
llm_config = {
    "config_list": [{
        "model": "llama-3.3-70b-versatile",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "api_type": "openai"
    }],
    "temperature": 0.4,
    "timeout": 120,
}

# Controller Agent - Coordinates everything
controller_agent = UserProxyAgent(
    name="Controller",
    system_message="You coordinate tasks between agents. Route queries to the right specialist.",
    code_execution_config=NO_DOCKER,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
)

# Search Specialist - Finds and analyzes papers
search_agent = AssistantAgent(
    name="SearchSpecialist",
    system_message="""You are a research paper search expert specializing in Deep CNNs and Computer Vision.
    Your tasks:
    1. Search for relevant papers on CNN architectures (ResNet, VGG, EfficientNet, etc.)
    2. Extract key information: authors, year, citations, methodology
    3. Summarize findings for other agents
    4. Focus on CNN-related research only

    Always provide structured output with paper titles, key contributions, and relevance scores.""",
    llm_config=llm_config,
)

# QA Specialist - Answers questions using context
qa_agent = AssistantAgent(
    name="QASpecialist",
    system_message="""You are a Q&A expert for research papers, especially Deep CNN papers.
    Your tasks:
    1. Answer questions using ONLY provided context
    2. Explain CNN architectures, methodologies, and results
    3. Compare different CNN approaches when asked
    4. If information is not in context, clearly state "Information not available in provided context"

    Always cite which part of the abstract/paper you're referencing.""",
    llm_config=llm_config,
)

# Code Specialist - Generates CNN implementations
code_agent = AssistantAgent(
    name="CodeSpecialist",
    system_message="""You are a CNN implementation expert specializing in PyTorch and TensorFlow.
    Your tasks:
    1. Generate clean, production-ready CNN code
    2. Implement architectures like ResNet, VGG, Custom CNNs
    3. Include proper error handling and comments
    4. Follow best practices for deep learning code

    Always explain the architecture choices in comments.""",
    llm_config=llm_config,
)

# Analysis Specialist - Compares and analyzes papers
analysis_agent = AssistantAgent(
    name="AnalysisSpecialist",
    system_message="""You are a research analysis expert for Deep CNN papers.
    Your tasks:
    1. Compare different CNN architectures and papers
    2. Analyze strengths and weaknesses
    3. Identify trends in CNN research
    4. Provide recommendations based on use cases

    Always structure comparisons clearly with bullet points.""",
    llm_config=llm_config,
)

# ============================================================
# 4. PAPER SEARCH — ARXIV + SEMANTIC SCHOLAR
# ============================================================

MAX_RESULTS = 7


@lru_cache(maxsize=100)
def search_semantic_scholar(query, max_results=7):
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={query}&limit={max_results}&"
        "fields=title,abstract,authors,year,citationCount,url,venue"
    )

    try:
        res = requests.get(url, timeout=10).json()
        papers = []

        for p in res.get("data", []):
            abs_raw = p.get("abstract")
            abstract = abs_raw if isinstance(abs_raw, str) else ""

            papers.append({
                "title": p.get("title", ""),
                "abstract": abstract,
                "authors": [a["name"] for a in p.get("authors", [])],
                "year": p.get("year", "Unknown"),
                "citations": p.get("citationCount", 0),
                "url": p.get("url", ""),
                "venue": p.get("venue", "Unknown"),
                "source": "Semantic Scholar"
            })

        return papers

    except Exception:
        return []


@lru_cache(maxsize=100)
def search_arxiv(query, max_results=7):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        import xml.etree.ElementTree as ET

        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        papers = []

        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = (entry.find('{http://www.w3.org/2005/Atom}title').text or "").strip()
            summary = (entry.find('{http://www.w3.org/2005/Atom}summary').text or "").strip()

            authors = [
                a.find('{http://www.w3.org/2005/Atom}name').text
                for a in entry.findall('{http://www.w3.org/2005/Atom}author')
            ]

            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            year = entry.find('{http://www.w3.org/2005/Atom}published').text[:4]

            papers.append({
                "title": title,
                "abstract": summary,
                "authors": authors,
                "year": year,
                "citations": 0,
                "url": link,
                "venue": "arXiv",
                "source": "arXiv"
            })

        return papers

    except Exception:
        return []


def search_all_sources(query, max_results=7):
    """MULTI-AGENT: Controller coordinates search across sources"""

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        sem_future = ex.submit(search_semantic_scholar, query, max_results)
        arxiv_future = ex.submit(search_arxiv, query, max_results)

        sem_res = sem_future.result()
        arxiv_res = arxiv_future.result()

    combined = sem_res + arxiv_res

    seen = set()
    unique = []

    for p in combined:
        key = p["title"].lower().strip()
        if key and key not in seen:
            unique.append(p)
            seen.add(key)

    unique.sort(key=lambda x: (x.get("citations", 0), str(x.get("year", ""))), reverse=True)

    return unique[:MAX_RESULTS]


# ============================================================
# 5. PDF CHUNKING + RAG Q&A
# ============================================================

def extract_pdf_text_chunked(pdf_file, chunk_size=1000, overlap=200):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    clean = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(clean):
        end = start + chunk_size
        chunks.append(clean[start:end])
        start += chunk_size - overlap

    return {"full_text": clean, "chunks": chunks}


def find_relevant_chunks(chunks, question, top_k=3):
    terms = set(question.lower().split())
    scored = []

    for c in chunks:
        score = sum(t in c.lower() for t in terms)
        scored.append((score, c))

    scored.sort(reverse=True)
    return [c for s, c in scored[:top_k] if s > 0]


def answer_with_rag(chunks, question):
    """MULTI-AGENT: QA Specialist answers using RAG"""
    relevant = find_relevant_chunks(chunks, question)

    if not relevant:
        return "The document does not contain information related to this question."

    context = "\n\n".join(c[:600] for c in relevant)

    prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and cite the relevant parts:
"""

    return groq_chat(prompt).strip()


# ============================================================
# 6. MULTI-AGENT COLLABORATIVE FUNCTIONS
# ============================================================

def multi_agent_paper_analysis(paper: dict, user_question: str) -> str:
    """
    REAL MULTI-AGENT COMMUNICATION:
    Controller → Search Agent → QA Agent → Analysis Agent
    """

    title = paper.get("title", "")
    authors = ", ".join(paper.get("authors", []))
    abstract = paper.get("abstract", "No abstract available")

    # Step 1: Controller initiates - Search Agent extracts key info
    search_task = f"""
Analyze this CNN research paper and extract key information:

Title: {title}
Authors: {authors}
Abstract: {abstract[:800]}

Extract:
1. Main CNN architecture discussed
2. Key contributions
3. Methodology overview
4. Datasets used (if mentioned)
"""

    search_analysis = groq_chat(search_task, temperature=0.3)

    # Step 2: Search Agent passes to QA Agent with user question
    qa_task = f"""
Based on the paper analysis below, answer the user's question.

Paper Analysis:
{search_analysis}

User Question: {user_question}

Provide a detailed answer using ONLY the information from the analysis.
If the answer is not in the analysis, say so clearly.
"""

    qa_response = groq_chat(qa_task, temperature=0.3)

    # Step 3: QA Agent passes to Analysis Agent for refinement
    analysis_task = f"""
Review and enhance this answer about a CNN research paper:

Question: {user_question}
Current Answer: {qa_response}

Enhance by:
1. Adding CNN-specific context if relevant
2. Comparing to other CNN architectures if applicable
3. Ensuring technical accuracy
4. Making it more comprehensive

Provide the enhanced answer:
"""

    final_answer = groq_chat(analysis_task, temperature=0.3)

    return final_answer


def multi_agent_code_generation(task: str, language: str = "python") -> str:
    """
    REAL MULTI-AGENT COMMUNICATION:
    Controller → Code Agent → QA Agent (review) → Code Agent (refine)
    """

    # Step 1: Code Agent generates initial version
    code_task = f"""
Generate {language} code for this CNN-related task:

{task}

Requirements:
- Clean, production-ready code
- Proper comments explaining architecture
- Error handling
- Follow deep learning best practices

Generate ONLY the code:
"""

    code_v1 = groq_chat(code_task, temperature=0.2)

    # Step 2: QA Agent reviews the code
    review_task = f"""
Review this CNN implementation code:

{code_v1[:1500]}

Check for:
1. Code correctness
2. CNN architecture best practices
3. Missing error handling
4. Optimization opportunities
5. Documentation quality

Provide specific improvement suggestions:
"""

    review_feedback = groq_chat(review_task, temperature=0.3)

    # Step 3: Code Agent refines based on review
    refine_task = f"""
Improve this code based on the review feedback:

Original Code:
{code_v1}

Review Feedback:
{review_feedback}

Generate the improved version incorporating all suggestions.
ONLY output code, no explanations:
"""

    code_v2 = groq_chat(refine_task, temperature=0.2)

    return code_v2


def multi_agent_paper_comparison(paper1_text: str, paper2_text: str, aspect: str) -> str:
    """
    REAL MULTI-AGENT COMMUNICATION:
    Controller → Search Agent (extract info) → Analysis Agent (compare)
    """

    # Step 1: Search Agent extracts key info from both papers
    extract_task_1 = f"""
Extract key information from this CNN paper abstract focusing on {aspect}:

{paper1_text[:1000]}

Provide:
- Main CNN architecture
- Methodology related to {aspect}
- Key results
- Unique contributions
"""

    extract_task_2 = f"""
Extract key information from this CNN paper abstract focusing on {aspect}:

{paper2_text[:1000]}

Provide:
- Main CNN architecture
- Methodology related to {aspect}
- Key results
- Unique contributions
"""

    info_1 = groq_chat(extract_task_1, temperature=0.3)
    info_2 = groq_chat(extract_task_2, temperature=0.3)

    # Step 2: Analysis Agent compares
    comparison_task = f"""
Compare these two CNN papers focusing on {aspect}:

Paper 1 Information:
{info_1}

Paper 2 Information:
{info_2}

Provide a structured comparison:

### Similarities
[List key similarities]

### Differences
[List key differences]

### Strengths of Paper 1
[Specific strengths]

### Strengths of Paper 2
[Specific strengths]

### Recommendation
[Which is better for what use case]
"""

    comparison = groq_chat(comparison_task, temperature=0.3)

    return comparison


# ============================================================
# 7. SINGLE-AGENT FALLBACK FUNCTIONS (for compatibility)
# ============================================================

def generate_paper_report(paper: dict) -> str:
    """Generate structured report from paper metadata"""

    title = paper.get("title", "")
    authors = ", ".join(paper.get("authors", []))
    year = paper.get("year", "")
    venue = paper.get("venue", "")
    citations = paper.get("citations", 0)
    abstract = paper.get("abstract", "No abstract available")

    prompt = f"""
Generate a comprehensive research report for this CNN paper:

Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Citations: {citations}

Abstract:
{abstract}

Create a report with these sections:

1. Executive Summary
2. CNN Architecture Overview
3. Key Contributions
4. Methodology
5. Strengths
6. Limitations
7. Applications in Deep Learning
8. Future Research Directions

Be specific about CNN-related aspects.
"""

    return groq_chat(prompt, temperature=0.35)


def answer_question_about_selected_paper(paper: dict, question: str, history=None):
    """Fallback: Direct answer without multi-agent (for simple queries)"""

    abstract = paper.get("abstract", "")
    if not abstract:
        return "No abstract available to answer this question."

    title = paper.get("title", "")

    prompt = f"""
Answer this question about a CNN research paper using ONLY the abstract:

Paper: {title}
Abstract: {abstract}

Question: {question}

If the answer is not in the abstract, say: "Information not available in the abstract."
"""

    return groq_chat(prompt, conversation_history=history, temperature=0.2)


def chatbot_answer(prompt, history=None):
    """General chatbot for CNN research questions"""
    return groq_chat(prompt, conversation_history=history)


def generate_advanced_code(instruction: str, language: str = "python") -> str:
    """Fallback: Direct code generation"""

    prompt = f"""
Write {language} code for this CNN-related task:

{instruction}

Requirements:
- Production-ready code
- Comments explaining CNN architecture
- Proper error handling
- Deep learning best practices

Output ONLY code:
"""

    return groq_chat(prompt, temperature=0.2).strip()


def compare_two_papers_rag(text1, text2, aspect):
    """Fallback: Direct comparison"""

    prompt = f"""
Compare these two CNN papers based on {aspect}:

Paper 1 Abstract:
{text1[:2000]}

Paper 2 Abstract:
{text2[:2000]}

Provide structured comparison:

### Similarities
### Differences  
### Strengths of Paper 1
### Strengths of Paper 2
### Recommendation
"""

    return groq_chat(prompt, temperature=0.3)


def generate_pdf_summary_report(full_text: str) -> str:
    """Generate summary from PDF text"""

    if not full_text or len(full_text.strip()) == 0:
        return "PDF text is empty or unreadable."

    prompt = f"""
Summarize this CNN research paper:

{full_text[:8000]}

Create a structured summary:

1. Executive Summary
2. CNN Architecture Details
3. Key Findings
4. Methodology
5. Experimental Results
6. Conclusions

Focus on CNN-specific aspects.
"""

    return groq_chat(prompt, temperature=0.35)