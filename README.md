ptoject live link : https://huggingface.co/spaces/Harshith945/resume-ats-assistant



# 📄 Resume Screening & ATS Assistant

An AI-powered Resume Screening and ATS (Applicant Tracking System) Assistant built using **Python, Streamlit, LangChain, Groq Llama 3.1, Pinecone, HuggingFace Embeddings, and Retrieval-Augmented Generation (RAG)**. The application helps recruiters efficiently screen and rank candidates while also helping job seekers improve their resumes through AI-generated recommendations.

---

## 🚀 Problem Statement

Recruiters often receive hundreds of resumes for a single job opening, making manual screening a time-consuming and error-prone process. Additionally, candidates may not know how well their resumes align with specific job requirements. This project addresses these challenges by automating resume analysis, ATS scoring, candidate ranking, and resume improvement suggestions using Generative AI and RAG.

---

## 🎯 Objectives

- Extract key requirements from Job Descriptions (JDs)
- Evaluate resumes against job requirements
- Generate ATS scores automatically
- Rank candidates based on suitability
- Store resumes in a vector database
- Perform semantic search using RAG
- Generate resume improvement suggestions

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Generative AI
- LangChain
- Groq
- Llama 3.1 8B Instant

### Vector Database
- Pinecone

### Embedding Model
- sentence-transformers/all-MiniLM-L6-v2

### Document Processing
- pdfplumber

### Structured Output
- Pydantic Output Parser

### RAG Components
- LangChain Retriever
- Pinecone Vector Search

---

## 🏗️ System Architecture

```text
Resume PDF
     │
     ▼
PDF Text Extraction
(pdfplumber)
     │
     ▼
Text Chunking
(RecursiveCharacterTextSplitter)
     │
     ▼
Embeddings
(all-MiniLM-L6-v2)
     │
     ▼
Pinecone Vector Database
     │
     ▼
Retriever (RAG)
     │
     ▼
Llama 3.1 via Groq
     │
     ▼
ATS Score + Ranking + Suggestions
```

---

## 🔄 Workflow

### Recruiter Mode

1. Upload multiple resumes in PDF format.
2. Paste a Job Description.
3. Extract top job requirements using Llama 3.1.
4. Compare resumes against requirements.
5. Generate ATS scores.
6. Rank candidates automatically.
7. Save resumes to Pinecone Vector Database.
8. Search stored resumes using RAG.

### Candidate Mode

1. Upload a resume.
2. Paste a target Job Description.
3. Extract job requirements.
4. Generate ATS score.
5. Identify missing skills.
6. Generate personalized improvement suggestions.

---

## 🤖 RAG Implementation

The project uses Retrieval-Augmented Generation (RAG) when searching resumes stored in Pinecone.

### Process

1. Resume text is extracted and split into chunks.
2. Chunks are converted into embeddings.
3. Embeddings are stored in Pinecone.
4. Job requirements are used as retrieval queries.
5. Relevant resume chunks are retrieved.
6. Retrieved context is passed to Llama 3.1.
7. ATS scores are generated based on retrieved information.

### Benefits

- Semantic search capability
- Faster retrieval
- Reduced token usage
- Improved scoring accuracy
- Scalable candidate search

---


## 📊 ATS Scoring Logic

The system evaluates resumes against extracted job requirements.

Each requirement is categorized as:

- ✅ Match
- ⚠️ Partial Match
- ❌ Missing

Final ATS Score:

```text
ATS Score = (Total Requirement Scores / Maximum Possible Score) × 100
```

---

## ✨ Features

### Recruiter Features

- Multi-resume upload
- Automated ATS scoring
- Candidate ranking
- Resume database management
- Semantic candidate search
- RAG-powered retrieval

### Candidate Features

- ATS score calculation
- Requirement breakdown
- Missing skill identification
- AI-generated improvement suggestions

---

## 💡 Future Enhancements

- DOCX Resume Support
- Resume Summarization
- Interview Question Generation
- Skill Gap Analysis
- Recruiter Analytics Dashboard
- Multi-language Support
- Email Notifications

---

## 🎓 Learning Outcomes

This project provided hands-on experience with:

- Generative AI Applications
- LangChain Framework
- Prompt Engineering
- Pydantic Output Parsing
- Vector Databases
- Pinecone Integration
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Streamlit Development
- LLM-Based ATS Systems
