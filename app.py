import streamlit as st
import os
import re
import json
from dotenv import load_dotenv

import pdfplumber
from pydantic import BaseModel, Field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pinecone import Pinecone

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Resume Screening & ATS Assistant",
    page_icon="📄",
    layout="wide"
)

load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "resume-ats"

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing.")
    st.stop()
if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY missing.")
    st.stop()

# ════════════════════════════════════════════════════════════
# LLM & EMBEDDINGS
# ════════════════════════════════════════════════════════════

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding = get_embeddings()

# ════════════════════════════════════════════════════════════
# PINECONE
# ════════════════════════════════════════════════════════════

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

def get_vectorstore():
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY
    )

# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════

if "fresh_results"      not in st.session_state: st.session_state.fresh_results     = []
if "fresh_resume_data"  not in st.session_state: st.session_state.fresh_resume_data = {}
if "db_results"         not in st.session_state: st.session_state.db_results        = []
if "candidate_result"   not in st.session_state: st.session_state.candidate_result  = None
if "saved_to_db"        not in st.session_state: st.session_state.saved_to_db       = False

# ════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════

class JDRequirements(BaseModel):
    requirements: List[str] = Field(description="List of 5 key requirements from the job description")

# ════════════════════════════════════════════════════════════
# PARSERS
# ════════════════════════════════════════════════════════════

jd_parser  = PydanticOutputParser(pydantic_object=JDRequirements)
str_parser = StrOutputParser()

# ════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════

jd_prompt = PromptTemplate(
    template="""
Extract the top 5 most important requirements from the job description below.
{format_instructions}
Job Description:
{jd_text}
""",
    input_variables=["jd_text"],
    partial_variables={"format_instructions": jd_parser.get_format_instructions()}
)

suggestions_prompt = PromptTemplate(
    template="""
You are a professional resume coach.
Job Description:
{jd_text}
Candidate Resume:
{resume_text}
Missing or weak areas: {missing_reqs}
Give 4 specific, actionable suggestions to improve this resume for this job.
Format as a numbered list. Be direct and practical.
""",
    input_variables=["jd_text", "resume_text", "missing_reqs"]
)

# ════════════════════════════════════════════════════════════
# LCEL CHAINS  —  prompt | llm | parser
# ════════════════════════════════════════════════════════════

jd_chain          = jd_prompt          | llm | jd_parser
suggestions_chain = suggestions_prompt | llm | str_parser

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_jd_requirements(jd_text: str) -> list:
    try:
        result = jd_chain.invoke({"jd_text": jd_text})
        return result.requirements
    except Exception as e:
        st.error(f"Error extracting requirements: {e}")
        return []


def score_resume(candidate_name: str, resume_text: str,
                  requirements: list, vectorstore=None) -> dict:
    """
    Score a resume against JD requirements.
    If vectorstore provided — uses RAG retrieval.
    Otherwise — scores directly from resume text.
    """
    all_context = ""

    if vectorstore:
        # RAG — retrieve relevant chunks from Pinecone
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k":      3,
                "filter": {"candidate": {"$eq": candidate_name}}
            }
        )
        for req in requirements:
            try:
                docs    = retriever.invoke(req)
                context = " ".join([doc.page_content for doc in docs
                                    if doc.metadata.get("candidate") == candidate_name])
            except Exception:
                context = resume_text[:500]
            all_context += f"Requirement: {req}\nRelevant Resume Content: {context}\n\n"
    else:
        # Direct — use resume text directly (fresh check, not in DB yet)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks   = splitter.split_text(resume_text)
        for req in requirements:
            context = " ".join(chunks[:3])
            all_context += f"Requirement: {req}\nRelevant Resume Content: {context}\n\n"

    # Single LLM call — score all requirements
    bulk_prompt = f"""
You are an ATS scoring system. Score the resume against each job requirement.
{all_context}
Return a JSON array with one object per requirement:
[
  {{"requirement": "Python", "score": 9, "status": "Match", "reason": "Strong Python experience mentioned"}},
  {{"requirement": "Docker", "score": 0, "status": "Missing", "reason": "No Docker experience found"}}
]
Rules:
- score: 0-10
- status: exactly "Match", "Partial", or "Missing"
- Return ONLY the JSON array, no explanation, no markdown
"""
    requirement_scores = []
    total_score = 0

    try:
        raw   = llm.invoke(bulk_prompt).content.strip()
        raw   = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        items = json.loads(raw)
        for item in items:
            score  = int(item.get("score", 0))
            status = item.get("status", "Missing")
            reason = item.get("reason", "")
            requirement_scores.append({
                "requirement": item.get("requirement", ""),
                "score": score, "status": status, "reason": reason
            })
            total_score += score
    except Exception:
        for req in requirements:
            requirement_scores.append({
                "requirement": req, "score": 0,
                "status": "Missing", "reason": "Could not evaluate"
            })

    max_possible = len(requirements) * 10
    ats_score    = round((total_score / max_possible) * 100) if max_possible > 0 else 0

    return {"ats_score": ats_score, "requirement_scores": requirement_scores}


def save_to_pinecone(resume_data: dict):
    """Save multiple resumes to Pinecone DB."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for candidate_name, info in resume_data.items():
        resume_text = info["resume_text"]
        filename    = info["filename"]
        chunks      = splitter.split_text(resume_text)

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "candidate":   candidate_name,
                    "filename":    filename,
                    "chunk_id":    i,
                    "resume_text": resume_text[:2000]
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embedding,
            index_name=INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY
        )


def get_all_candidates() -> dict:
    """Fetch all unique candidates stored in Pinecone."""
    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        if total == 0:
            return {}

        dummy_vector = [0.0] * 384
        results      = index.query(
            vector=dummy_vector,
            top_k=min(total, 10000),
            include_metadata=True
        )

        candidates = {}
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            name = meta.get("candidate", "")
            if name and name not in candidates:
                candidates[name] = {
                    "filename":    meta.get("filename", ""),
                    "resume_text": meta.get("resume_text", "")
                }
        return candidates
    except Exception as e:
        st.error(f"Error fetching candidates: {e}")
        return {}


def delete_candidate(candidate_name: str):
    try:
        index = get_pinecone_index()
        index.delete(filter={"candidate": {"$eq": candidate_name}})
    except Exception as e:
        st.error(f"Error deleting: {e}")


def display_results(results: list):
    """Reusable results display component."""
    st.markdown("#### 📊 Score Overview")
    st.table({
        "Rank":           [i + 1 for i in range(len(results))],
        "Candidate":      [r["candidate"] for r in results],
        "ATS Score":      [f"{r['ats_score']}%" for r in results],
        "Recommendation": [
            "✅ Strong Match"   if r["ats_score"] >= 70
            else "⚠️ Moderate" if r["ats_score"] >= 40
            else "❌ Weak Match"
            for r in results
        ]
    })

    st.markdown("#### 🔍 Detailed Breakdown")
    for rank, result in enumerate(results, 1):
        with st.expander(
            f"#{rank}  {result['candidate']}  —  {result['ats_score']}%",
            expanded=(rank == 1)
        ):
            for r in result["requirement_scores"]:
                icon = "✅" if r["status"] == "Match" else "⚠️" if r["status"] == "Partial" else "❌"
                st.markdown(
                    f"{icon} **{r['requirement']}**  "
                    f"Score: {r['score']}/10  —  *{r['reason']}*"
                )


# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════

st.title("📄 Resume Screening & ATS Assistant")
st.caption("AI-powered resume analysis for recruiters and candidates.")

recruiter_tab, candidate_tab = st.tabs(["🏢 Recruiter Mode", "👤 Candidate Mode"])


# ════════════════════════════════════════════════════════════
# RECRUITER MODE
# ════════════════════════════════════════════════════════════
with recruiter_tab:

    st.subheader("🏢 Recruiter Mode")

    fresh_tab, db_tab = st.tabs(["🆕 Fresh Check", "🗄️ Search from Database"])

    # ── FRESH CHECK ──────────────────────────────────────────
    with fresh_tab:

        st.markdown("Upload resumes and paste a JD to screen candidates. Save to database after reviewing.")

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_resumes = st.file_uploader(
                "📂 Upload Resumes (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key="fresh_resumes"
            )

        with col2:
            jd_fresh = st.text_area(
                "📋 Paste Job Description",
                height=200,
                placeholder="Paste the full job description here...",
                key="fresh_jd"
            )

        screen_btn = st.button("🔍 Screen Resumes", use_container_width=True,
                                key="screen_btn")

        if screen_btn:
            if not uploaded_resumes:
                st.warning("⚠️ Please upload at least one resume.")
            elif not jd_fresh.strip():
                st.warning("⚠️ Please paste a job description.")
            else:
                st.session_state.fresh_results    = []
                st.session_state.fresh_resume_data = {}
                st.session_state.saved_to_db       = False

                with st.spinner("🔍 Extracting job requirements..."):
                    requirements = extract_jd_requirements(jd_fresh)

                if requirements:
                    st.info(f"📌 Requirements: **{', '.join(requirements)}**")

                progress = st.progress(0)
                results  = []
                resume_data = {}

                for i, resume_file in enumerate(uploaded_resumes):
                    candidate_name = resume_file.name.replace(".pdf", "")
                    st.text(f"⏳ Scoring {candidate_name}...")

                    resume_text = extract_text_from_pdf(resume_file)
                    if not resume_text:
                        st.warning(f"⚠️ Could not extract text from {resume_file.name}")
                        continue

                    # Score directly — not from DB yet
                    result = score_resume(candidate_name, resume_text, requirements)
                    result["candidate"] = candidate_name
                    results.append(result)

                    # Store resume data temporarily in session
                    resume_data[candidate_name] = {
                        "resume_text": resume_text,
                        "filename":    resume_file.name
                    }

                    progress.progress((i + 1) / len(uploaded_resumes))

                results.sort(key=lambda x: x["ats_score"], reverse=True)
                st.session_state.fresh_results     = results
                st.session_state.fresh_resume_data = resume_data
                st.success("✅ Screening complete!")

        # Show fresh results
        if st.session_state.fresh_results:
            st.divider()
            st.subheader("🏆 Results")
            display_results(st.session_state.fresh_results)

            st.divider()

            # Save to DB button
            if not st.session_state.saved_to_db:
                if st.button("💾 Save These Resumes to Database",
                              use_container_width=True, type="primary"):
                    with st.spinner("💾 Saving to Pinecone database..."):
                        save_to_pinecone(st.session_state.fresh_resume_data)
                    st.session_state.saved_to_db = True
                    st.success(
                        f"✅ {len(st.session_state.fresh_resume_data)} resume(s) "
                        f"saved to database! You can search them anytime from "
                        f"**Search from Database** tab."
                    )
            else:
                st.success("✅ Resumes already saved to database.")

    # ── SEARCH FROM DB ───────────────────────────────────────
    with db_tab:

        st.markdown("Search across all previously stored resumes using a job description.")

        candidates = get_all_candidates()

        if not candidates:
            st.warning("⚠️ No resumes in database yet. Use **Fresh Check** tab to screen and save resumes first.")
        else:
            st.info(f"📁 **{len(candidates)} resumes** stored in database")

            # Show stored candidates
            with st.expander("👥 View Stored Candidates"):
                for name, info in candidates.items():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"👤 **{name}**")
                    with col2:
                        st.caption(info.get("filename", ""))
                    with col3:
                        if st.button("🗑️", key=f"del_{name}", help=f"Remove {name}"):
                            delete_candidate(name)
                            st.success(f"Removed {name}")
                            st.rerun()

            st.divider()

            jd_db = st.text_area(
                "📋 Paste Job Description",
                height=200,
                placeholder="Paste the job description to search from database...",
                key="db_jd"
            )

            db_search_btn = st.button("🔍 Search Database", use_container_width=True,
                                       key="db_search_btn")

            if db_search_btn:
                if not jd_db.strip():
                    st.warning("⚠️ Please paste a job description.")
                else:
                    st.session_state.db_results = []

                    with st.spinner("🔍 Extracting job requirements..."):
                        requirements = extract_jd_requirements(jd_db)

                    if requirements:
                        st.info(f"📌 Requirements: **{', '.join(requirements)}**")

                    vectorstore = get_vectorstore()
                    progress    = st.progress(0)
                    results     = []

                    for i, (candidate_name, info) in enumerate(candidates.items()):
                        with st.spinner(f"Scoring {candidate_name}..."):
                            # RAG — retrieves from Pinecone
                            result = score_resume(
                                candidate_name,
                                info.get("resume_text", ""),
                                requirements,
                                vectorstore=vectorstore
                            )
                            result["candidate"] = candidate_name
                            results.append(result)
                        progress.progress((i + 1) / len(candidates))

                    results.sort(key=lambda x: x["ats_score"], reverse=True)
                    st.session_state.db_results = results
                    st.success("✅ Search complete!")

            if st.session_state.db_results:
                st.divider()
                st.subheader("🏆 Results from Database")
                display_results(st.session_state.db_results)


# ════════════════════════════════════════════════════════════
# CANDIDATE MODE
# ════════════════════════════════════════════════════════════
with candidate_tab:

    st.subheader("👤 Candidate — Check Your ATS Score")
    st.markdown("Upload your resume and paste a job description to get your ATS score and improvement tips.")

    col1, col2 = st.columns([1, 1])
    with col1:
        candidate_resume = st.file_uploader(
            "📂 Upload Your Resume (PDF)",
            type=["pdf"],
            key="candidate_resume"
        )
    with col2:
        jd_candidate = st.text_area(
            "📋 Paste Job Description",
            height=200,
            placeholder="Paste the job description you are applying for...",
            key="candidate_jd"
        )

    analyse_btn = st.button("🎯 Analyse My Resume", use_container_width=True,
                             key="analyse_btn")

    if analyse_btn:
        if not candidate_resume:
            st.warning("⚠️ Please upload your resume.")
        elif not jd_candidate.strip():
            st.warning("⚠️ Please paste a job description.")
        else:
            with st.spinner("📥 Reading your resume..."):
                resume_text = extract_text_from_pdf(candidate_resume)

            if not resume_text:
                st.error("❌ Could not extract text. Make sure it is a text-based PDF.")
            else:
                with st.spinner("🔍 Extracting job requirements..."):
                    requirements = extract_jd_requirements(jd_candidate)

                with st.spinner("🎯 Scoring your resume..."):
                    result = score_resume(
                        candidate_resume.name.replace(".pdf", ""),
                        resume_text,
                        requirements
                    )

                missing = [
                    r["requirement"]
                    for r in result["requirement_scores"]
                    if r["status"] == "Missing"
                ]

                with st.spinner("💡 Generating improvement tips..."):
                    suggestions = suggestions_chain.invoke({
                        "jd_text":      jd_candidate[:1000],
                        "resume_text":  resume_text[:1500],
                        "missing_reqs": ", ".join(missing) if missing else "None"
                    })

                st.session_state.candidate_result = {
                    "score_data":  result,
                    "suggestions": suggestions
                }

    if st.session_state.candidate_result:
        data        = st.session_state.candidate_result["score_data"]
        suggestions = st.session_state.candidate_result["suggestions"]
        ats_score   = data["ats_score"]

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Your ATS Score", f"{ats_score}%")
        with col2:
            if ats_score >= 70:
                st.success("✅ Strong Match")
            elif ats_score >= 40:
                st.warning("⚠️ Moderate Match")
            else:
                st.error("❌ Weak Match — Needs Improvement")
        with col3:
            matched   = sum(1 for r in data["requirement_scores"] if r["status"] == "Match")
            partial   = sum(1 for r in data["requirement_scores"] if r["status"] == "Partial")
            missing_c = sum(1 for r in data["requirement_scores"] if r["status"] == "Missing")
            st.metric("Requirements", f"✅ {matched}  ⚠️ {partial}  ❌ {missing_c}")

        st.divider()

        st.subheader("📋 Requirement Breakdown")
        for r in data["requirement_scores"]:
            icon = "✅" if r["status"] == "Match" else "⚠️" if r["status"] == "Partial" else "❌"
            st.markdown(
                f"{icon} **{r['requirement']}**  "
                f"— Score: {r['score']}/10  \n*{r['reason']}*"
            )

        st.divider()

        st.subheader("💡 How to Improve Your Resume")
        st.markdown(suggestions)