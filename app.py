import streamlit as st
from research_assistant import (
    search_semantic_scholar,
    generate_fast_report,
    answer_question_about_paper,
    extract_pdf_text,
    generate_report_from_pdf,
    answer_question_about_pdf,
    generate_code,
)

# ============================================
#           STREAMLIT PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    page_icon="🤖",
)

st.title("🤖 AI Research Assistant")
st.write(
    "A simple demo of a **multi-agent research system** using:\n"
    "- Search Agent (Semantic Scholar)\n"
    "- Reporter Agent (summaries & reports)\n"
    "- PDF Agent (text extraction & reporting)\n"
    "- Q&A Agent (question answering)\n"
    "- Code Generation Agent (Groq LLaMA 3.3 70B)"
)
st.markdown("---")

# ============================================
#                    TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Search Papers",
        "📄 PDF Upload",
        "❓ Q&A (Paper/PDF)",
        "💻 Code Generator",
    ]
)

# =========================================================
# TAB 1 — SEARCH PAPERS
# =========================================================
with tab1:
    st.header("🔍 Search Research Papers (Search Agent)")

    topic = st.text_input(
        "Enter a research topic:",
        placeholder="e.g., CNN, Transformers, Reinforcement Learning",
        key="search_topic",
    )

    if st.button("Search", key="search_button"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Searching Semantic Scholar (Search Agent)..."):
                papers = search_semantic_scholar(topic)

            if not papers:
                st.error("No papers found.")
            else:
                st.session_state["papers"] = papers
                st.success(f"Found {len(papers)} papers!")

    if "papers" in st.session_state:
        papers = st.session_state["papers"]
        titles = [f"{i+1}. {p['title']}" for i, p in enumerate(papers)]

        index = st.selectbox(
            "Select a paper:", range(len(papers)), format_func=lambda x: titles[x]
        )
        paper = papers[index]

        st.subheader("📘 Paper Details")
        st.write(f"**Title:** {paper['title']}")
        st.write(f"**Authors:** {', '.join(paper['authors'])}")
        if paper["url"]:
            st.write(f"[🔗 View Paper]({paper['url']})")

        if st.button("📝 Generate Report (Reporter Agent)", key="report_button"):
            with st.spinner("Reporter Agent generating report via Groq..."):
                report = generate_fast_report(paper)

            st.subheader("📄 Research Report")
            st.markdown(report)


# =========================================================
# TAB 2 — PDF UPLOAD
# =========================================================
with tab2:
    st.header("📄 Upload a Research PDF (PDF Agent + Reporter Agent)")

    pdf = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")

    if pdf:
        with st.spinner("PDF Agent extracting text..."):
            pdf_text = extract_pdf_text(pdf)

        st.session_state["pdf_text"] = pdf_text
        st.success("PDF text extracted successfully!")

        if st.button("📝 Generate Report from PDF", key="pdf_report_button"):
            with st.spinner("Reporter Agent generating PDF-based report..."):
                pdf_report = generate_report_from_pdf(pdf_text)

            st.subheader("📄 Research Report (From PDF)")
            st.markdown(pdf_report)


# =========================================================
# TAB 3 — Q&A (Paper / PDF)
# =========================================================
with tab3:
    st.header("❓ Q&A Agent")

    qa_mode = st.radio(
        "Choose context for Q&A:",
        ["Searched Paper Abstract", "Uploaded PDF Text"],
        key="qa_mode",
    )

    if qa_mode == "Searched Paper Abstract":
        if "papers" not in st.session_state:
            st.info("Please search for a paper first in the 'Search Papers' tab.")
        else:
            papers = st.session_state["papers"]
            titles = [f"{i+1}. {p['title']}" for i, p in enumerate(papers)]
            idx = st.selectbox(
                "Select paper for Q&A:",
                range(len(papers)),
                format_func=lambda x: titles[x],
                key="qa_paper_select",
            )
            selected_paper = papers[idx]

            question = st.text_input(
                "Your question about this paper:",
                placeholder="e.g., What is the main contribution?",
                key="paper_question",
            )

            if st.button("Ask (Paper)", key="ask_paper_button"):
                if not question.strip():
                    st.warning("Enter a question.")
                else:
                    with st.spinner("Q&A Agent answering from abstract..."):
                        answer = answer_question_about_paper(selected_paper, question)

                    st.subheader("🧠 Answer")
                    st.write(answer)

    else:  # Uploaded PDF Text
        if "pdf_text" not in st.session_state:
            st.info("Please upload a PDF first in the 'PDF Upload' tab.")
        else:
            pdf_question = st.text_input(
                "Your question about the uploaded PDF:",
                placeholder="e.g., What dataset is used? What is the main result?",
                key="pdf_question",
            )

            if st.button("Ask (PDF)", key="ask_pdf_button"):
                if not pdf_question.strip():
                    st.warning("Enter a question.")
                else:
                    with st.spinner("Q&A Agent answering from PDF text..."):
                        answer = answer_question_about_pdf(
                            st.session_state["pdf_text"], pdf_question
                        )

                    st.subheader("🧠 Answer")
                    st.write(answer)


# =========================================================
# TAB 4 — CODE GENERATOR
# =========================================================
with tab4:
    st.header("💻 Code Generation Agent")

    code_instruction = st.text_area(
        "Describe the code you want:",
        placeholder=(
            "Examples:\n"
            "- Write a Python function to implement bubble sort.\n"
            "- Generate a PyTorch CNN model for CIFAR-10.\n"
            "- Create a FastAPI endpoint that returns JSON.\n"
        ),
        key="code_instruction",
        height=200,
    )

    if st.button("Generate Code", key="generate_code_button"):
        if not code_instruction.strip():
            st.warning("Please describe the code you want.")
        else:
            with st.spinner("Code Agent generating code via Groq..."):
                code_result = generate_code(code_instruction)

            st.subheader("🧠 Generated Code")
            # Just show whatever the model returns; it may already contain ``` fences
            st.code(code_result, language="python")
