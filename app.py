import streamlit as st
from research_assistant import (
    search_semantic_scholar,
    generate_fast_report,
    answer_question_about_paper,
    extract_pdf_text,
    generate_report_from_pdf
)

# =====================================================
#                    PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    page_icon="🤖"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #4DB6AC;
        text-align: center;
        padding-bottom: 20px;
    }
    .sub-section-title {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #81D4FA;
        padding-top: 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
#                    MAIN TITLE
# =====================================================
st.markdown("<div class='main-title'>🤖 AI Research Paper Assistant</div>", unsafe_allow_html=True)
st.write("A multi-agent research assistant powered by **Semantic Scholar**, **Ollama LLMs**, and **AutoGen Agents**.")

# =====================================================
#                    TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔎 Search Papers", "        📄 PDF Upload", "       ❓ Q&A"])

# =====================================================
#              TAB 1 — SEARCH PAPERS
# =====================================================
with tab1:

    st.markdown("<div class='sub-section-title'>🔎 Search Research Papers</div>", unsafe_allow_html=True)

    query = st.text_input("Enter a research topic:", placeholder="e.g., CNN, Transformers, Quantum ML...")

    if st.button("Search Papers"):
        if not query.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("🔍 Search Agent is searching Semantic Scholar..."):
                papers = search_semantic_scholar(query)

            if not papers:
                st.error("No papers found for this topic.")
            else:
                st.session_state["papers"] = papers
                st.success(f"Found {len(papers)} relevant papers!")


    # -------------------------- Results --------------------------
    if "papers" in st.session_state:

        papers = st.session_state["papers"]

        st.markdown("<div class='sub-section-title'>📄 Select a Paper</div>", unsafe_allow_html=True)

        paper_titles = [f"{i+1}. {p['title']}" for i, p in enumerate(papers)]

        selected_index = st.selectbox("Choose a paper:", range(len(papers)),
                                      format_func=lambda x: paper_titles[x])

        selected_paper = papers[selected_index]

        # Paper Info
        st.write("### 📘 Paper Details")
        st.write(f"**Title:** {selected_paper['title']}")
        st.write(f"**Authors:** {', '.join(selected_paper['authors'])}")
        if selected_paper["url"]:
            st.write(f"[🔗 View Paper]({selected_paper['url']})")

        # -------------------------- Report Button --------------------------
        if st.button("📝 Generate Research Report"):
            with st.spinner("🧠 Reporter Agent is generating the report..."):
                report = generate_fast_report(selected_paper)

            st.subheader("📑 Research Report")
            st.markdown(report)


# =====================================================
#              TAB 2 — PDF UPLOAD
# =====================================================
with tab2:

    st.markdown("<div class='sub-section-title'>📄 Upload Research PDF</div>", unsafe_allow_html=True)

    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if pdf_file is not None:
        with st.spinner("📘 PDF Agent is extracting text..."):
            pdf_text = extract_pdf_text(pdf_file)

        st.success("PDF text extracted successfully! 🎉")

        if st.button("📝 Generate Report from PDF"):
            with st.spinner("🧠 Reporter Agent is generating the report..."):
                pdf_report = generate_report_from_pdf(pdf_text)

            st.subheader("📑 Research Report (From PDF)")
            st.markdown(pdf_report)


# =====================================================
#              TAB 3 — QUESTION ANSWERING
# =====================================================
with tab3:

    st.markdown("<div class='sub-section-title'>❓ Ask a Question About a Paper</div>", unsafe_allow_html=True)

    if "papers" not in st.session_state:
        st.info("🔍 Please search for a paper first in the 'Search Papers' tab.")
    else:
        selected_paper = st.session_state["papers"][0]   # default first paper
        question = st.text_input("Enter your question:", placeholder="e.g., What dataset was used?")

        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a valid question.")
            else:
                with st.spinner("💬 Q&A Agent is answering..."):
                    answer = answer_question_about_paper(selected_paper, question)

                st.subheader("🧠 Answer")
                st.write(answer)
