import streamlit as st  # pyright: ignore[reportMissingImports]

from step10 import app

st.set_page_config(page_title="ResearchPilot", page_icon="🔎")
st.title("🔎 ResearchPilot")
st.caption("Agentic RAG over arXiv papers · Qdrant hybrid search · LangGraph")

question = st.chat_input("Ask about RAG research, or anything else…")
if question:
    st.chat_message("user").write(question)
    state = {}
    with st.chat_message("assistant"):
        with st.status("Agent working…", expanded=True) as status:
            for update in app.stream({"question": question}, stream_mode="updates"):
                for delta in update.values():
                    for line in delta.get("trace", []):
                        st.write(f"`{line}`")
                    state.update({k: v for k, v in delta.items() if k != "trace"})
            status.update(label="Done", state="complete", expanded=False)
        st.markdown(state.get("text", "No answer produced."))
        docs = [d for d in state.get("docs", []) if d.get("url")]
        if docs:
            with st.expander("Sources"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"{i}. [{d['title']}]({d['url']})")
