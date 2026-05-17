from typing import Any, Dict, List
import streamlit as st
from core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Helper function to format sources for display."""
    return [
        str((meta.get("source") or "Unknown Source"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper", page_icon="📚", layout="centered")
st.title("📚 LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role" : "assistant",
            "content" : "Ask me anything about LangChain. I'll retrieve relevant context and cite sources",
            "sources" : [],
        }
    ]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("sources"):
                for s in msg.get("sources"):
                    st.markdown(f"- {s}")


prompt = st.chat_input("Aks a question about LangChain...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answers..."):
                result : Dict[str,Any] =run_llm(prompt)
                answer = str(result.get("answer","")).strip() or "(No answer returned.)"

                sources =_format_sources(result.get("context",[]))
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role" : "assistant", "content": answer, "sources" :sources}
            )


        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)


