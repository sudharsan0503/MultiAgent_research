"""
Streamlit UI for the multi-agent research pipeline.

Place this file in the SAME folder as agents.py and pipeline.py
(the root of your project, next to your .env), then run:

    streamlit run streamlit_app.py
"""

import streamlit as st
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(
    page_title="SentinelIQ — AI",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = {}
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app runs a 4-stage agent pipeline:\n\n"
        "1. **Search Agent** — finds recent, reliable sources\n"
        "2. **Reader Agent** — scrapes the most relevant page\n"
        "3. **Writer Chain** — drafts a report\n"
        "4. **Critic Chain** — reviews the report\n"
    )
    st.divider()
    if st.button("🗑️ Clear results", use_container_width=True):
        st.session_state.state = {}
        st.rerun()

# ---------------------------------------------------------------------------
# Header + input
# ---------------------------------------------------------------------------
st.title("🛡️ SentinelIQ — AI")
st.caption("Search → Read → Write → Critique, powered by your LangGraph/LangChain agents.")

with st.form("topic_form", clear_on_submit=False):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Latest developments in solid-state batteries",
    )
    submitted = st.form_submit_button(
        "Run research pipeline",
        use_container_width=True,
        disabled=st.session_state.is_running,
    )

# ---------------------------------------------------------------------------
# Pipeline execution (with live, per-step UI feedback)
# ---------------------------------------------------------------------------
if submitted:
    if not topic or not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.is_running = True
        state = {}

        # ---- Step 1: Search ----
        with st.status("Step 1/4 — Search agent is working...", expanded=True) as status:
            search_agent = build_search_agent()
            search_result = search_agent.invoke(
                {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
            )
            state["search_results"] = search_result["messages"][-1].content
            status.update(label="Step 1/4 — Search complete ✅", state="complete", expanded=False)

        # ---- Step 2: Read / scrape ----
        with st.status("Step 2/4 — Reader agent is scraping top sources...", expanded=True) as status:
            reader_agent = build_reader_agent()
            reader_result = reader_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Based on the following search results about '{topic}', "
                            f"pick the most relevant URL and scrape it for deeper content.\n\n"
                            f"Search Results:\n{state['search_results'][:800]}",
                        )
                    ]
                }
            )
            state["scraped_content"] = reader_result["messages"][-1].content
            status.update(label="Step 2/4 — Scraping complete ✅", state="complete", expanded=False)

        # ---- Step 3: Write ----
        with st.status("Step 3/4 — Writer is drafting the report...", expanded=True) as status:
            research_combined = (
                f"SEARCH RESULTS:\n{state['search_results']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
            status.update(label="Step 3/4 — Draft complete ✅", state="complete", expanded=False)

        # ---- Step 4: Critique ----
        with st.status("Step 4/4 — Critic is reviewing the report...", expanded=True) as status:
            state["feedback"] = critic_chain.invoke({"report": state["report"]})
            status.update(label="Step 4/4 — Review complete ✅", state="complete", expanded=False)

        state["topic"] = topic
        st.session_state.state = state
        st.session_state.is_running = False
        st.success("Pipeline finished! Scroll down for results.")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
state = st.session_state.state

if state:
    st.divider()
    st.header(f"Results for: {state.get('topic', '')}")

    report_tab, critique_tab, sources_tab, scraped_tab = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "📰 Scraped Content"]
    )

    with report_tab:
        st.markdown(state.get("report", "_No report generated._"))
        st.download_button(
            "Download report (.md)",
            data=str(state.get("report", "")),
            file_name=f"{state.get('topic', 'report').replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with critique_tab:
        st.markdown(state.get("feedback", "_No feedback generated._"))

    with sources_tab:
        st.markdown(state.get("search_results", "_No search results._"))

    with scraped_tab:
        st.markdown(state.get("scraped_content", "_No scraped content._"))
else:
    st.info("Enter a topic above and click **Run research pipeline** to get started.")