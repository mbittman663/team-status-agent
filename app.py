import streamlit as st
from main import run_agent  # import your function

st.set_page_config(page_title="Agent UI", layout="centered")

st.title("🧠 LangGraph Agent")
st.write("Ask anything and the agent will process it step-by-step.")

# Input box
user_input = st.text_area("Enter your task:")

# Run button
if st.button("Run Agent"):
    if user_input.strip():
        with st.spinner("Running agent..."):
            result = run_agent(user_input)

        st.subheader("✅ Final Output")
        st.write(result["output"])

        # Debug (expandable)
        with st.expander("🛠 Debug Steps"):
            for step in result["history"]:
                st.text(step)
    else:
        st.warning("Please enter a task.")
