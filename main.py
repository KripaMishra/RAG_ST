import streamlit as st
import json
from datetime import datetime
from typing import Dict
from components.s5_RAG_LLM import RAGModel

# Initialize the RAG model
model_name = "gpt2-xl"  # Replace with your preferred model
rag_model = RAGModel(model_name)

def main():
    st.title("RAG Application")

    # User input
    query = st.text_area("Enter your query:", height=100)
    
    if st.button("Generate Answer"):
        if query:
            with st.spinner("Generating answer..."):
                result = rag_model.process_query(query)
                
                # Save the results
                rag_model.save_query_results(result, file_path)

                # Display the results
                st.subheader("Results:")
                st.write(f"**Original Query:** {result['original_query']}")
                st.write(f"**Timestamp:** {result['timestamp']}")
                
                st.subheader("Context:")
                st.json(result['context'])
                
                st.subheader("Answer:")
                st.write(result['answer'])

                # Option to download results as JSON
                st.download_button(
                    label="Download Results as JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
