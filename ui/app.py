import gradio as gr
import requests

BACKEND = "http://backend:8000"

def generate_summary(text):
    res = requests.post(f"{BACKEND}/summarize", json={"text": text})
    return res.json()["summary"]

def search_summary(query):
    res = requests.post(f"{BACKEND}/search", json={"query": query})
    return res.json()

with gr.Blocks() as demo:

    gr.Markdown("Text Summarization App")

    with gr.Tab("Summarize Text"):
        input_text = gr.Textbox(lines=8, label="Enter Text")
        output_summary = gr.Textbox(lines=6, label="Summary")
        summarize_btn = gr.Button("Summarize")
        summarize_btn.click(generate_summary, inputs=input_text, outputs=output_summary)

    with gr.Tab("Semantic Search (ChromaDB)"):
        search_query = gr.Textbox(label="Search Query")
        search_output = gr.JSON(label="Search Results")
        search_btn = gr.Button("Search")
        search_btn.click(search_summary, inputs=search_query, outputs=search_output)
    with gr.Tab("Summary History"):
        history_output = gr.JSON(label="All Summaries")
        refresh_btn = gr.Button("Refresh History")
        refresh_btn.click(fn=lambda: requests.get(f"{BACKEND}/history").json(), inputs=[], outputs=history_output)


demo.launch(server_name="0.0.0.0", server_port=3000)
