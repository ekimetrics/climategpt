import gradio as gr
from transformers import pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import numpy as np
import openai

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

system_template = {
    "role": "system",
    "content": "You have been a climate change expert for 30 years. You answer questions about climate change in an educationnal and concise manner.",
}


document_store = FAISSDocumentStore.load(
    index_path=f"./climate_gpt.faiss",
    config_path=f"./climate_gpt.json",
)
dense = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
)


def is_climate_change_related(sentence: str) -> bool:
    results = classifier(
        sequences=sentence,
        candidate_labels=["climate change related", "non climate change related"],
    )
    return results["labels"][np.argmax(results["scores"])] == "climate change related"


def make_pairs(lst):
    """from a list of even lenght, make tupple pairs"""
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]


def gen_conv(query: str, history=[system_template], ipcc=True):
    """return (answer:str, history:list[dict], sources:str)"""
    retrieve = ipcc and is_climate_change_related(query)
    sources = ""
    messages = history + [
        {"role": "user", "content": query},
    ]

    if retrieve:
        docs = dense.retrieve(query=query, top_k=5)
        sources = "\n\n".join(
            ["If relevant, use those extracts from IPCC reports in your answer"]
            + [
                f"{d.meta['path']} Page {d.meta['page_id']} paragraph {d.meta['paragraph_id']}:\n{d.content}"
                for d in docs
            ]
        )
        messages.append({"role": "system", "content": sources})

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        #         max_tokens=200,
    )["choices"][0]["message"]["content"]

    if retrieve:
        messages.pop()
        answer = "(top 5 documents retrieved) " + answer
        sources = "\n\n".join(
            f"{d.meta['path']} Page {d.meta['page_id']} paragraph {d.meta['paragraph_id']}:\n{d.content[:100]} [...]"
            for d in docs
        )

    messages.append({"role": "assistant", "content": answer})

    gradio_format = make_pairs([a["content"] for a in messages[1:]])

    return gradio_format, messages, sources


def connect(text):
    openai.api_key = text
    return "You're all set"


with gr.Blocks(title="Eki IPCC Explorer") as demo:
    with gr.Row():
        with gr.Column():
            api_key = gr.Textbox(label="Open AI api key")
            connect_btn = gr.Button(value="Connect")
        with gr.Column():
            result = gr.Textbox(label="Connection")

    connect_btn.click(connect, inputs=api_key, outputs=result, api_name="Connection")

    gr.Markdown(
        """
        # Ask me anything, I'm an IPCC report
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            state = gr.State([system_template])

            with gr.Row():
                ask = gr.Textbox(
                    show_label=False, placeholder="Enter text and press enter"
                ).style(container=False)

        with gr.Column(scale=1, variant="panel"):

            gr.Markdown("### Sources")
            sources_textbox = gr.Textbox(
                interactive=False, show_label=False, max_lines=50
            )

    ask.submit(
        fn=gen_conv, inputs=[ask, state], outputs=[chatbot, state, sources_textbox]
    )

demo.launch(share=True)
