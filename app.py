import gradio as gr
from openai import OpenAI

# <-- CHANGE THIS -->
CLIENT = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="none")

def parse_thought_and_response(full_text):
    """
    Helper function to separate the model's internal reasoning from the final answer.
    """
    if "<|channel>thought" in full_text and "<channel|>" in full_text:
        # Split the text around the closing tag
        parts = full_text.split("<channel|>")

        # Clean up the thought process
        thought_process = parts[0].replace("<|channel>thought", "").strip()

        # The final answer is everything after the closing tag
        final_answer = parts[1].strip()

        return thought_process, final_answer

    # Fallback if the model skipped reasoning
    return "No explicit thought process detected.", full_text.strip()

def generate(
    system_prompt, 
    user_query, 
    max_tokens, 
    temperature, 
    top_p, 
    frequency_penalty, 
    presence_penalty,
    enable_thinking
):
    # Enable thinking by injecting the token at the start of the system instructions
    thinking_system_prompt = f"<|think|>\n{system_prompt}" if enable_thinking else system_prompt

    response = CLIENT.chat.completions.create(
        model="Qwen/Qwen2.5-14B-Instruct",
        messages=[
            {"role": "system", "content": thinking_system_prompt},
            {"role": "user", "content": user_query},
        ],
        # Dynamically inject the UI parameters here
        max_tokens=int(max_tokens), 
        temperature=float(temperature),
        top_p=float(top_p),
        frequency_penalty=float(frequency_penalty),
        presence_penalty=float(presence_penalty)
    )

    full_response = response.choices[0].message.content
    return parse_thought_and_response(full_response)


with gr.Blocks() as demo:
    gr.Markdown("## google/gemma-4-26B-A4B-it (Thinking Enabled)")

    with gr.Row():
        with gr.Column(scale=1):
            system = gr.Textbox(
                label="System Prompt",
                value="System prompt for model to extract triplets...",
                lines=3
            )
            query = gr.Textbox(label="User Query", lines=5)
            
            # Put advanced parameters inside an accordion to keep the UI clean
            with gr.Accordion("Generation Parameters", open=False):
                enable_thinking = gr.Checkbox(label="Enable Thinking (<|think|>)", value=True)
                max_tokens = gr.Slider(minimum=1, maximum=32768, step=1, value=13576, label="Max Tokens")
                temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=0.2, label="Temperature")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=1.0, label="Top P")
                frequency_penalty = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, value=0.5, label="Frequency Penalty")
                presence_penalty = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, value=0.0, label="Presence Penalty")
                
            submit = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            # Separated the outputs so the UI shows the reasoning clearly
            thoughts = gr.Textbox(label="Thought Process (Reasoning)", lines=10, interactive=False)
            output = gr.Textbox(label="Final Answer", lines=10, interactive=False)

    # Pass all UI components to the generate function
    submit.click(
        fn=generate, 
        inputs=[
            system, 
            query, 
            max_tokens, 
            temperature, 
            top_p, 
            frequency_penalty, 
            presence_penalty,
            enable_thinking
        ], 
        outputs=[thoughts, output]
    )

demo.launch(server_name="127.0.0.1", server_port=7860)
