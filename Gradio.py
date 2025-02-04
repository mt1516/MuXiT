import gradio as gr

def echo(message, history): # This mirrors your input for now
    return message

def vote(data: gr.LikeData):
    if data.liked:
        print("Thank you! Glad you liked it :)")
    else:
        print("Thanks for your feedback!")

with gr.Blocks() as testApp:
    gr.Markdown("**HKUST CSE 2024-25 FYP | HO3**")
    gr.Markdown("To generate a music clip, try typing in a prompt or upload a music clip.")
    chatbot = gr.Chatbot(placeholder="<strong>HKUST CSE 2024-25 FYP</strong><br>HO3 | Music Generation with Generative AI")
    chatbot.like(vote, None, None)

    gr.ChatInterface(
        fn=echo,
        multimodal=True,
        editable=True,
        # inputs='textbox',
        # outputs='textbox',
        type='messages',
        title="Music Generator",
        description="Generate pop music clips based on user input (text, song, etc.) with generative AI.",
        chatbot=chatbot,
        stop_btn=True,
        save_history=True,
        show_progress="full",
        examples=["Hello there", "Generate a J-pop music clip", "Extend this song"]
    )

if __name__ == "__main__":
    testApp.launch(pwa=True, share=True)
