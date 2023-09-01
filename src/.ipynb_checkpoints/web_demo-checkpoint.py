from llmtuner import create_web_demo


def main():
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=26921, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
