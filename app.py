import os
import streamlit.web.bootstrap as bootstrap


def main():
	# Delegate to the existing Streamlit app in ui/
	# This entrypoint lets Hugging Face Spaces run `python app.py`.
	ui_path = os.path.join(os.path.dirname(__file__), "ui", "app_streamlit.py")
	bootstrap.run(ui_path, f"streamlit run {ui_path}", [], {})


if __name__ == "__main__":
	main()


