# Quiz Me

## Description
A learning aid generating study questions from a knowledge base of lecture notes.

🔗 Check out the [demo](https://quiz-me.streamlit.app/)!

*Note: you will need an Anthropic API key.*

👏 Template credits: [Chanin Nantasenamat](https://blog.streamlit.io/langchain-tutorial-1-build-an-llm-powered-app-in-18-lines-of-code/).


### Features

- 💽 End-to-end pipeline for indexing, retrieval and generation of `.pdf` files
- 🤗 Choose different indexing strategies: either text only or text & images
- 😄 Add or switch different prompts easily
- 📝 Generate study questions based on existing prompts, or create your own!

### Background

Studying can be overwhelming under time pressure, as students:

- 🫠 struggle to retain information
- 😞 don't know how to spot questions
- 😔 can't find practice questions with verified answers to tackle exams more confidently

In other words, students need a solution to study better.

❗️ Sources suggest that an effective studying technique is active recall, which helps with memorization (Source: [Memorization Strategies](https://learningcenter.unc.edu/tips-and-tools/enhancing-your-memory/#:~:text=Use%20distributed%20practice.&text=Use%20repetition%20to%20firmly%20lodge,in%20between%20each%20study%20session.)).

😎 To take away the guesswork of generating questions, prompts for the specific modules are pre-defined. All users have to do is enter the topic, select the question type, and the questions and the relevant sources are generated.

## Installation
1. Install the following requirements in your system:

```bash
ffmpeg libsm6 libxext6 qpdf poppler-utils libmagic-dev tesseract-ocr libreoffice pandoc
```

- Linux command:
```bash
sudo apt-get install ffmpeg libsm6 libxext6 qpdf poppler-utils libmagic-dev tesseract-ocr libreoffice pandoc
```

2. Clone this repository
3. Create a `data` folder at the project root directory, and copy your `.pdf` files over to the folder. Here's an example:

```bash
├── data
│   ├── 01_raw
│   └── 02_intermediate
```

4. Replace the relevant paths and adjust configurations in `conf/catalog.yml` and `conf/config.yml`
5. Run the following to create and activate your virtual environment.

```bash
uv venv
uv lock
uv sync
source .venv/bin/activate
```

6. Generate the vectorstore (and docstore, if multimodal indexing is selected).
- Under `conf/config.yml`, set `activate` to `True` under the `indexing` key.
- Run the following command:

```bash
uv run src/quiz_me/pipelines/main.py
```


7. Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

8. Or, run individual pipelines (indexing, generation).
- Under `conf/config.yml`, set `activate` to `True` under the pipelines you wish to activate.

## Usage

Coming soon

## Roadmap

### User features
* Chatbot function to query over the knowledge base (e.g. general admin, deadlines, due dates)
* More question types (e.g. Flashcards, MCQ questions for other modules)
* Personal tutor for the students' strengths and weaknesses (e.g. testing them more on certain topics that they're weaker in)
* Emotional support while studying, offering kind words and encouragement.
* Implementation of active recall and retention techniques (e.g. adjusting the time between the last time they were tested on the topic, and the next time they should be tested, based on their familiarity with the topic).

### Technical features
* Overall
    * Topic generation and document tagging
    * Find out learning objectives
    * Generate questions over specific files/topics (rather than entire knowledge base)
* Data pipeline improvements
    * Pre-indexing:
        * Cleaning the loaded documents' contents
        * Remove duplicated images
        * Remove irrelevant images (e.g. logos)
    * Indexing:
        * Use UnstructuredLoader from Langchain library
* Retrieval pipeline improvements
    * Retrieval:
        * More advanced retrieval techniques
* Generation pipeline improvements
    * Stricter typing for MCQ questions for the anatomy scenario
* MLOps
    * Containerize the application such that it is easily rebuilt and redeployable
* Frontend
    * Move frontend to React & related tech stack
* Backend (databases)
    * Deploy vectorstore and docstore to cloud
