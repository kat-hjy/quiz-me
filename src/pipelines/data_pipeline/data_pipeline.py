import re
import pypdf
import pathlib
from typing import List

class DataPipeline():
    def __init__(self):
        pass

    def load_documents(self, path: str) -> List[str]:
        path = pathlib.Path(path)
        if path.is_dir():
            docs = []
            for file in path.iterdir():
                if file.is_file() and file.suffix == ".pdf":
                    pdf = pypdf.PyPDF(file)
                    text = pdf.text
                    docs.append(text)
            return docs
        else:
            raise FileNotFoundError(f"Directory {path} not found")

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split(" ")

    def preprocess(self, text: str) -> List[str]:
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        return tokens