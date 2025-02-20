from data_pipeline import DataPipeline

if __name__ == "__main__":
    dp = DataPipeline()
    docs = dp.load_documents("data/lecture_notes")