import asyncio
import sys
from pathlib import Path

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from index_loader import load_index
from sec_query import SecQueryEngine

async def main():
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o")

    print("Loading index...")
    index = load_index()
    print("Index loaded.")

    engine = SecQueryEngine(index=index)
    response = await engine.query("What risks did Amazon cite in Q1 and Q2 of 2022?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
