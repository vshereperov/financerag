import sys
from .config import settings
from .retrieve import retrieve
from .generate import generate_answer


def main():
    """Run the CLI."""
    question = " ".join(sys.argv[1:]) or input("Question: ")
    points = retrieve(question, k=settings.top_k)
    answer = generate_answer(question, points)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
