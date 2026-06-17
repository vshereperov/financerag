import sys
from .retrieve import retrieve
from .generate import generate_answer


def main():
    """Run the CLI."""
    question = " ".join(sys.argv[1:]) or input("Question: ")
    points = retrieve(question, k=5)
    answer = generate_answer(question, points)
    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    for point in points:
        payload = point.payload
        if payload is None:
            continue
        print(
            f"  {payload['company']} | {payload['doc_name']} | page {payload['page']} | score {point.score:.3f}"
        )


if __name__ == "__main__":
    main()
