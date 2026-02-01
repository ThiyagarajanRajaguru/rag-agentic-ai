from rag_graph import rag_app

if __name__ == "__main__":
    question = "What is Agentic AI?"

    result = rag_app.invoke({
        "question": question
    })

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nCONTEXTS USED:")
    for i, ctx in enumerate(result["contexts"], 1):
        print(f"\n--- Context {i} ---")
        print(ctx[:300], "...")

    print("\nCONFIDENCE SCORE:")
    print(result["confidence"])
