from .summarizer import summarize_pdf


def main():
    print("=== PDF SUMMARY ===\n")
    summary = summarize_pdf(max_chars=3000, summary_words=200)
    print(summary)
    print("\n=====================")


if __name__ == "__main__":
    main()
