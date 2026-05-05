"""
predict.py
-----------
CLI tool to screen a single resume against a job description.
Uses both saved models and shows combined verdict.

Usage:
    python predict.py --resume path/to/resume.pdf --jd path/to/jd.txt
    python predict.py --resume-text "John Doe, Python, ML..." --jd-text "We need a Data Scientist..."
    python predict.py  # interactive mode
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model1_tfidf_rf import TFIDFRandomForestScreener
from models.model2_bert import BERTSimilarityScreener


MODEL1_PATH = "saved_models/model1_tfidf_rf.pkl"
MODEL2_PATH = "saved_models/model2_bert.pkl"


def read_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        print(f"[Error] Could not read PDF: {e}")
        return ""


def read_docx(path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(path)
        return " ".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        print(f"[Error] Could not read DOCX: {e}")
        return ""


def read_file(path: str) -> str:
    """Auto-detect and read resume file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext in [".docx", ".doc"]:
        return read_docx(path)
    elif ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        print(f"[Warning] Unknown file type '{ext}'. Attempting plain text read.")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()


def load_models():
    """Load both trained models."""
    model1 = TFIDFRandomForestScreener()
    model2 = BERTSimilarityScreener()

    if not os.path.exists(MODEL1_PATH):
        print(f"[Error] Model 1 not found at {MODEL1_PATH}. Run train.py first.")
        sys.exit(1)
    if not os.path.exists(MODEL2_PATH):
        print(f"[Error] Model 2 not found at {MODEL2_PATH}. Run train.py first.")
        sys.exit(1)

    model1.load(MODEL1_PATH)
    model2.load(MODEL2_PATH)
    return model1, model2


def screen_resume(resume_text: str, job_desc: str, model1, model2):
    """Run both models and display results."""
    print("\n" + "="*60)
    print("  RESUME SCREENING RESULTS")
    print("="*60)

    result1 = model1.predict(resume_text, job_desc)
    result2 = model2.predict(resume_text, job_desc)

    # Combined score (average)
    combined_score = (result1["score"] + result2["score"]) / 2
    combined_shortlisted = combined_score >= 50

    print(f"\n{'Model':<35} {'Score':>8} {'Verdict'}")
    print("-"*60)
    print(f"{result1['model_name']:<35} {result1['score']:>7.1f}%  {result1['verdict']}")
    print(f"{result2['model_name']:<35} {result2['score']:>7.1f}%  {result2['verdict']}")
    print("-"*60)
    verdict = "✅ SHORTLISTED" if combined_shortlisted else "❌ REJECTED"
    print(f"{'COMBINED (Average)':<35} {combined_score:>7.1f}%  {verdict}")
    print("="*60)

    # Extra details for Model 2
    if "cosine_similarity" in result2:
        print(f"\n[BERT Details]")
        print(f"  Semantic Similarity : {result2['cosine_similarity']:.1f}%")
        print(f"  Keyword Overlap     : {result2['keyword_overlap']:.1f}%")

    return {
        "model1": result1,
        "model2": result2,
        "combined_score": combined_score,
        "combined_verdict": verdict,
    }


def interactive_mode(model1, model2):
    """Interactive CLI mode."""
    print("\n" + "="*60)
    print("  INTERACTIVE RESUME SCREENER")
    print("  (Type 'quit' to exit)")
    print("="*60)

    while True:
        print("\nPaste resume text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines:
                    break
            elif line.lower() == "quit":
                print("Goodbye!")
                return
            else:
                lines.append(line)
        resume_text = "\n".join(lines)

        print("\nPaste job description (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines:
                    break
            elif line.lower() == "quit":
                return
            else:
                lines.append(line)
        job_desc = "\n".join(lines)

        screen_resume(resume_text, job_desc, model1, model2)


def main():
    parser = argparse.ArgumentParser(description="Resume Screening CLI")
    parser.add_argument("--resume", type=str, help="Path to resume file (PDF/DOCX/TXT)")
    parser.add_argument("--jd", type=str, help="Path to job description file (TXT)")
    parser.add_argument("--resume-text", type=str, help="Resume text directly")
    parser.add_argument("--jd-text", type=str, help="Job description text directly")
    args = parser.parse_args()

    model1, model2 = load_models()

    # Determine input mode
    if args.resume or args.resume_text:
        resume_text = args.resume_text or read_file(args.resume)
        job_desc = args.jd_text or (read_file(args.jd) if args.jd else "")

        if not resume_text:
            print("[Error] Resume text is empty.")
            sys.exit(1)
        if not job_desc:
            print("[Error] Job description text is empty.")
            sys.exit(1)

        screen_resume(resume_text, job_desc, model1, model2)
    else:
        interactive_mode(model1, model2)


if __name__ == "__main__":
    main()
