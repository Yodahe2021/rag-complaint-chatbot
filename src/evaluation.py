# =============================================================================
# evaluation.py - RAG Evaluation Helpers
# =============================================================================
"""
This module provides tools for evaluating RAG pipeline quality.

KEY CONCEPTS FOR BEGINNERS:

WHY EVALUATE RAG?
- RAG systems can fail in many ways
- Retrieval might return irrelevant documents
- LLM might ignore context or hallucinate
- We need systematic ways to measure quality

EVALUATION DIMENSIONS:
1. Retrieval Quality: Are the right documents being retrieved?
2. Answer Quality: Is the answer correct and helpful?
3. Faithfulness: Does the answer stick to the context?
4. Relevance: Does the answer address the question?

MANUAL vs AUTOMATIC EVALUATION:
- Manual: Human reviews and scores (gold standard, but slow)
- Automatic: Metrics like BLEU, ROUGE, semantic similarity
- For learning, we'll focus on manual evaluation with clear criteria

SCORING GUIDE (1-5):
5 = Excellent: Accurate, complete, well-sourced
4 = Good: Mostly accurate, minor issues
3 = Acceptable: Partially answers question
2 = Poor: Significant issues or missing info
1 = Fail: Wrong, irrelevant, or hallucinated
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd

from .rag_pipeline import RAGPipeline, RAGResponse


@dataclass
class EvaluationResult:
    """
    Container for a single evaluation result.
    
    Attributes:
        question: The test question
        answer: Generated answer
        sources_summary: Brief summary of retrieved sources
        score: Quality score (1-5)
        comments: Evaluator's comments
    """
    question: str
    answer: str
    sources_summary: str
    score: Optional[int] = None
    comments: str = ""


@dataclass
class EvaluationReport:
    """
    Container for full evaluation report.
    
    Attributes:
        results: List of individual evaluation results
        settings: Configuration used for evaluation
    """
    results: List[EvaluationResult] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def average_score(self) -> float:
        """Calculate average score across all scored results."""
        scored = [r.score for r in self.results if r.score is not None]
        return sum(scored) / len(scored) if scored else 0.0
    
    @property
    def num_evaluated(self) -> int:
        """Count of results with scores."""
        return sum(1 for r in self.results if r.score is not None)


# =============================================================================
# SAMPLE EVALUATION QUESTIONS
# =============================================================================

# These questions cover different aspects of the support ticket data
SAMPLE_EVALUATION_QUESTIONS = [
    # Product-specific questions
    "What are the most common issues reported for smart TVs?",
    "What problems do customers face with GoPro cameras?",
    
    # Issue type questions
    "What are typical billing and payment issues customers report?",
    "What technical issues are most frequently mentioned?",
    
    # Priority/severity questions
    "What types of issues are marked as critical priority?",
    "What patterns do you see in high priority tickets?",
    
    # Resolution questions
    "How are refund requests typically handled?",
    "What solutions are provided for device connectivity issues?",
    
    # Channel-specific questions
    "What issues come through social media channels?",
    "Are there differences in issues reported via email vs chat?",
]


def summarize_sources(sources: list, max_sources: int = 2) -> str:
    """
    Create a brief summary of retrieved sources.
    
    Args:
        sources: List of Document objects
        max_sources: Maximum number of sources to include
        
    Returns:
        Formatted string summarizing sources
    """
    if not sources:
        return "No sources retrieved"
    
    summaries = []
    for doc in sources[:max_sources]:
        ticket_id = doc.metadata.get('ticket_id', 'N/A')
        product = doc.metadata.get('product', 'N/A')
        preview = doc.page_content[:100].replace('\n', ' ')
        summaries.append(f"[{ticket_id}] {product}: {preview}...")
    
    return " | ".join(summaries)


def run_evaluation(
    rag_pipeline: RAGPipeline,
    questions: List[str] = None,
    verbose: bool = True
) -> EvaluationReport:
    """
    Run RAG pipeline on a set of questions and collect results.
    
    This function runs each question through the pipeline and
    collects the answers and sources for manual evaluation.
    
    Args:
        rag_pipeline: Initialized RAGPipeline
        questions: List of questions (uses SAMPLE_EVALUATION_QUESTIONS if None)
        verbose: Print progress
        
    Returns:
        EvaluationReport with results ready for scoring
        
    Example:
        >>> rag = RAGPipeline()
        >>> report = run_evaluation(rag)
        >>> # Now manually score each result
        >>> report.results[0].score = 4
        >>> report.results[0].comments = "Good answer, missed one detail"
    """
    if questions is None:
        questions = SAMPLE_EVALUATION_QUESTIONS
    
    report = EvaluationReport(
        settings={
            "num_questions": len(questions),
            "retrieval_k": rag_pipeline.retrieval_k,
        }
    )
    
    if verbose:
        print("=" * 60)
        print(f"Running evaluation on {len(questions)} questions...")
        print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        if verbose:
            print(f"\n[{i}/{len(questions)}] {question[:50]}...")
        
        # Run RAG pipeline
        response = rag_pipeline.ask(question)
        
        # Create evaluation result
        result = EvaluationResult(
            question=question,
            answer=response.answer,
            sources_summary=summarize_sources(response.sources),
        )
        
        report.results.append(result)
        
        if verbose:
            print(f"    Answer: {response.answer[:100]}...")
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Evaluation complete. Results ready for scoring.")
        print("=" * 60)
    
    return report


def results_to_dataframe(report: EvaluationReport) -> pd.DataFrame:
    """
    Convert evaluation results to a pandas DataFrame.
    
    Args:
        report: EvaluationReport with results
        
    Returns:
        DataFrame with columns: Question, Answer, Sources, Score, Comments
    """
    data = []
    for r in report.results:
        data.append({
            "Question": r.question,
            "Generated Answer": r.answer,
            "Retrieved Sources": r.sources_summary,
            "Quality Score": r.score,
            "Comments": r.comments,
        })
    
    return pd.DataFrame(data)


def results_to_markdown_table(report: EvaluationReport) -> str:
    """
    Convert evaluation results to a Markdown table.
    
    Args:
        report: EvaluationReport with results
        
    Returns:
        Markdown-formatted table string
    """
    # Header
    lines = [
        "| # | Question | Generated Answer | Sources | Score | Comments |",
        "|---|----------|------------------|---------|-------|----------|",
    ]
    
    # Rows
    for i, r in enumerate(report.results, 1):
        # Truncate long text for table readability
        question = r.question[:40] + "..." if len(r.question) > 40 else r.question
        answer = r.answer[:60] + "..." if len(r.answer) > 60 else r.answer
        sources = r.sources_summary[:40] + "..." if len(r.sources_summary) > 40 else r.sources_summary
        score = str(r.score) if r.score else "-"
        comments = r.comments[:30] + "..." if len(r.comments) > 30 else r.comments
        
        # Escape pipe characters
        question = question.replace("|", "\\|")
        answer = answer.replace("|", "\\|")
        sources = sources.replace("|", "\\|")
        comments = comments.replace("|", "\\|")
        
        lines.append(f"| {i} | {question} | {answer} | {sources} | {score} | {comments} |")
    
    # Summary
    lines.append("")
    lines.append(f"**Average Score:** {report.average_score:.2f} / 5.0")
    lines.append(f"**Questions Evaluated:** {report.num_evaluated} / {len(report.results)}")
    
    return "\n".join(lines)


def print_scoring_guide():
    """Print the manual scoring guide for evaluators."""
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║                    RAG EVALUATION SCORING GUIDE                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Score 5 - EXCELLENT                                              ║
║  • Answer is accurate and complete                                ║
║  • Directly addresses the question                                ║
║  • Well-supported by retrieved sources                            ║
║  • No hallucinations or made-up information                       ║
║                                                                   ║
║  Score 4 - GOOD                                                   ║
║  • Answer is mostly accurate                                      ║
║  • Addresses the main point of the question                       ║
║  • Minor omissions or imprecisions                                ║
║  • Sources are relevant                                           ║
║                                                                   ║
║  Score 3 - ACCEPTABLE                                             ║
║  • Answer partially addresses the question                        ║
║  • Some relevant information provided                             ║
║  • May be incomplete or slightly off-topic                        ║
║  • Sources somewhat relevant                                      ║
║                                                                   ║
║  Score 2 - POOR                                                   ║
║  • Answer has significant issues                                  ║
║  • Missing key information                                        ║
║  • May contain minor inaccuracies                                 ║
║  • Sources may not be well-utilized                               ║
║                                                                   ║
║  Score 1 - FAIL                                                   ║
║  • Answer is wrong or irrelevant                                  ║
║  • Contains hallucinations (made-up facts)                        ║
║  • Does not address the question                                  ║
║  • Sources are irrelevant or ignored                              ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# =============================================================================
# DEBUG HELPERS
# =============================================================================

def compare_retrieval_k(
    rag_pipeline: RAGPipeline,
    question: str,
    k_values: List[int] = [3, 5, 8]
) -> Dict[int, List[str]]:
    """
    Compare retrieval results with different k values.
    
    Useful for understanding how k affects retrieval.
    
    Args:
        rag_pipeline: RAGPipeline (will modify k temporarily)
        question: Question to test
        k_values: List of k values to try
        
    Returns:
        Dictionary mapping k to list of ticket IDs retrieved
    """
    results = {}
    
    print(f"Comparing retrieval for: '{question}'")
    print("-" * 50)
    
    for k in k_values:
        # Get retriever with this k
        from .vectorstore import search_similar
        docs = search_similar(rag_pipeline.vectorstore, question, k=k)
        
        ticket_ids = [doc.metadata.get('ticket_id', 'N/A') for doc in docs]
        results[k] = ticket_ids
        
        print(f"k={k}: {ticket_ids}")
    
    return results


def compare_chunk_settings(
    questions: List[str],
    chunk_configs: List[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Compare RAG performance with different chunk settings.
    
    NOTE: This requires rebuilding the vector store for each config,
    so it's slow. Use for experimentation only.
    
    Args:
        questions: Questions to test
        chunk_configs: List of {"chunk_size": X, "chunk_overlap": Y} dicts
        
    Returns:
        DataFrame comparing results
    """
    if chunk_configs is None:
        chunk_configs = [
            {"chunk_size": 300, "chunk_overlap": 30},
            {"chunk_size": 500, "chunk_overlap": 50},
            {"chunk_size": 800, "chunk_overlap": 80},
        ]
    
    print("⚠️  This function requires rebuilding vector stores.")
    print("   For a quick demo, see the evaluation notebook.")
    
    # Return empty DataFrame as placeholder
    # Full implementation would rebuild vector stores
    return pd.DataFrame({
        "chunk_size": [c["chunk_size"] for c in chunk_configs],
        "chunk_overlap": [c["chunk_overlap"] for c in chunk_configs],
        "note": ["Rebuild vector store to test"] * len(chunk_configs),
    })
