from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from app.schemas import ReviewRequest, ReviewResponse, DocResult, DocInput
from app.models import EnsembleReviewer
from app.providers.ollama_client import OllamaClient
import os

app = FastAPI(title="Responsiveness & ECA API", version="0.3.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, recommend specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
MODEL_DEFAULTS = ["gemma3:4b", "deepseek-r1:8b"]
DEFAULT_CASE_BACKGROUND = "Legal discovery case involving email communications between corporate executives regarding energy trading and market analysis."

# ---------- Utility Functions ----------
def load_test_emails():
    """Load test email files - simplified version"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of paths to try
    possible_paths = [
        os.path.join(current_dir, "app", "ECA", "test_emails"),
        os.path.join(os.getcwd(), "ai_review", "app", "ECA", "test_emails"),
        os.path.join(os.getcwd(), "app", "ECA", "test_emails"),
    ]
    
    for test_path in possible_paths:
        if os.path.exists(test_path):
            return _load_emails_from_path(test_path)
    
    return []

def _load_emails_from_path(path: str) -> List[Dict[str, str]]:
    """Load email files from specified path"""
    emails = []
    try:
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                filepath = os.path.join(path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    emails.append({"filename": filename, "text": f.read()})
    except Exception:
        pass  # Silent error handling
    return emails

def format_review_results(results: List[DocResult], mode: str = "ensemble") -> Dict:
    """Format review results - unified formatting function"""
    detailed_results = []
    responsive_count = 0
    
    for doc_result in results:
        is_responsive = doc_result.final_label.lower() == "responsive"
        if is_responsive:
            responsive_count += 1
        
        detailed_results.append({
            "filename": doc_result.filename,
            "is_responsive": doc_result.final_label,
            "status": "RESPONSIVE" if is_responsive else "NON-RESPONSIVE",
            "confidence_votes": doc_result.votes,  # Match frontend expected field name
            "ai_reasoning": doc_result.rationales  # Match frontend expected field name
        })
    
    return {
        "SUMMARY": {  # Match frontend expected key name
            "total_emails_analyzed": len(results),  # Match frontend expected field name
            "responsive_emails": responsive_count,
            "non_responsive_emails": len(results) - responsive_count,
            "mode": mode.upper()
        },
        "DETAILED_RESULTS": detailed_results,  # Match frontend expected key name
        "quick_view": [f"{r['filename']} -> {r['status']}" for r in detailed_results]
    }

# ---------- Core API Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "version": "2025-09-22"}

@app.get("/models")
def models():
    return {"defaults": MODEL_DEFAULTS}

@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    """Core review endpoint"""
    client = OllamaClient()
    engine = EnsembleReviewer(client)
    results: List[DocResult] = []
    mode = (req.mode or "ensemble").lower()
    
    if mode == "single":
        model_name = req.model_name or MODEL_DEFAULTS[0]
        for d in req.docs:
            out = engine.review_doc_single(model_name, req.case_background, d.text)
            results.append(DocResult(
                filename=d.filename,
                final_label=out["final"],
                votes=out["votes"],
                rationales=out["rationales"]
            ))
    elif mode == "collab":
        names = req.model_names or MODEL_DEFAULTS
        arbiter = req.model_name or names[0]
        for d in req.docs:
            out = engine.review_doc_collab(names, req.case_background, d.text,
                                         arbiter=arbiter, critique_rounds=1)
            rats = {**out["rationales"], "__arbiter__": out.get("arbiter_rationale", "")}
            results.append(DocResult(
                filename=d.filename,
                final_label=out["final"],
                votes=out["votes"],
                rationales=rats
            ))
    else:  # ensemble (majority)
        names = req.model_names or MODEL_DEFAULTS
        for d in req.docs:
            out = engine.review_doc_ensemble(names, req.case_background, d.text)
            results.append(DocResult(
                filename=d.filename,
                final_label=out["final"],
                votes=out["votes"],
                rationales=out["rationales"]
            ))
    
    return ReviewResponse(results=results)

# ---------- Test Data Endpoints ----------
@app.get("/test/emails")
def get_test_emails():
    """Get test email files"""
    emails = load_test_emails()
    return {"count": len(emails), "emails": emails}

@app.post("/test/review")
def test_review(mode: str = "ensemble", case_background: Optional[str] = None):
    """Unified test review endpoint"""
    emails = load_test_emails()
    if not emails:
        raise HTTPException(status_code=404, detail="No test email files found")
    
    docs = [DocInput(filename=email["filename"], text=email["text"]) for email in emails]
    
    req = ReviewRequest(
        mode=mode,
        case_background=case_background or DEFAULT_CASE_BACKGROUND,
        docs=docs
    )
    
    result = review(req)
    return format_review_results(result.results, mode)

# ---------- Legacy Compatibility Endpoints ----------
@app.get("/test/data")
def get_test_cases():
    """Backward compatible test data endpoint"""
    emails = load_test_emails()
    return {
        "cases": [{
            "id": "case_001",
            "name": "Real Email Responsiveness Test",
            "case_background": DEFAULT_CASE_BACKGROUND,
            "docs": emails
        }]
    }

@app.get("/test/data/{case_id}")
def get_test_case(case_id: str):
    """Get specific test case"""
    if case_id != "case_001":
        raise HTTPException(status_code=404, detail="Test case not found")
    return get_test_cases()["cases"][0]

@app.post("/test/review/{case_id}")
def test_review_by_case(case_id: str, mode: str = "ensemble"):
    """Run review using specified test case - redirect to unified endpoint"""
    if case_id != "case_001":
        raise HTTPException(status_code=404, detail="Test case not found")
    return test_review(mode)

@app.post("/test/review-files")
def test_review_from_files(mode: str = "ensemble"):
    """Test review from files - redirect to unified endpoint"""
    return test_review(mode)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)