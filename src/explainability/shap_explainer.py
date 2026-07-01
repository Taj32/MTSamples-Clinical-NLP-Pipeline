import torch
import numpy as np
import shap
from transformers import AutoTokenizer
from loguru import logger
from src.model.clinicalbert_trainer import ChunkPoolClinicalBERT, collate_fn
from src.stage1_doctype.structural_features import extract_structural_features, N_STRUCTURAL_FEATURES


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


def load_model_from_checkpoint(checkpoint_path: str, stage: int) -> tuple:
    """
    Load a trained ChunkPoolClinicalBERT model from checkpoint.
    Auto-detects whether structural features were used from the classifier weight shape.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    label2id = checkpoint["label2id"]
    id2label = checkpoint["id2label"]
    n_classes = len(label2id)

    # Auto-detect n_structural_features from saved classifier shape
    saved_classifier_shape = checkpoint["model_state_dict"]["classifier.weight"].shape
    # shape is (n_classes, hidden_size + n_struct)
    total_input_size = saved_classifier_shape[1]
    n_struct = total_input_size - 768  # 768 is BERT hidden size

    logger.info(f"Checkpoint classifier shape: {saved_classifier_shape} "
                f"→ detected {n_struct} structural features")

    model = ChunkPoolClinicalBERT(
        n_classes=n_classes,
        n_structural_features=n_struct,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info(f"Loaded Stage {stage} model — {n_classes} classes: {list(label2id.keys())}")
    return model, tokenizer, label2id, id2label

def predict_single(
    text: str,
    model: ChunkPoolClinicalBERT,
    tokenizer,
    id2label: dict,
    stage: int,
    max_length: int = 512,
    stride: int = 256,
    max_chunks: int = 4,
) -> dict:
    """
    Run inference on a single text.
    Returns dict with predicted label, confidence scores, and token count warning.
    """
    from src.stage1_doctype.structural_features import extract_structural_features

    # Tokenize and chunk
    encoding = tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    )
    input_ids = encoding["input_ids"][0]
    token_count = len(input_ids)
    over_limit = token_count > max_length

    chunks_ids, chunks_mask = [], []
    step = max_length - stride

    for start in range(0, max(1, len(input_ids) - 1), step):
        end = start + max_length - 2
        chunk_ids = input_ids[start:end]
        cls = torch.tensor([tokenizer.cls_token_id])
        sep = torch.tensor([tokenizer.sep_token_id])
        chunk_ids = torch.cat([cls, chunk_ids, sep])
        chunk_msk = torch.ones(len(chunk_ids), dtype=torch.long)

        pad_len = max_length - len(chunk_ids)
        if pad_len > 0:
            chunk_ids = torch.cat([chunk_ids, torch.full((pad_len,), tokenizer.pad_token_id)])
            chunk_msk = torch.cat([chunk_msk, torch.zeros(pad_len, dtype=torch.long)])

        chunks_ids.append(chunk_ids[:max_length])
        chunks_mask.append(chunk_msk[:max_length])
        if end >= len(input_ids):
            break

    chunks_ids  = chunks_ids[:max_chunks]
    chunks_mask = chunks_mask[:max_chunks]

    ids  = torch.stack(chunks_ids).unsqueeze(0).to(DEVICE)   # (1, n_chunks, 512)
    mask = torch.stack(chunks_mask).unsqueeze(0).to(DEVICE)

    struct_feats = None
    if stage == 1:
        feats = extract_structural_features(text)
        struct_feats = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(ids, mask, struct_feats)
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = id2label[pred_idx]

    return {
        "predicted_label": pred_label,
        "confidence":      float(probs[pred_idx]),
        "all_scores":      {id2label[i]: float(p) for i, p in enumerate(probs)},
        "token_count":     token_count,
        "over_512_warning": over_limit,
        "n_chunks_used":   len(chunks_ids),
    }


def get_shap_explanation(
    text: str,
    model: ChunkPoolClinicalBERT,
    tokenizer,
    id2label: dict,
    stage: int,
    n_samples: int = 100,
    max_tokens_to_explain: int = 200,
) -> dict:
    """
    Generate SHAP token-level explanation for a single text.
    Uses shap.Explainer with a text masker.

    Returns dict with:
      - shap_values: array of shape (n_tokens, n_classes)
      - tokens: list of token strings
      - predicted_label: str
      - top_tokens: list of (token, shap_value) for predicted class, sorted by importance
    """
    logger.info("Computing SHAP explanation (this may take 30-60 seconds)...")

    # Get prediction first
    prediction = predict_single(text, model, tokenizer, id2label, stage)
    pred_label = prediction["predicted_label"]
    pred_idx   = list(id2label.values()).index(pred_label)

    # Truncate text for SHAP — full-length SHAP is very slow
    tokens_full = tokenizer.tokenize(text)
    if len(tokens_full) > max_tokens_to_explain:
        logger.warning(
            f"Text has {len(tokens_full)} tokens — truncating to "
            f"{max_tokens_to_explain} for SHAP explanation"
        )
        text_for_shap = tokenizer.convert_tokens_to_string(
            tokens_full[:max_tokens_to_explain]
        )
    else:
        text_for_shap = text

    def model_predict(texts):
        """Wrapper for SHAP — takes list of strings, returns probability arrays."""
        results = []
        for t in texts:
            pred = predict_single(t, model, tokenizer, id2label, stage)
            probs = [pred["all_scores"][id2label[i]] for i in range(len(id2label))]
            results.append(probs)
        return np.array(results)

    # Use SHAP's text masker with word-level masking
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(model_predict, masker, output_names=list(id2label.values()))

    shap_values = explainer([text_for_shap], fixed_context=1, batch_size=8)

    # Extract token-level values for predicted class
    tokens = shap_values.data[0]
    values_for_pred_class = shap_values.values[0][:, pred_idx]

    # Build top tokens list
    token_importance = list(zip(tokens, values_for_pred_class.tolist()))
    top_tokens = sorted(token_importance, key=lambda x: abs(x[1]), reverse=True)[:20]

    return {
        "predicted_label":   pred_label,
        "confidence":        prediction["confidence"],
        "tokens":            list(tokens),
        "shap_values":       values_for_pred_class.tolist(),
        "top_tokens":        top_tokens,
        "all_scores":        prediction["all_scores"],
        "token_count":       prediction["token_count"],
        "over_512_warning":  prediction["over_512_warning"],
    }


def format_explanation_text(explanation: dict) -> str:
    """
    Format a SHAP explanation dict into a readable text summary.
    Used for logging and the API response.
    """
    lines = [
        f"Prediction: {explanation['predicted_label']} "
        f"(confidence: {explanation['confidence']:.1%})",
        "",
        "Top tokens driving this prediction:",
    ]
    for token, value in explanation["top_tokens"][:10]:
        direction = "↑ supports" if value > 0 else "↓ opposes"
        lines.append(f"  {direction}  '{token}'  (SHAP: {value:+.4f})")

    if explanation["over_512_warning"]:
        lines += [
            "",
            f"⚠ Note: report has {explanation['token_count']} tokens — "
            f"SHAP explanation computed on first {200} tokens only."
        ]

    return "\n".join(lines)

def render_html_explanation(explanation: dict) -> str:
    """
    Render token-level SHAP values as a color-coded HTML overlay.
    Positive SHAP = green highlight (supports prediction).
    Negative SHAP = red highlight (opposes prediction).
    Intensity scales with absolute SHAP value.
    """
    tokens = explanation["tokens"]
    shap_values = explanation["shap_values"]

    if not tokens or not shap_values:
        return "<p>No explanation available.</p>"

    max_val = max(abs(v) for v in shap_values) or 1.0

    html_parts = [
        f"<div style='font-family: monospace; font-size: 14px; "
        f"line-height: 2; padding: 16px; background: #f9f9f9; "
        f"border-radius: 8px;'>",
        f"<p><strong>Prediction:</strong> {explanation['predicted_label']} "
        f"({explanation['confidence']:.1%} confidence)</p>",
        "<p><strong>Token importance overlay "
        "(green = supports, red = opposes):</strong></p>",
        "<p style='line-height: 2.5;'>",
    ]

    for token, value in zip(tokens, shap_values):
        intensity = abs(value) / max_val
        alpha = 0.15 + 0.7 * intensity  # min 0.15 so low values still visible

        if value > 0:
            color = f"rgba(0, 180, 0, {alpha:.2f})"
        else:
            color = f"rgba(220, 0, 0, {alpha:.2f})"

        # Tooltip shows exact SHAP value on hover
        html_parts.append(
            f"<span style='background-color: {color}; "
            f"padding: 2px 4px; margin: 1px; border-radius: 3px;' "
            f"title='SHAP: {value:+.4f}'>{token}</span>"
        )

    if explanation.get("over_512_warning"):
        html_parts.append(
            f"<br><br><em style='color: #888;'>⚠ Report has "
            f"{explanation['token_count']} tokens — explanation shown "
            f"for first 200 tokens only.</em>"
        )

    html_parts.append("</p></div>")
    return "".join(html_parts)


def export_explanation_to_mlflow(
    explanation: dict,
    html: str,
    stage: int,
    run_name: str = "shap_explanation_sample",
) -> None:
    """
    Export a sample SHAP explanation as MLflow artifacts.
    Saves: explanation JSON, HTML overlay, and top-tokens text summary.
    """
    import mlflow
    import json
    from pathlib import Path

    output_dir = Path("outputs/explanations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"stage{stage}_shap_explanation.json"
    safe_explanation = {k: v for k, v in explanation.items()
                        if k != "shap_values"}  # shap_values is a list, fine to keep
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2)

    # Save HTML overlay
    html_path = output_dir / f"stage{stage}_shap_overlay.html"
    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>Stage {stage} SHAP Explanation</title>
</head><body>
{html}
<hr>
<pre>{format_explanation_text(explanation)}</pre>
</body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    # Log to MLflow
    mlflow.set_experiment(f"stage{stage}_shap_explanations")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(html_path))
        mlflow.log_metrics({
            "explanation_confidence": explanation["confidence"],
            "n_tokens_explained": len(explanation["tokens"]),
            "top_token_shap": abs(explanation["top_tokens"][0][1])
            if explanation["top_tokens"] else 0.0,
        })
    print(f"Exported explanation artifacts to MLflow and {output_dir}/")