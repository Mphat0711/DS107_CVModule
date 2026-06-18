from cv_module.config import HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, MIN_CONFIDENCE_MARGIN


def build_confidence_ir(detections):
    if not detections:
        return {
            "level": "no_detection",
            "top_confidence": None,
            "second_confidence": None,
            "confidence_margin": None,
            "needs_llm_confirmation": True,
            "llm_strategy": "ask_for_clearer_image_or_field_context",
            "reason": "Khong co bbox nao vuot nguong confidence.",
        }

    top = detections[0]
    second = detections[1] if len(detections) > 1 else None
    top_conf = float(top["confidence"])
    second_conf = float(second["confidence"]) if second else None
    margin = top_conf - second_conf if second_conf is not None else None

    ambiguous = margin is not None and margin < MIN_CONFIDENCE_MARGIN
    if top_conf >= HIGH_CONFIDENCE and not ambiguous:
        level = "high"
        needs_llm_confirmation = False
        strategy = "direct_advice"
        reason = "YOLO confidence cao va cach biet voi nhan dung thu hai du lon."
    elif top_conf >= MEDIUM_CONFIDENCE:
        level = "ambiguous" if ambiguous else "medium"
        needs_llm_confirmation = True
        strategy = "cautious_advice_with_confirmation_question"
        reason = (
            "YOLO co du tin hieu de goi DS107, nhung nen de LLM hoi them "
            "hoac dien dat can trong."
        )
    else:
        level = "low"
        needs_llm_confirmation = True
        strategy = "ask_for_clearer_image_or_manual_check"
        reason = "YOLO confidence thap, khong nen tu van chac chan."

    return {
        "level": level,
        "top_confidence": round(top_conf, 6),
        "second_confidence": round(second_conf, 6) if second_conf is not None else None,
        "confidence_margin": round(margin, 6) if margin is not None else None,
        "needs_llm_confirmation": needs_llm_confirmation,
        "llm_strategy": strategy,
        "reason": reason,
    }

