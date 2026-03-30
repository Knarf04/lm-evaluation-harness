def qasper_doc_to_target(doc):
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("abstract"):
        parts.append(doc["abstract"])
    full_text = doc.get("full_text", {})
    for section, paras in zip(
        full_text.get("section_name", []), full_text.get("paragraphs", [])
    ):
        if section:
            parts.append(section)
        parts.extend(paras)
    return "\n\n".join(parts)
