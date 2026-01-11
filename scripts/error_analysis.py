def check_bio_violation(labels):
    errors = []
    for i, tag in enumerate(labels):
        if tag.startswith("I-"):
            if i == 0 or labels[i-1] != "B-" + tag[2:]:
                errors.append({
                    "type": "BIO_VIOLATION",
                    "position": i,
                    "label": tag
                })
    return errors


def check_boundary_errors(tokens, labels):
    errors = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            ent_type = labels[i][2:]
            j = i + 1
            while j < len(labels) and labels[j] == "I-" + ent_type:
                j += 1

            entity_tokens = tokens[i:j]

            # heuristic: financial values shouldn't be fragmented
            if ent_type == "VALUE" and len(entity_tokens) > 1:
                errors.append({
                    "type": "BOUNDARY_ERROR",
                    "entity": ent_type,
                    "tokens": entity_tokens
                })
            i = j
        else:
            i += 1
    return errors


def check_tokenization_errors(tokens):
    errors = []
    for i in range(len(tokens) - 1):
        if tokens[i] in ["$", "€"] and tokens[i+1].replace(".", "").isdigit():
            errors.append({
                "type": "TOKENIZATION_ERROR",
                "pattern": f"{tokens[i]} {tokens[i+1]}"
            })
    return errors


def analyze_ner(tokens, labels):
    report = {
        "bio_errors": check_bio_violation(labels),
        "boundary_errors": check_boundary_errors(tokens, labels),
        "tokenization_errors": check_tokenization_errors(tokens)
    }
    return report
