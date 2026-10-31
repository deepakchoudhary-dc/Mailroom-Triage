from mailroom_triage.heads import build_registry, canonicalize, permute_target, resolve_head


def _row(options, target, kind="choice", question="Pick a category."):
    return {"question": question, "options": options, "target": target, "kind": kind, "metadata": {"question_id": "category"}}


def test_canonicalize_sorts_and_permutes():
    ordered, order = canonicalize(["b", "A", "c"])
    assert ordered == ("A", "b", "c")
    assert permute_target([1.0, 2.0, 3.0], order) == [2.0, 1.0, 3.0]


def test_registry_splits_two_taxonomies_and_picks_default():
    six = ["other", "telecom", "banking", "airline", "education", "electricity"]
    seven = six + ["insurance"]
    rows = [_row(six, [1, 0, 0, 0, 0, 0]) for _ in range(3)]
    rows += [_row(seven, [1, 0, 0, 0, 0, 0, 0]) for _ in range(1)]
    registry = build_registry(rows)
    assert set(registry) == {"category#6", "category#7"}
    assert registry["category#6"].default is True
    assert registry["category#7"].default is False
    assert resolve_head("category", registry) == "category#6"
    assert registry["category#6"].question == "Pick a category."


def test_duplicate_question_without_option_variants_is_single_head():
    rows = [_row(["no", "yes"], [1, 0], kind="noul", question="Is it promotional?")]
    registry = build_registry(rows)
    assert list(registry) == ["category"]
    assert registry["category"].kind == "noul"
