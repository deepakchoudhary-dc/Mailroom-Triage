from mailroom_triage.preprocess import clean_body, clean_state


def test_html_is_extracted():
    body = "<html><head><style>p{color:red}</style></head><body><p>Invoice</p><p>Amount: EUR 5</p></body></html>"
    out = clean_body(body)
    assert "Invoice" in out and "Amount: EUR 5" in out
    assert "<p>" not in out and "color:red" not in out


def test_quoted_reply_is_removed():
    body = (
        "Amount now owed: EUR 10.\n\n"
        "On Mon, Jan 12, 2026 at 9:14 AM Support <support@example.com> wrote:\n"
        "> old invoice EUR 99"
    )
    out = clean_body(body)
    assert "EUR 10" in out
    assert "EUR 99" not in out


def test_signature_is_removed():
    body = "Your invoice is attached.\n\n-- \nJane Doe\n+1 555 0100"
    out = clean_body(body)
    assert "invoice is attached" in out
    assert "Jane Doe" not in out and "555" not in out


def test_unicode_is_normalised():
    assert clean_body("ＥＵＲ\u00a0１２ ﬁle\u200b") == "EUR 12 file"


def test_clean_body_is_idempotent():
    body = "<p>Invoice</p><p>Amount owed: EUR 10</p>\n\n-- \nJane Doe"
    once = clean_body(body)
    assert clean_body(once) == once


def test_clean_state_headers():
    state = clean_state({"subject": "  Hello\n world ", "from": "a@b.c", "date": "Mon", "body": "x"})
    assert state["subject"] == "Hello world"
    assert state["body"] == "x"


def test_dict_sender_is_formatted_like_real_mail():
    state = clean_state(
        {
            "subject": "Invoice",
            "from": {"display_name": "Luma Billing", "domain": "luma.example", "email": "billing@luma.example"},
            "date": "Mon",
            "body": "Amount owed: EUR 5",
        }
    )
    assert state["from"] == "Luma Billing <billing@luma.example>"


def test_forward_headers_removed_but_body_kept():
    body = (
        "---------- Forwarded message ---------\n"
        "From: accounts@example.com\nDate: Mon, Jan 12, 2026\nSubject: Fwd: message\n\n"
        "Amount owed: EUR 5"
    )
    out = clean_body(body)
    assert "Amount owed: EUR 5" in out
    assert "accounts@example.com" not in out
    assert "Forwarded message" not in out
