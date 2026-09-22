from text_preprocessing import normalize_text


def test_apostrophe_s_is_stripped():
    assert normalize_text("What's the capital of France?") == "What the capital of France?"


def test_cant_and_9_11():
    assert normalize_text("I can't believe it's already 9 11 today") == "I cannot believe it already911today"


def test_us_and_email():
    assert normalize_text("I'm going to the U.S. via e-mail") == "I m going to the U S via email"


def test_would_and_will_contractions():
    assert normalize_text("She said she'd go, and he'll follow") == "She said she would go and he will  follow"


def test_have_contraction():
    assert normalize_text("They've    already left") == "They have already left"


def test_jk():
    assert normalize_text("Just kidding, j k!") == "Just kidding jk ! "


def test_k_thousands_and_eg():
    assert normalize_text("100k dollars for e g purposes") == "100000 dollars for eg purposes"


def test_whitespace_collapsing():
    assert normalize_text("Multiple   spaces   here") == "Multiple spaces here"


def test_non_string_input_is_stringified():
    assert normalize_text(123) == "123"
