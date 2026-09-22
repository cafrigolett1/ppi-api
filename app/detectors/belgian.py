"""Belgian recognisers for Presidio, validated by checksum.

IBAN and the Belgian national number (rijksregisternummer) are the two labels
in this task that arithmetic decides exactly. Both carry a mod-97 check, so a
candidate can be *verified* rather than guessed at. A model can only ever
approximate what a two-line calculation settles.

These plug into Presidio's own architecture — ``PatternRecognizer`` with a
``validate_result`` hook is precisely the extension point it provides — so the
enhanced engine is still Presidio, not a parallel rule system.

Two concrete defects in the stock recognisers are fixed here:

1. The generic IBAN pattern is greedy across spaces. Given a valid IBAN
   followed by a four-letter word, it absorbs the word into the account
   number, fails its own checksum, and drops the entity entirely:

       "Account BE68 5390 0754 7034 here"           -> nothing detected
       "Account BE68 5390 0754 7034 xy"             -> detected

   Whether a valid IBAN is found depends on the length of the next English
   word. For anonymisation that is a silent leak. The Belgian pattern below is
   length-anchored (BE IBANs are exactly 16 characters), so there is nothing
   to over-consume.

2. There is no Belgian national number recogniser at all. Presidio ships
   US_SSN, UK_NHS and several other national identifiers, but the Belgian
   rijksregisternummer is absent, so it goes completely undetected.
"""

from __future__ import annotations

import re

from presidio_analyzer import Pattern, PatternRecognizer

BE_IBAN_RE = r"\bBE\d{2}(?:[ -]?\d{4}){3}\b"
BE_NATIONAL_NUMBER_RE = r"\b\d{2}[.\-/ ]?\d{2}[.\-/ ]?\d{2}[-.\s]?\d{3}[.\-/ ]?\d{2}\b"


def iban_checksum_ok(candidate: str) -> bool:
    """ISO 13616 mod-97: rotate the first four characters to the end, map
    letters to digits (A=10 … Z=35), remainder must be 1."""
    compact = re.sub(r"[\s-]", "", candidate).upper()
    if not 15 <= len(compact) <= 34:
        return False
    rotated = compact[4:] + compact[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rotated)
    if not digits.isdigit():
        return False
    return int(digits) % 97 == 1


def national_number_checksum_ok(candidate: str) -> bool:
    """Belgian rijksregisternummer mod-97.

    The last two digits are 97 minus the remainder of the nine preceding
    digits. People born from 2000 onward have a '2' prefixed before the
    division, so both variants must be tried.
    """
    digits = re.sub(r"\D", "", candidate)
    if len(digits) != 11:
        return False

    body, check = digits[:9], int(digits[9:])
    month, day = int(digits[2:4]), int(digits[4:6])
    # bis-numbers encode an unknown birth date by adding 20 or 40 to the month
    if not (0 <= month <= 52 and 0 <= day <= 31):
        return False

    return any(97 - (int(p + body) % 97) == check for p in ("", "2"))


class BelgianIbanRecognizer(PatternRecognizer):
    """Length-anchored Belgian IBAN with mod-97 validation."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IBAN_CODE",
            name="BelgianIbanRecognizer",
            patterns=[Pattern(name="be_iban", regex=BE_IBAN_RE, score=0.5)],
            context=["iban", "account", "rekening", "compte", "bank"],
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        """Return True to promote the match to certainty, False to discard it.

        A checksum pass is not 'probably an IBAN' — it is an IBAN. Presidio
        maps True to a score of 1.0 and False removes the result entirely.
        """
        return iban_checksum_ok(pattern_text)


class BelgianNationalNumberRecognizer(PatternRecognizer):
    """Belgian rijksregisternummer (national register number), mod-97 checked.

    Presidio ships no recogniser for this, so it is invisible to the stock
    engine. Without the checksum the pattern alone is dangerously loose — it
    matches many ordinary 11-digit strings — which is exactly why validation
    rather than a raw regex is the right tool.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity="BE_NATIONAL_NUMBER",
            name="BelgianNationalNumberRecognizer",
            patterns=[Pattern(name="be_rrn", regex=BE_NATIONAL_NUMBER_RE, score=0.3)],
            context=[
                "national",
                "rijksregister",
                "rijksregisternummer",
                "registre national",
                "insz",
                "niss",
                "born",
            ],
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return national_number_checksum_ok(pattern_text)
