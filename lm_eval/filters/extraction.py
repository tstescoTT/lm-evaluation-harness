import re
import sys
import unicodedata

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("regex")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        return filtered_resps


@register_filter("regex_pos")
class POSFilter(Filter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"\['(.*?)'\]",
        group_select=0,
        fallback=None,
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        if fallback is None:
            fallback = ["invalid"]
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):
        def extract_tagged_tokens(text):
            # Extract tagged tokens list from text input using regex
            tokens = re.findall(r"\('([^']*)', '([^']*)'\)", text)
            return [(token, pos) for token, pos in tokens]

        def extract_pos_tags(result):
            pos_tags = []
            if isinstance(result, str):
                result = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result)
            return pos_tags if pos_tags else self.fallback

        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = extract_pos_tags(resp)
                filtered.append(match)
            return filtered

        filtered_resps = map(lambda x: filter_set(x), resps)

        return filtered_resps


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filters out leading and trailing whitespace from responses."""

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                resp = resp.strip()
                filtered_resp.append(resp)
            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """
    A filter used to extract a model's answer on multiple choice questions with
    letter answers. assumes each document has a "choices" field
    containing the list of answer choices and that the answer label symbols
    are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex: r's*([A-?])', where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            if not isinstance(resp, str):
                resp = ""
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


@register_filter("boxed_choice")
class BoxedChoiceFilter(Filter):
    r"""Robust A-D multiple-choice extractor for generate-then-answer tasks.

    ``multi_choice_regex`` with ``regex_pattern: "(\([A-Z]\))"`` and
    ``group_select: -1`` takes the LAST parenthesized capital letter anywhere in
    the response. On reasoning outputs (e.g. GPQA) that is routinely a chemistry
    stereodescriptor / bound variable inside a formula -- ``(E)-bicyclo...`` ->
    ``(E)``, ``((R)-...`` -> ``(R)``, ``(choice B) ... e^{3J}`` -> ``(J)`` --
    not the answer choice.

    Extraction priority (all constrained to A-D):
      1. A ``\\boxed{}`` whose content is *cleanly* a choice letter (e.g.
         ``\\boxed{A}``, ``\\boxed{(A)}``, ``\\boxed{\\text{A}}``). Boxes that
         hold a value/formula are ignored here (their LaTeX command names like
         ``\\displaystyle`` contain A-D letters and must not be mistaken for the
         answer).
      2. A STRONG answer marker only: "answer"/"the answer"/"final answer"/
         "correct answer"/"correct choice" optionally followed by is/:/= and
         ``**``/``(`` then the letter. The LAST such match wins (models restate
         the final choice near the end). Covers ``Answer: (B)``,
         ``Correct choice: C``. Bare "choice"/"option" are intentionally NOT
         markers -- they also occur in explanatory prose ("option B has the
         wrong sign", "the only choice is (C)") and would grab the wrong letter;
         those are resolved by step 3 via the parenthesized answer.
      3. Delegation to ``multi_choice_regex`` constrained to ``[A-D]`` (only when
         the doc carries ``choices``) for the last ``(A-D)`` paren AND for
         answers written out as the choice *text* rather than a letter.

    Output is ``"(X)"`` to match the ``"(A)"``-style target, else ``fallback``.
    """

    # Only STRONG answer markers fire here. Bare "choice"/"option" are
    # deliberately excluded: they also appear in explanatory prose ("option B
    # has the wrong sign", "the only choice is ...") and would hijack the result.
    # Those are handled by the paren/choice-text delegation below, which keys on
    # the parenthesized letter (only the real answer is written as "(X)").
    _MARKER_RE = re.compile(
        r"(?:final\s+answer|correct\s+answer|correct\s+choice|the\s+answer|answer)s?"
        r"\b\s*(?:is|are|:|=|->|=>)?\s*\*{0,2}\s*\(?\s*([A-D])(?![A-Za-z])",
        flags=re.IGNORECASE,
    )
    _CLEAN_BOXED_RE = re.compile(r"^\(?([A-D])\)?$", flags=re.IGNORECASE)

    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback
        # Constrained multi_choice_regex reused for paren + choice-text matching.
        self._mcr = MultiChoiceRegexFilter(
            regex_pattern=r"(\([A-D]\))",
            group_select=-1,
            fallback=fallback,
            ignore_case=True,
            ignore_punctuation=True,
        )

    @staticmethod
    def _strip_think(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _extract_boxed(text: str) -> str:
        r"""Content of the last ``\boxed{...}`` via brace matching, else ''."""
        if "boxed" not in text:
            return ""
        ans = text.split("boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            out = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    out += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    out += c
                else:
                    out += c
            return out
        return ans.split("$")[0].strip()

    def _clean_boxed_letter(self, text: str) -> str:
        """Return the letter only if the box holds *just* a choice letter."""
        boxed = self._extract_boxed(text)
        if not boxed:
            return ""
        s = boxed.strip()
        # unwrap \text{...}/\mathrm{...}/\mathbf{...} and drop $, *, whitespace
        s = re.sub(r"\\(?:text|mathrm|mathbf|rm|bf|mathsf)\s*\{([^}]*)\}", r"\1", s)
        s = s.replace("$", "").replace("*", "").strip()
        m = self._CLEAN_BOXED_RE.match(s)
        return m.group(1).upper() if m else ""

    def _marker_letter(self, text: str) -> str:
        matches = self._MARKER_RE.findall(text)
        return matches[-1].upper() if matches else ""

    def _extract_one(self, pred: str, doc: dict) -> str:
        if not isinstance(pred, str) or not pred:
            return self.fallback
        text = self._strip_think(pred)

        # 1) Boxed answer, only if it is cleanly a choice letter.
        letter = self._clean_boxed_letter(text)
        if letter:
            return f"({letter})"

        # 2) Explicit answer/choice marker (last occurrence).
        letter = self._marker_letter(text)
        if letter:
            return f"({letter})"

        # 3) Constrained multi_choice_regex (paren + choice-text). Needs choices.
        if isinstance(doc, dict) and doc.get("choices"):
            return self._mcr.apply([[text]], [doc])[0][0]

        # 3b) No choices available: last (A-D) paren as a final resort.
        paren = re.findall(r"\(([A-D])\)", text, flags=re.IGNORECASE)
        return f"({paren[-1].upper()})" if paren else self.fallback

    def apply(self, resps, docs):
        filtered_resps = []
        for group, doc in zip(resps, docs, strict=False):
            filtered = [self._extract_one(resp, doc) for resp in group]
            filtered_resps.append(filtered)
        return filtered_resps
