"""Language and Swiss German dialect detection from transcriptions and conversation cues."""

from __future__ import annotations

import re

SWISS_GERMAN_MARKERS: frozenset[str] = frozenset(
    {
        "grüezi",
        "grüessech",
        "guete",
        "guet",
        "mitenand",
        "zäme",
        "zämu",
        "chasch",
        "cha",
        "chan",
        "chönd",
        "chume",
        "chunsch",
        "chunt",
        "nöd",
        "nid",
        "öd",
        "üt",
        "hüt",
        "hite",
        "hitte",
        "öppis",
        "öpis",
        "öpper",
        "allweg",
        "gsi",
        "gsii",
        "luege",
        "lueg",
        "lose",
        "säge",
        "ghöre",
        "schaffe",
        "schwiizerdütsch",
        "schwyzerdütsch",
        "wallisertiitsch",
        "bärndütsch",
        "züridütsch",
        "baseldytsch",
        "wätter",
        "isch",
        "bisch",
        "häsch",
        "händ",
        "hei",
        "hälfe",
        "hälfu",
        "ig",
        "ier",
        "dier",
        "mer",
        "mier",
        "wier",
        "ew",
        "eui",
        "embrüf",
        "embri",
        "güet",
        "schee",
        "schöu",
        "willchu",
        "leiggerbad",
        "loukerbad",
        "geit",
        "giele",
        "uf",
        "wiederseh",
        "wiederluege",
        "ade",
        "sali",
        "hoi",
        "gönd",
        "füre",
        "fürä",
        "vüre",
        "zrugg",
        "graduus",
        "rächts",
        "träi",
        "dräi",
        "füüf",
    }
)

# Distinct lexical cues for SwissDial regional dialects
DIALECT_LEXICAL_CUES: dict[str, frozenset[str]] = {
    "vs": frozenset(
        {
            "wallisertiitsch",
            "embrüf",
            "embri",
            "güet",
            "wier",
            "mier",
            "dier",
            "ier",
            "ew",
            "hälfu",
            "ischs",
            "leukerbad",
            "leiggerbad",
            "loukerbad",
            "oberwallis",
            "brig",
            "visp",
            "zermatt",
        }
    ),
    "be": frozenset({"bärndütsch", "grüessech", "ig", "nid", "geit", "hei", "allweg", "giele", "modis", "bärn"}),
    "zh": frozenset({"züridütsch", "nöd", "händ", "züri", "tsüri", "öpper", "gsii"}),
    "bs": frozenset({"baseldytsch", "sali", "nit", "basel", "drämmli", "velo"}),
    "sg": frozenset({"ostschwiizerdütsch", "öd", "ebs", "sankt", "galle", "thurgi"}),
}

FRENCH_MARKERS: frozenset[str] = frozenset(
    {
        "bonjour",
        "salut",
        "bonsoir",
        "comment",
        "allez",
        "merci",
        "beaucoup",
        "pouvez",
        "pouvez-vous",
        "vous",
        "nous",
        "je",
        "suis",
        "est",
        "dans",
        "pour",
        "avec",
        "une",
        "des",
        "les",
        "quel",
        "quelle",
        "aujourd'hui",
        "maintenant",
        "français",
        "francais",
        "s'il",
        "plaît",
        "valais",
        "suisse",
        "avance",
        "avancer",
        "avancez",
        "recule",
        "reculer",
        "tourne",
        "gauche",
        "droite",
        "retourne",
        "mètre",
        "mètres",
        "de",
        "arrête",
        "regarde",
        "vérifier",
        "état",
    }
)

SWISS_FRENCH_MARKERS: frozenset[str] = frozenset(
    {"septante", "huitante", "nonante", "natel", "déjeuner", "dîner", "souper", "valais", "loèche"}
)

STANDARD_GERMAN_MARKERS: frozenset[str] = frozenset(
    {
        "guten",
        "tag",
        "morgen",
        "abend",
        "hallo",
        "wie",
        "geht",
        "ihnen",
        "ich",
        "sie",
        "wir",
        "mir",
        "mich",
        "dich",
        "sich",
        "uns",
        "ein",
        "eine",
        "einen",
        "der",
        "die",
        "das",
        "den",
        "dem",
        "und",
        "oder",
        "aber",
        "dann",
        "jetzt",
        "hier",
        "nach",
        "um",
        "können",
        "kannst",
        "nicht",
        "heute",
        "wetter",
        "bitte",
        "danke",
        "roboter",
        "vorwärts",
        "rückwärts",
        "zurück",
        "zurückkehren",
        "geh",
        "gehe",
        "gehen",
        "fahr",
        "fahre",
        "fahren",
        "beweg",
        "bewege",
        "bewegen",
        "lauf",
        "laufe",
        "laufen",
        "vorne",
        "hinten",
        "geradeaus",
        "dreh",
        "drehe",
        "drehen",
        "umdrehen",
        "wende",
        "wenden",
        "links",
        "rechts",
        "stopp",
        "anhalten",
        "basis",
        "akku",
        "grad",
        "meter",
        "schau",
        "ist",
        "sind",
        "haben",
        "etwas",
        "deutsch",
        "weitermachen",
    }
)

STRICT_HIGH_GERMAN_ONLY: frozenset[str] = frozenset(
    {
        "ich",
        "nicht",
        "gehen",
        "fahren",
        "drehen",
        "bewegen",
        "vorwärts",
        "rückwärts",
        "ihnen",
        "können",
        "kannst",
        "heute",
        "etwas",
        "sich",
        "dich",
        "mich",
        "umdrehen",
        "zurück",
        "zurückkehren",
        "anhalten",
        "geradeaus",
        "weitermachen",
    }
)

_NON_GSW_LI_WORDS: frozenset[str] = frozenset(
    {
        "juli",
        "müsli",
        "muesli",
        "brokkoli",
        "chili",
        "famili",
        "stabili",
        "textili",
        "reptili",
        "fossili",
        "utensili",
    }
)

_SHARED_COGNATES: frozenset[str] = frozenset({"stop", "status", "ok", "okay", "halt", "robot", "meter"})

ENGLISH_MARKERS: frozenset[str] = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "good",
        "morning",
        "afternoon",
        "evening",
        "how",
        "are",
        "you",
        "can",
        "could",
        "would",
        "should",
        "please",
        "thank",
        "thanks",
        "what",
        "where",
        "when",
        "why",
        "who",
        "which",
        "the",
        "this",
        "that",
        "with",
        "from",
        "about",
        "weather",
        "today",
        "now",
        "speak",
        "english",
        "continue",
        "status",
        "battery",
        "go",
        "move",
        "drive",
        "walk",
        "forward",
        "forwards",
        "ahead",
        "backward",
        "backwards",
        "back",
        "left",
        "right",
        "turn",
        "rotate",
        "home",
        "meters",
        "metres",
        "check",
        "inspect",
        "look",
        "stop",
        "help",
        "tell",
        "translate",
        "translation",
        "welcome",
        "to",
        "into",
        "and",
        "for",
        "french",
        "chinese",
        "mandarin",
        "german",
        "swiss",
        "spanish",
        "czech",
        "around",
        "yes",
        "sure",
    }
)

SPANISH_MARKERS: frozenset[str] = frozenset(
    {
        "hola",
        "buenos",
        "días",
        "dias",
        "buenas",
        "tardes",
        "noches",
        "cómo",
        "estás",
        "está",
        "estamos",
        "gracias",
        "por",
        "favor",
        "puedes",
        "puede",
        "podemos",
        "hablar",
        "español",
        "espanol",
        "castellano",
        "continuar",
        "ahora",
        "qué",
        "dónde",
        "cuál",
        "cuánto",
        "tiempo",
        "clima",
        "hoy",
        "estado",
        "batería",
        "bateria",
        "avanza",
        "avanzar",
        "adelante",
        "hacia",
        "retrocede",
        "retroceder",
        "atrás",
        "atras",
        "gira",
        "girar",
        "rota",
        "izquierda",
        "derecha",
        "date",
        "vuelta",
        "metros",
        "metro",
        "grados",
        "detente",
        "detén",
        "parada",
        "emergencia",
        "vuelve",
        "regresa",
        "inicio",
        "claro",
        "vale",
        "enseguida",
        "sí",
        "el",
        "los",
        "las",
        "del",
        "al",
        "y",
        "luego",
        "después",
        "cinco",
        "cuatro",
        "tres",
        "dos",
        "para",
        "alto",
        "casa",
        "recto",
        "media",
        "inspecciona",
        "mira",
        "revisa",
        "traduce",
        "traducir",
        "checo",
        "alemán",
        "aleman",
        "francés",
        "frances",
        "inglés",
        "ingles",
        "chino",
    }
)

CZECH_MARKERS: frozenset[str] = frozenset(
    {
        "dobrý",
        "dobry",
        "ahoj",
        "čau",
        "děkuji",
        "dekuji",
        "díky",
        "diky",
        "prosím",
        "prosim",
        "jak",
        "máte",
        "máš",
        "jsem",
        "jsme",
        "jsou",
        "můžeš",
        "muzes",
        "můžete",
        "muzete",
        "mluvit",
        "česky",
        "cesky",
        "čeština",
        "cestina",
        "pokračovat",
        "pokracovat",
        "teď",
        "ted",
        "dnes",
        "počasí",
        "pocasi",
        "stav",
        "baterie",
        "robota",
        "jdi",
        "jděte",
        "jdete",
        "jeď",
        "jeďte",
        "jed",
        "pohyb",
        "posuň",
        "posun",
        "vpřed",
        "vpred",
        "dopředu",
        "dopredu",
        "rovně",
        "rovne",
        "dozadu",
        "zpět",
        "zpet",
        "zpátky",
        "zpatky",
        "couvej",
        "otoč",
        "otoc",
        "otočte",
        "otocte",
        "zatoč",
        "zatoc",
        "doleva",
        "vlevo",
        "doprava",
        "vpravo",
        "dokola",
        "čelem",
        "celem",
        "vzad",
        "stupňů",
        "stupnu",
        "metrů",
        "metru",
        "metry",
        "metr",
        "zastav",
        "zastavte",
        "stůj",
        "stuj",
        "nouzové",
        "nouzove",
        "zastavení",
        "vrať",
        "vrat",
        "základnu",
        "zakladnu",
        "domů",
        "domu",
        "zkontroluj",
        "podívej",
        "podivej",
        "prohlédni",
        "prohledni",
        "přelož",
        "preloz",
        "přeložit",
        "prelozit",
        "španělsky",
        "spanelsky",
        "anglicky",
        "německy",
        "nemecky",
        "francouzsky",
        "čínsky",
        "cinsky",
        "pak",
        "potom",
        "ano",
        "jasně",
        "jasne",
        "pět",
        "pet",
        "čtyři",
        "ctyri",
        "tři",
        "tri",
        "dva",
    }
)

_HINT_MAP: dict[str, str] = {
    "auto": "auto",
    "": "auto",
    "gsw": "gsw",
    "de": "de-DE",
    "de-de": "de-DE",
    "de-ch": "de-CH",
    "fr": "fr-FR",
    "fr-fr": "fr-FR",
    "fr-ch": "fr-CH",
    "en": "en-US",
    "en-us": "en-US",
    "en-gb": "en-GB",
    "es": "es-ES",
    "es-es": "es-ES",
    "spa": "es-ES",
    "es-us": "es-US",
    "es-mx": "es-US",
    "es-419": "es-US",
    "cs": "cs-CZ",
    "cs-cz": "cs-CZ",
    "cz": "cs-CZ",
    "ces": "cs-CZ",
    "cze": "cs-CZ",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-sg": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-hans-cn": "zh-CN",
    "cmn": "zh-CN",
    "cmn-cn": "zh-CN",
    "cmn-hans": "zh-CN",
    "cmn-hans-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "zh-hk": "zh-TW",
    "zh-mo": "zh-TW",
    "zh-hant": "zh-TW",
    "zh-hant-tw": "zh-TW",
    "cmn-tw": "zh-TW",
    "cmn-hant": "zh-TW",
    "cmn-hant-tw": "zh-TW",
}

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿĀ-ž']+|[0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TRADITIONAL_ONLY_CJK_RE = re.compile(
    r"[請問機狀態灣歡氣謝語確認緊開關車電網點對說時間東體現幫檢們這個嗎麼為剛]"
)


def clean_cjk_spacing(text: str) -> str:
    """Collapse artificial intra-CJK single spaces emitted by Chirp 3 auto-mode while preserving Latin spaces."""
    if not text or not _CJK_RE.search(text):
        return text
    out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。！？；：、,.?!])", "", out)
    out = re.sub(r"(?<=[，。！？；：、])\s+(?=[\u4e00-\u9fff])", "", out)
    out = re.sub(r"(?<=[\u4e00-\u9fff][,.?!])\s+(?=[\u4e00-\u9fff])", "", out)
    return out


def normalize_language_code(code: str | None) -> str:
    """Normalize short ISO-639-1 or BCP-47 tags into canonical platform language codes."""
    if not code:
        return "auto"
    cleaned = code.strip().replace("_", "-").lower()
    if cleaned in _HINT_MAP:
        return _HINT_MAP[cleaned]
    if any(tag in cleaned for tag in ("hant", "zh-tw", "zh-hk", "zh-mo", "cmn-tw")):
        return "zh-TW"
    if cleaned.startswith(("zh", "cmn")):
        return "zh-CN"
    if cleaned.startswith("en-gb"):
        return "en-GB"
    if cleaned.startswith("en"):
        return "en-US"
    if cleaned.startswith(("es-us", "es-mx", "es-419", "es-ar", "es-co", "es-cl", "es-pe")):
        return "es-US"
    if cleaned.startswith("es"):
        return "es-ES"
    if cleaned.startswith(("cs", "cz")):
        return "cs-CZ"
    if cleaned.startswith("fr-ch"):
        return "fr-CH"
    if cleaned.startswith("fr"):
        return "fr-FR"
    if cleaned.startswith("de-ch"):
        return "de-CH"
    if cleaned.startswith("de"):
        return "de-DE"
    return code.strip()


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def detect_swiss_dialect(tokens: list[str], fallback_dialect: str | None = None) -> str | None:
    """Infer Swiss German regional dialect code from distinctive dialect tokens."""
    best_dialect: str | None = None
    best_hits = 0
    token_set = set(tokens)
    for dialect_code, cues in DIALECT_LEXICAL_CUES.items():
        hits = len(token_set & cues)
        if hits > best_hits:
            best_hits = hits
            best_dialect = dialect_code
    return best_dialect or fallback_dialect


def detect_language_and_dialect_from_text(
    text: str,
    hint_language: str = "auto",
    default_dialect: str = "vs",
) -> tuple[str, str | None]:
    """Classify utterance text across Swiss German, German, French, English, Spanish, Czech, and Chinese."""
    norm_hint = normalize_language_code(hint_language)
    cjk_count = len(_CJK_RE.findall(text))
    tokens = tokenize_words(text)

    if cjk_count >= 2 or (cjk_count == 1 and not tokens):
        if norm_hint == "zh-CN":
            return "zh-CN", None
        is_trad = norm_hint == "zh-TW" or bool(_TRADITIONAL_ONLY_CJK_RE.search(text))
        return ("zh-TW" if is_trad else "zh-CN"), None

    if not tokens:
        resolved = norm_hint if norm_hint != "auto" else "en-US"
        return resolved, (default_dialect if resolved == "gsw" else None)

    token_set = set(tokens)

    # Strong phonological/orthographic patterns characteristic of Swiss German
    gsw_hits = len(token_set & SWISS_GERMAN_MARKERS)
    if tokens and tokens[0] == "gang":
        # Imperative "Gang ..." ("Go ...") at start of utterance is Swiss German (never a bare German noun)
        gsw_hits += 1
    for tok in tokens:
        if tok.startswith(("chö", "chü", "gsch", "schwiiz", "zürid", "bärnd")):
            gsw_hits += 1
        elif (
            len(tok) >= 5
            and tok.endswith(("li", "eli"))
            and tok not in _NON_GSW_LI_WORDS
            and any(c in tok for c in ("ä", "ö", "ü", "ch"))
        ):
            gsw_hits += 1

    fr_hits = len(token_set & FRENCH_MARKERS)
    if any(ch in text.lower() for ch in ("ç", "œ", "ê", "è", "ù", "î")):
        fr_hits += 2
    fr_ch_hits = len(token_set & SWISS_FRENCH_MARKERS)

    de_hits = len(token_set & STANDARD_GERMAN_MARKERS)
    strict_de_hits = len(token_set & STRICT_HIGH_GERMAN_ONLY)
    if "ß" in text:
        # 'ß' is Standard German (never used in Swiss orthography)
        de_hits += 3
        strict_de_hits += 2

    en_hits = len(token_set & ENGLISH_MARKERS)

    es_hits = len(token_set & SPANISH_MARKERS)
    if any(ch in text.lower() for ch in ("¿", "¡", "ñ")):
        es_hits += 3

    cs_hits = len(token_set & CZECH_MARKERS)
    if any(ch in text.lower() for ch in ("ě", "ř", "ů", "č", "š", "ž", "ď", "ť", "ň", "ý")):
        cs_hits += 3

    # Preserve non-English acoustic hint on single-word cross-lingual cognates like "Stop!" or "Status?"
    if (
        norm_hint in {"de-DE", "de-CH", "gsw", "fr-FR", "fr-CH", "es-ES", "es-US", "cs-CZ"}
        and token_set <= _SHARED_COGNATES
    ):
        if norm_hint == "gsw":
            return "gsw", detect_swiss_dialect(tokens, fallback_dialect=default_dialect)
        return norm_hint, None

    # Unambiguous Swiss German markers that survive partial ASR normalization (e.g. Chirp 3 'auto')
    swiss_greetings = {"grüezi", "grüessech", "mitenand", "zäme", "zämu", "sali", "hoi"}
    strong_gsw_grammar = SWISS_GERMAN_MARKERS - swiss_greetings - {"guet", "guete", "uf", "ade"}
    has_strong_gsw = len(token_set & strong_gsw_grammar) > 0 or (bool(tokens) and tokens[0] == "gang")
    has_swiss_greeting = len(token_set & swiss_greetings) > 0
    explicitly_hochdeutsch = "ß" in text or "hochdeutsch" in token_set

    # 1. Czech wins when clearly highest (or tied when acoustic/session hint is Czech)
    if cs_hits >= 1 and (
        cs_hits > max(en_hits, fr_hits, es_hits, gsw_hits, de_hits)
        or (norm_hint == "cs-CZ" and cs_hits >= max(en_hits, fr_hits, es_hits, gsw_hits, de_hits))
    ):
        return "cs-CZ", None

    # 2. Spanish wins when clearly highest (or tied when acoustic/session hint is Spanish)
    if es_hits >= 1 and (
        es_hits > max(en_hits, fr_hits, cs_hits, gsw_hits, de_hits)
        or (norm_hint in {"es-ES", "es-US"} and es_hits >= max(en_hits, fr_hits, cs_hits, gsw_hits, de_hits))
    ):
        lang = "es-US" if norm_hint == "es-US" else "es-ES"
        return lang, None

    # 3. English wins when clearly highest
    if en_hits > max(fr_hits, es_hits, cs_hits, gsw_hits, de_hits) and en_hits >= 1:
        lang = "en-GB" if norm_hint == "en-GB" else "en-US"
        return lang, None

    # 4. French wins when clearly highest
    if fr_hits > max(gsw_hits, de_hits, en_hits, es_hits, cs_hits) and fr_hits >= 1:
        lang = "fr-CH" if (fr_ch_hits > 0 or norm_hint == "fr-CH") else "fr-FR"
        return lang, None

    # 5. Swiss Standard German ("Grüezi, gehen Sie bitte 5 m vorwärts...") stays High German (de-CH), not gsw
    if has_swiss_greeting and not has_strong_gsw and strict_de_hits >= 2:
        return "de-CH", None

    # 6. Swiss German wins when strong dialect grammar or Swiss greetings appear (unless explicit de-CH / Hochdeutsch)
    if not explicitly_hochdeutsch and gsw_hits >= max(en_hits, fr_hits, es_hits, cs_hits):
        if has_strong_gsw or (has_swiss_greeting and norm_hint != "de-CH") or (gsw_hits >= 1 and gsw_hits >= de_hits):
            dialect = detect_swiss_dialect(tokens, fallback_dialect=default_dialect)
            return "gsw", dialect

    # 7. Standard German
    if de_hits > 0 and de_hits >= max(en_hits, fr_hits, es_hits, cs_hits):
        if norm_hint == "de-CH":
            return "de-CH", None
        return "de-DE", None

    # 8. Fall back to acoustic ASR hint or explicit user selection when text has no decisive markers
    if norm_hint == "gsw":
        return "gsw", detect_swiss_dialect(tokens, fallback_dialect=default_dialect)
    if norm_hint in {
        "en-US",
        "en-GB",
        "es-ES",
        "es-US",
        "cs-CZ",
        "de-CH",
        "de-DE",
        "fr-CH",
        "fr-FR",
        "zh-CN",
        "zh-TW",
    }:
        return norm_hint, None

    return "de-DE", None
