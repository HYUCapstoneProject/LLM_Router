"""조건 비교 규칙. 두 질문의 '조건'을 뽑아 서로 어긋나면 캐시 재사용을 막는다.

Krites 의 judge 가 확인하는 "개체·조건(이름, 장소, 시간, 수량) 일치"를 규칙으로 흉내 낸다. 규칙은 일반 범주로만
쓰고, 어떤 규칙을 켤지는 calibration 에서 정한다(PLAN2.md 3절).

check(a, b) -> {rule_name: True(어긋남) / False}
"""
import re

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000, "million": 1_000_000, "dozen": 12,
}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
ALIASES = {"js": "javascript", "ts": "typescript", "py": "python", "postgres": "postgresql",
           "golang": "go", "nodejs": "node", "osx": "macos", "k8s": "kubernetes", "cpp": "c++"}
TECH = {
    "python", "java", "javascript", "typescript", "go", "rust", "c", "c++", "c#", "kotlin", "swift", "ruby",
    "php", "scala", "perl", "mysql", "postgresql", "sqlite", "mongodb", "redis", "sqlserver", "oracle",
    "react", "vue", "angular", "svelte", "django", "flask", "spring", "node", "npm", "yarn", "pip", "conda",
    "docker", "kubernetes", "git", "linux", "windows", "macos", "ubuntu", "powershell", "bash", "pandas",
    "numpy", "axios", "selenium", "jquery", "excel",
}
NEGATION = {"not", "no", "without", "never", "none", "non", "cannot", "unsorted"}
ANTONYMS = [
    ({"largest", "biggest", "highest", "maximum", "max", "most", "tallest", "farthest", "deepest"},
     {"smallest", "lowest", "minimum", "min", "least", "shortest", "closest", "shallowest"}),
    ({"first", "beginning", "start", "head", "front"}, {"last", "end", "tail"}),
    ({"create", "add", "install", "insert", "open", "enable", "build", "built", "start"},
     {"delete", "remove", "drop", "uninstall", "close", "disable", "destroy", "stop", "abort"}),
    ({"upper", "uppercase", "capitalize"}, {"lower", "lowercase"}),
    ({"ascending"}, {"descending"}),
    ({"increase", "increment"}, {"decrease", "decrement"}),
    ({"synonym"}, {"antonym", "opposite"}),
    ({"advantages", "advantage", "pros", "benefits", "purpose"}, {"disadvantages", "disadvantage", "cons",
                                                                   "downsides", "drawbacks", "weaknesses"}),
    ({"accept"}, {"decline", "reject"}),
    ({"read", "load", "import", "download", "pull", "parse"}, {"write", "save", "export", "upload", "push", "stringify"}),
    ({"encrypt", "encode"}, {"decrypt", "decode"}),
    ({"even"}, {"odd"}),
    ({"before"}, {"after"}),
    ({"left"}, {"right"}),
    ({"soft", "runny"}, {"hard"}),
    ({"cool", "cold"}, {"warm", "hot", "heat"}),
    ({"float", "floats"}, {"sink", "sinks"}),
    ({"active"}, {"passive"}),
    ({"comparative"}, {"superlative"}),
    ({"deep"}, {"shallow"}),
    ({"appear", "appears", "show"}, {"disappear", "disappears", "hide"}),
    ({"fall", "fell", "end", "ended"}, {"build", "built", "start", "started", "begin"}),
]
FORMATS = {"table", "bullet", "bullets", "bulleted", "json", "yaml", "xml", "csv", "sentence", "sentences",
           "paragraph", "paragraphs", "haiku", "limerick", "poem", "tweet", "pressrelease", "essay"}
TIME = {"today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening", "night", "midnight",
        "noon", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "spring",
        "summer", "autumn", "winter", "daily", "weekly", "monthly", "yearly", "sunset", "sunrise"}
TIME_NORMALIZE = {"am": "morning", "pm": "evening", "nightly": "night"}
SCOPE = {"only", "all", "every", "each", "including", "single", "both", "average", "worst", "internal",
         "native", "entire", "whole", "remote", "local", "nested", "arbitrarily", "unique", "latest", "specific"}
STOP = set("a an the of in on to for is are do does did i my me you your what how which why when who whom can "
           "should would could with and or it its that this be by at from as into using use way get".split())


def tokens(text):
    t = text.lower().replace("n't", " not").replace("press release", "pressrelease").replace("sql server", "sqlserver")
    raw = re.findall(r"c\+\+|c#|[a-z]+|\d+(?:\.\d+)?", t)
    return [ALIASES.get(w, w) for w in raw]


def numbers(toks):
    out = set()
    for w in toks:
        if re.fullmatch(r"\d+(?:\.\d+)?", w):
            out.add(float(w))
        elif w in NUM_WORDS:
            out.add(float(NUM_WORDS[w]))
    return out


def proper_nouns(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", text)
    out = set()
    for i, w in enumerate(words):
        if i > 0 and w[0].isupper() and w.lower() not in STOP and w not in {"I"}:
            out.add(w.lower().strip(".-"))
    return out


def direction(text):
    m = re.search(r"\b(convert|converting|change|turn|transform|cast|rewrite)\b(.+?)\b(to|into)\b(.+)", text.lower())
    if not m:
        return None
    src = {w for w in tokens(m.group(2)) if w not in STOP}
    dst = {w for w in tokens(m.group(4)) if w not in STOP}
    return src, dst


def both_differ(x, y):
    return bool(x) and bool(y) and x != y


def check(a, b):
    ta, tb = tokens(a), tokens(b)
    sa, sb = set(ta), set(tb)
    res = {}
    res["numbers"] = both_differ(numbers(ta), numbers(tb))
    res["ordinal"] = both_differ({ORDINALS[w] for w in ta if w in ORDINALS}, {ORDINALS[w] for w in tb if w in ORDINALS})
    res["tech"] = both_differ(sa & TECH, sb & TECH)
    pa, pb = proper_nouns(a), proper_nouns(b)
    res["proper_noun"] = bool((pa | pb) and any((p in sa) != (p in sb) for p in (pa | pb)
                                                 if re.fullmatch(r"[a-z0-9+#]+", p)))
    res["negation"] = bool(sa & NEGATION) != bool(sb & NEGATION)
    res["antonym"] = any((sa & x and sb & y and not (sa & y) and not (sb & x)) or
                         (sa & y and sb & x and not (sa & x) and not (sb & y)) for x, y in ANTONYMS)
    da, db = direction(a), direction(b)
    res["direction"] = bool(da and db and (da[0] & db[1]) and (da[1] & db[0]))
    res["format"] = both_differ(sa & FORMATS, sb & FORMATS)
    time_a = {TIME_NORMALIZE.get(w, w) for w in ta} & TIME
    time_b = {TIME_NORMALIZE.get(w, w) for w in tb} & TIME
    res["time_both"] = both_differ(time_a, time_b)
    res["time_any"] = time_a != time_b
    res["scope"] = (sa & SCOPE) != (sb & SCOPE)
    return res


RULES = ["numbers", "ordinal", "tech", "proper_noun", "negation", "antonym", "direction", "format",
         "time_both", "time_any", "scope"]
