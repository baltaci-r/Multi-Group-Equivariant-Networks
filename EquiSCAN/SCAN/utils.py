import re
from tqdm import tqdm


def clean_data(data):
    cleaned = []
    for example in data:
        for i, tok in enumerate(example):
            if tok in ["opposite", "around"]:
                example.pop(i - 1)
                example.pop(i)
        cleaned.append(' '.join(example))
    return cleaned


def repeat(example):
    for s, n in [(" twice", 2), (" thrice", 3), ("", 1)]:
        m = len(re.findall(f"(\[[^\[\]]*\]){s}", example))
        for _ in range(m):
            g = re.search(f"(\[[^\[\]]*\]){s}", example)
            r = ' '.join([g.group(1)[2:-2]] * n)
            example = f"{example[:g.start(0)].strip()} {r} {example[g.end(0) + 1:].strip()}"
    return example.strip()


def match(example):
    g = re.match("(\([^\(\)]*\)) and (\([^\(\)]*\))", example)
    if g:
        return repeat(f"{g.group(1)[2:-2]} {g.group(2)[2:-2]}")
    g = re.match("(\([^\(\)]*\)) after (\([^\(\)]*\))", example)
    if g:
        return repeat(f"{g.group(2)[2:-2]} {g.group(1)[2:-2]}")
    g = re.match("\(.*\)", example)
    if g:
        return repeat(example[1:-1].strip())


def _translate(map, example):
    for k, v in map.items():
        example = example.replace(k, v)
    return match(example)


def translate(map, data):

    pairs = []
    for example in tqdm(data):
        pair = "IN: " + ' '.join(re.sub("\(|\)|\[|\]", "", example).split())
        example = _translate(map, example)
        pair += " OUT: " + example
        pairs.append(pair)
    return pairs
