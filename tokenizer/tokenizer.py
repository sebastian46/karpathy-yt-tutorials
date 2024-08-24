import regex as re

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

class BasicTokenizer():

    def __init__(self):
        # initialize the variables needed
        super().__init__()
        # vocab is going to be the byte representation of 0-255
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

    def train(self, text, vocab_size, regex, verbose=False):
        # Read text
        regex = re.compile(regex)
        tokens = re.findall(regex, text)
        ids = [list(ch.encode("utf-8")) for ch in tokens]
        print(ids[:10])
        print(self.decode(ids[:1][0]))

        # tokens = text.encode("utf-8") # raw bytes
        # tokens =  list(map(int,tokens)) # convert to a list of integers in range 0..255 for convenience
        # print(tokens[:30])
        # ids = list(tokens) # copy so we don't destroy the original list
        # Create merges until we have vocab_size or run out
        num_merges = vocab_size - 256
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose: print(f"merging {pair} into a new token {idx}")
            # ids = merge(ids, pair, idx)
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

# text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
with open('taylorswift.txt', 'r', encoding='utf-8') as file:
    # Read the contents of the file
    text = file.read()
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
basicTokenizer = BasicTokenizer()
basicTokenizer.train(text, 300, GPT4_SPLIT_PATTERN, False)
for (p0, p1), idx in basicTokenizer.merges.items():
    print(f"[{basicTokenizer.decode([p0])}] [{basicTokenizer.decode([p1])}] => {idx}")
