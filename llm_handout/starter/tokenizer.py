

"""
Unified UTF-8 Greedy Tokenizer (Characters + Subwords + Byte Fallback)
Vocab size: Configurable (Default 512).
0-255: Raw UTF-8 bytes.
256+: Top frequent Unicode characters AND subwords (e.g., "में").

Fully lossless, greedy longest-match encoding.
"""
import json
import os
import re
from collections import Counter

class GreedyByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.item_to_id = {}
        self.id_to_bytes = {}
        self.sorted_vocab = []

    def train(self, text, target_vocab_size=512):
        """Scans the text for frequent characters AND whole subwords."""
        counts = Counter()
        
        # 1. Count frequent non-ASCII continuous sequences (Subwords)
        subwords = re.findall(r'[^\x00-\x7F]+', text)
        for w in subwords:
            counts[w] += 1
            
        # 2. Count individual characters
        for char in text:
            if ord(char) > 127:
                counts[char] += 1
                
        # Take the top items to fill our remaining vocab budget
        target_extra = target_vocab_size - 256
        top_items = [item for item, freq in counts.most_common(target_extra)]
        
        # 3. Sort by length descending! (Greedy Longest-Match)
        top_items.sort(key=len, reverse=True)
        self.sorted_vocab = top_items
        
        # Assign IDs starting from 256
        for i, item in enumerate(self.sorted_vocab):
            idx = 256 + i
            self.item_to_id[item] = idx
            self.id_to_bytes[idx] = list(item.encode("utf-8"))
        
        self.vocab_size = 256 + len(self.sorted_vocab)

    def encode(self, text):
        """Encodes string using greedy longest-prefix match."""
        ids = []
        i = 0
        n = len(text)
        
        while i < n:
            match_found = False
            for item in self.sorted_vocab:
                if text.startswith(item, i):
                    ids.append(self.item_to_id[item])
                    i += len(item)
                    match_found = True
                    break
            
            if not match_found:
                raw_bytes = list(text[i].encode("utf-8"))
                ids.extend(raw_bytes)
                i += 1
                
        return ids

    def decode(self, ids):
        """Decodes list of ints back to string losslessly."""
        byte_list = []
        for idx in ids:
            if idx < 256:
                byte_list.append(idx)
            else:
                byte_list.extend(self.id_to_bytes[idx])
        return bytes(byte_list).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "greedy_subword_fallback",
                "vocab_size": self.vocab_size,
                "item_to_id": self.item_to_id,
                "sorted_vocab": self.sorted_vocab
            }, f, ensure_ascii=False)


def load(path=None):
    """
    Called by evaluate.py and train.py with NO arguments.
    Just loads the pre-trained vocab.json.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "vocab.json")
    
    tok = GreedyByteTokenizer()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.vocab_size = data.get("vocab_size", 256)
        tok.item_to_id = data.get("item_to_id", {})
        tok.sorted_vocab = data.get("sorted_vocab", [])
        
        tok.id_to_bytes = {
            int(v): list(k.encode("utf-8")) for k, v in tok.item_to_id.items()
        }
    return tok


# ---------------------------------------------------------
# NEW: Run this file directly to train the tokenizer ONCE.
# This keeps train.py completely untouched.
# ---------------------------------------------------------
if __name__ == "__main__":
    corpus_path = "../data/train_corpus.txt"
    vocab_path = "vocab.json"
    
    print(f"Reading {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("Training Greedy Char-Subword tokenizer (vocab_size=512)...")
    tok = GreedyByteTokenizer()
    tok.train(text, target_vocab_size=512)
    tok.save(vocab_path)
    print(f"Saved {vocab_path}! You can now run the standard train.py.")