import os
import json
import re
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):

    text = text.lower()

    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    out = []
    for tok in tokens:
        if not tok.isalpha():
            continue
        if tok in STOPWORDS:
            continue
        s = stemmer.stem(tok)
        if s:
            out.append(s)
    return out


doc_folder = "documents"
documents = {}

for i in range(1, 4):
    path = os.path.join(doc_folder, f"document{i}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing expected file: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        documents[i] = f.read()


class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.keys = []      
        self.children = []    
        self.leaf = leaf
        self.files = {}        


class BTree:
    def __init__(self, t=3):
        self.t = t
        self.root = BTreeNode(t, leaf=True)

    def search(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return node, i

        if node.leaf:
            return None

        return self.search(node.children[i], key)

    def split_child(self, parent, idx, child):
        t = self.t
        new_node = BTreeNode(t, leaf=child.leaf)
        median_key = child.keys[t - 1]
        parent.keys.insert(idx, median_key)
        parent.children.insert(idx + 1, new_node)

        new_node.keys = child.keys[t:]
        child.keys = child.keys[: t - 1]

        if child.leaf:
            for k in list(new_node.keys):
                if k in child.files:
                    new_node.files[k] = child.files[k]
                    del child.files[k]
        else:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

    def insert(self, key, file_id):

        root = self.root

        found = self.search(root, key)
        if found:
            node, pos = found
            if node.leaf:
                node.files[key].add(file_id)
                return
            else:
                child_idx = pos + 1
                cur = node.children[child_idx]
                while not cur.leaf:
                    i = 0
                    while i < len(cur.keys) and key > cur.keys[i]:
                        i += 1
                    cur = cur.children[i]
                if key not in cur.keys:
                    cur.keys.append(key)
                    cur.keys.sort()
                if key not in cur.files:
                    cur.files[key] = set()
                cur.files[key].add(file_id)
                return

        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(root)
            self.split_child(new_root, 0, root)
            self.root = new_root
            self._insert_non_full(new_root, key, file_id)
        else:
            self._insert_non_full(root, key, file_id)

    def _insert_non_full(self, node, key, file_id):
        if node.leaf:
            if key not in node.keys:
                i = len(node.keys) - 1
                node.keys.append(None)
                while i >= 0 and key < node.keys[i]:
                    node.keys[i + 1] = node.keys[i]
                    i -= 1
                node.keys[i + 1] = key
                node.files[key] = set([file_id])
            else:
                node.files[key].add(file_id)
            return

        i = len(node.keys) - 1
        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1

        if len(node.children[i].keys) == 2 * self.t - 1:
            self.split_child(node, i, node.children[i])
            if key > node.keys[i]:
                i += 1

        self._insert_non_full(node.children[i], key, file_id)

    def collect_leaves(self, node=None, leaves=None):
        if leaves is None:
            leaves = []
        if node is None:
            node = self.root
        if node.leaf:
            leaves.append(node)
        else:
            for c in node.children:
                self.collect_leaves(c, leaves)
        return leaves

    def print_leaves(self, node=None):
        if node is None:
            node = self.root
        if node.leaf:
            print("Leaf:", node.keys)
            for k in node.keys:
                print(f"   {k} -> files: {sorted(node.files.get(k, []))}")
            return
        for c in node.children:
            self.print_leaves(c)


inverted_index = defaultdict(set)  
btree = BTree(t=3)
process_trace = []


def log(step, detail):
    process_trace.append({"step": step, "detail": detail})


for doc_id, text in documents.items():
    log("DOCUMENT", f"Doc {doc_id} loaded (len={len(text)}).")
    tokens = preprocess(text)
    log("TOKENIZE", f"Doc {doc_id} tokens sample: {tokens[:30]}")

    unique_tokens = set(tokens) 
    for tok in unique_tokens:
        inverted_index[tok].add(doc_id)
        log("SID_INSERT", f"term '{tok}' in doc {doc_id}")

        btree.insert(tok, doc_id)
        log("BTREE_INSERT", f"Inserted/updated '{tok}' in B-Tree for doc {doc_id}")


def print_sid_table():
    print("\n=== Standard Inverted Index (term -> [doc_ids]) ===\n")
    for term in sorted(inverted_index.keys()):
        print(f"{term:<20} -> {sorted(inverted_index[term])}")


print_sid_table()

os.makedirs("output", exist_ok=True)
with open(os.path.join("output", "sid.json"), "w", encoding="utf-8") as f:
    json.dump({k: list(v) for k, v in inverted_index.items()}, f, ensure_ascii=False, indent=2)

with open(os.path.join("output", "process_trace.json"), "w", encoding="utf-8") as f:
    json.dump(process_trace, f, ensure_ascii=False, indent=2)


print("\n=== B-Tree Leaves ===\n")
btree.print_leaves()


class TreeVisualizer:
    def __init__(self, btree, postings):
        self.btree = btree
        self.postings = postings
        self.pos = {}
        self.level_y = 130
        self.w = 140
        self.h = 50

        self.root = tk.Tk()
        self.root.title("B-Tree Visualizer")

        self.canvas = tk.Canvas(self.root, width=1400, height=800, bg="white", scrollregion=(0, 0, 3000, 2000))
        hbar = tk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._calc_width(self.btree.root)
        self._place(self.btree.root, 0)
        self._draw()
        self.root.mainloop()

    def _calc_width(self, node):
        if node.leaf:
            self.pos[id(node)] = {"width": 1}
            return 1
        total = 0
        for c in node.children:
            total += self._calc_width(c)
        self.pos[id(node)] = {"width": max(total, 1)}
        return self.pos[id(node)]["width"]

    def _place(self, node, start):
        if node.leaf:
            self.pos[id(node)]["x"] = start
            self.pos[id(node)]["d"] = self._depth(node)
            return start + 1

        cur = start
        for c in node.children:
            cur = self._place(c, cur)

        xs = [self.pos[id(c)]["x"] for c in node.children]
        center = (xs[0] + xs[-1]) / 2 if xs else start
        self.pos[id(node)]["x"] = center
        self.pos[id(node)]["d"] = self._depth(node)
        return cur

    def _depth(self, node):
        d = 0
        cur = node
        while not cur.leaf:
            cur = cur.children[0]
            d += 1
        return d

    def _pixel(self, node):
        info = self.pos[id(node)]
        ux = info["x"]
        depth = info["d"]
        max_u = max(p["x"] for p in self.pos.values())
        cw = max(self.canvas.winfo_width(), 1400)
        margin = 80
        avail = cw - 2 * margin
        px = margin + ux * (avail / max(1, max_u))
        py = 80 + depth * self.level_y
        return px, py

    def _draw(self):
        self._draw_edges(self.btree.root)
        self._draw_nodes(self.btree.root)

    def _draw_edges(self, node):
        if not node.children:
            return
        px, py = self._pixel(node)
        for c in node.children:
            cx, cy = self._pixel(c)
            self.canvas.create_line(px, py + self.h // 2, cx, cy - self.h // 2, fill='gray', width=1)
            self._draw_edges(c)

    def _draw_nodes(self, node):
        x, y = self._pixel(node)
        rect = self.canvas.create_rectangle(
            x - self.w // 2, y - self.h // 2,
            x + self.w // 2, y + self.h // 2,
            fill='#e1f5fe', outline='#0288d1', width=2
        )

        label = ", ".join(node.keys)
        txt_id = self.canvas.create_text(x, y, text=label, font=('Consolas', 10, 'bold'))

        def on_click(event, keys=node.keys):
            lines = []
            for k in keys:
                p = dict(self.postings.get(k, {}))
                lines.append(f"{k}: {p}" if p else f"{k}: (no postings)")
            messagebox.showinfo("Postings", "\n".join(lines) if lines else "No data")

        self.canvas.tag_bind(rect, "<Button-1>", on_click)
        self.canvas.tag_bind(txt_id, "<Button-1>", on_click)

        for c in node.children:
            self._draw_nodes(c)


TreeVisualizer(btree, inverted_index)