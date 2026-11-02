"""Fetch a public-domain English corpus from GITenberg (Project Gutenberg mirrors on GitHub).

Only hosts reachable from the sandbox are used (github.com / api.github.com).
Gutenberg header/footer boilerplate is stripped. Output: data/raw/books/<id>.txt
"""
import json, os, re, sys, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "books")
UA = {"User-Agent": "cadence-lm-research"}
TITLES = [
    "Frankenstein", "Pride and Prejudice", "Dracula", "A Tale of Two Cities",
    "Great Expectations", "Heart of Darkness", "The Time Machine", "The War of the Worlds",
    "The Picture of Dorian Gray", "The Yellow Wallpaper", "Alices Adventures in Wonderland",
    "Treasure Island", "Moby Dick", "The Adventures of Sherlock Holmes",
    "The Hound of the Baskervilles", "Jane Eyre", "Wuthering Heights", "The Scarlet Letter",
    "Anne of Green Gables", "A Christmas Carol", "The Wonderful Wizard of Oz",
    "Adventures of Huckleberry Finn", "The Adventures of Tom Sawyer", "Emma",
    "Sense and Sensibility", "Mansfield Park", "Middlemarch", "The Secret Garden",
    "Little Women", "The Jungle Book", "The Call of the Wild", "White Fang",
    "Gullivers Travels", "Robinson Crusoe", "The Count of Monte Cristo", "Les Miserables",
    "The Three Musketeers", "Crime and Punishment", "Anna Karenina", "War and Peace",
    "The Odyssey", "Walden", "The Turn of the Screw",
    "The Strange Case of Dr Jekyll and Mr Hyde", "A Study in Scarlet",
    "The Sign of the Four", "Nostromo", "Lord Jim", "The Age of Innocence",
    "The House of Mirth", "Sister Carrie", "The Red Badge of Courage", "The Awakening",
    "The Innocence of Father Brown", "The Lost World", "Kim", "Ivanhoe",
    "David Copperfield", "Bleak House", "Hard Times", "Oliver Twist",
    "Vanity Fair", "The Woman in White", "The Moonstone", "Far from the Madding Crowd",
    "Tess of the dUrbervilles", "The Mayor of Casterbridge", "Silas Marner",
    "Persuasion", "Northanger Abbey", "A Room with a View", "Howards End",
    "Sons and Lovers", "Dubliners", "The Great Gatsby", "Of Human Bondage",
    "The Prisoner of Zenda", "King Solomons Mines", "The Thirty-Nine Steps",
    "The Adventures of Robin Hood", "A Connecticut Yankee in King Arthurs Court",
    "The Prince and the Pauper", "Puddnhead Wilson", "The Jungle",
    "The Custom of the Country", "The Beautiful and Damned", "Main Street",
    "Babbitt", "My Antonia", "O Pioneers", "The Age of Innocence",
    "The Ambassadors", "The Portrait of a Lady", "Washington Square",
    "The Europeans", "The Bostonians", "Bleak House", "Little Dorrit",
    "Barnaby Rudge", "Dombey and Son", "The Old Curiosity Shop", "Martin Chuzzlewit",
    "The Mystery of Edwin Drood", "Wives and Daughters", "North and South",
    "Cranford", "The Tenant of Wildfell Hall", "Agnes Grey", "Villette",
    "Shirley", "The Professor", "Under Western Eyes", "The Secret Agent",
    "Typhoon", "The Nigger of the Narcissus", "Victory", "Chance",
    "The Arrow of Gold", "The Rescue", "An Outcast of the Islands",
    "Almayer", "The Shadow Line", "The Nigger of the Narcissus",
]


def get(url, raw=False):
    hdrs = dict(UA)
    if raw:
        hdrs["Accept"] = "application/vnd.github.raw"
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read() if raw else json.load(r)


def strip_gutenberg(s: str) -> str:
    m = re.search(r"\*\*\* *START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", s, re.S)
    if m:
        s = s[m.end():]
    m = re.search(r"\*\*\* *END OF (THE|THIS) PROJECT GUTENBERG", s, re.S)
    if m:
        s = s[:m.start()]
    return s.strip() + "\n"


def main(limit_bytes=48_000_000, sleep=8.0):
    os.makedirs(OUT, exist_ok=True)
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"existing {total/1e6:.1f} MB", flush=True)
    for title in TITLES:
        if total >= limit_bytes:
            break
        try:
            q = urllib.parse.quote(f"org:GITenberg {title} in:name")
            d = get(f"https://api.github.com/search/repositories?q={q}&per_page=5")
        except Exception as e:
            print("search-fail", title, e, flush=True)
            time.sleep(30)
            continue
        cands = []
        for r in d.get("items", []):
            m = re.search(r"_(\d+)$", r["name"])
            if m:
                cands.append((m.group(1), r["full_name"]))
        if not cands:
            print("miss", title, flush=True)
            time.sleep(sleep)
            continue
        bid, repo = cands[0]
        path = os.path.join(OUT, f"{bid}.txt")
        if os.path.exists(path):
            print("skip", repo, flush=True)
            time.sleep(sleep)
            continue
        try:
            payload = get(f"https://api.github.com/repos/{repo}/contents/{bid}.txt", raw=True)
            body = payload.decode("utf-8", "replace")
            if body.lstrip().startswith("{") and '"content"' in body:
                import base64
                body = base64.b64decode(json.loads(body)["content"]).decode("utf-8", "replace")
            text = strip_gutenberg(body)
            if len(text) < 5000:
                print("too-small", repo, len(text), flush=True)
                time.sleep(sleep)
                continue
            with open(path, "w") as f:
                f.write(text)
            total += len(text)
            print(f"ok {repo} {len(text)/1e6:.2f} MB  total {total/1e6:.1f} MB", flush=True)
        except Exception as e:
            print("dl-fail", repo, e, flush=True)
        time.sleep(sleep)


if __name__ == "__main__":
    main(*[float(a) for a in sys.argv[1:]])
