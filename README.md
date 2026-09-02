# ctximg

Find photos by describing what they show.

Nothing here matches filenames, folders, tags, or keywords. A CLIP model maps
every photo and every query into one shared vector space, and search is cosine
similarity, ranked. That is the whole retrieval model — one score per photo,
sorted descending, no filters and no boolean operators.

Everything runs locally. Your photos are never uploaded, and the page makes no
external requests.

## The app

Double-click the desktop icon. Your browser opens on the folder list: tick the
folders to search, type a description, get a ranked grid of photos.

```powershell
ctximg install-shortcut     # puts the icon on your desktop
```

The icon runs `pythonw`, so no console window appears. Clicking it while the app
is already running just opens a tab rather than starting a second server, and it
keeps running when you close the tab — quit it from the tray icon.

In the app you can:

- **Tick and untick folders** to choose what a search covers. Results are
  labelled with the folder they came from.
- **Add a folder** with the Browse button (a real Windows folder picker, since
  the server is on your own machine) or by pasting a path. It indexes in the
  background with a progress bar.
- **Update or remove** any folder. Removing deletes only its index — never your
  photos.

`ctximg app` starts the same app from a terminal.

## The terminal

The terminal works on the folder you are standing in. Searching across folders
lives in the app, where you can actually see the photos rather than reading
fifty filenames.

```powershell
cd C:\Users\you\Pictures\2019
ctximg
```

```
  C:\Users\you\Pictures\2019
  304 photos indexed and ready.

  ViT-L-14/laion2b_s32b_b82k on cuda
  ctximg - describe what you are looking for. :help for commands.

> someone laughing at a dinner table
   1  0.284  reunion\IMG_2841.jpg
   2  0.241  IMG_0102.jpg
>
```

The first visit offers to index; later visits go straight to the prompt and tell
you if photos were added or removed since. The model loads once per session, so
every query after the first is instant.

| Command | What it does |
|---|---|
| `ctximg` | Search this folder. Offers to index it the first time. |
| `ctximg index` | Index or catch up this folder. `--rebuild` re-embeds everything. |
| `ctximg forget` | Delete this folder's index. Photos untouched. |
| `ctximg web` | Open this folder in the app. |
| `ctximg app` | Launch the app. |
| `ctximg list` | Every folder indexed so far. |
| `ctximg stats` | Counts, model, index size for this folder. |
| `ctximg config show` | Settings, live GPU/CPU state, and the quality tiers. |
| `ctximg config set tier <fast\|balanced\|best>` | Choose accuracy vs speed. |
| `ctximg doctor` | Python, torch, GPU, model, folders. |
| `ctximg install-shortcut` | Desktop icon (`--start-menu` for the Start menu). |

At the prompt: `:open 3` opens a result in your viewer, `:web` brings the folder
into the app, `:reindex` catches up, `:q` quits.

A run that cannot ask — piped or scripted — never starts an index on its own.
Pass `-y` to mean yes up front.

## Install

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -e .
```

## Quality tiers

Accuracy and indexing time trade directly against each other, so you choose.
Pick a tier in **Settings** in the app, or from the terminal:

```powershell
ctximg config set tier balanced
```

| Tier | Model | Vector | Input | Speed* |
|---|---|---|---|---|
| **Fast** | `ViT-L-14` (LAION-2B) | 768 | 224px | ~100 img/s |
| **Balanced** | `ViT-L-16-SigLIP2-384` | 1024 | 384px | ~49 img/s |
| **Most accurate** | `ViT-SO400M-16-SigLIP2-384` | 1152 | 384px | ~32 img/s |

\* Measured on an RTX 3060 Laptop (6 GB) at fp16. The app replaces these with
what your machine actually achieves as soon as it has indexed something.

SigLIP 2 is a large step up from the LAION CLIP models, and the two things that
matter most for accuracy are both in the upper tiers: **384px input** (nearly
three times the visual tokens, so small detail and text in a photo register)
and a **wider vector**.

Bigger models than these were tried and rejected. `ViT-H-14-378` (3.8 GB) and
`ViT-gopt-16-SigLIP2-384` (3.7 GB) do not fit once the display has taken its
share of a 6 GB card; they spill to system memory and collapse to about
1 img/s. `ctximg config show` tells you what will actually run.

Changing tier means every folder must be indexed again - vectors from two
models cannot be compared - so the app tells you how long that will take before
you commit.

## GPU or CPU

The device is chosen at runtime: CUDA if torch can see a GPU, CPU otherwise,
and **fp16 on a GPU**. Vectors are stored as float16 regardless, so computing
in fp32 and discarding the precision costs memory and speed for nothing: fp16
is roughly three times faster here, and it is what makes the larger models fit
at all. Batch size is scaled to your VRAM, and a CUDA out-of-memory splits the
batch and retries rather than losing the run.

The catch is that `pip install torch` on Windows gives you a **CPU-only wheel**,
which reports no GPU even when you have one, so auto-detection can only ever
pick CPU. If you have an NVIDIA card, install the CUDA build:

```powershell
.venv\Scripts\python -m pip uninstall -y torch torchvision
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then `ctximg doctor` should report `cuda available True` and `device cuda`.

## Watching it work

The app header carries a live pulse - GPU or CPU load, or indexing progress
with throughput - so you can tell at a glance whether anything is happening.
**Settings** opens the full readout: what is configured versus what is actually
running, the GPU's utilisation, memory, temperature and power draw, CPU and RAM,
and during an index run the rate and time remaining.

Adding a folder prices the job first: how many photos, how many gigabytes, and
roughly how long at the current tier's speed.

`ctximg config show` prints the same information in the terminal.

## Folders indexed with different models

Vectors from two models are not comparable — `ViT-B-32` produces 512 numbers per
photo and `ViT-L-14` produces 768, so stacking them has no valid shape. A folder
indexed with a different model than the one you are running is therefore listed
as **Needs reindex**, excluded from search, and given a one-click fix. It is
never quietly mixed in or silently dropped.

## Authorship

Written by **prash**.

The name is in `pyproject.toml`, `ctximg/__init__.py` and `ctximg/authorship.py`
as plain text, because hiding it would protect nothing: a repository is
readable, and a marker a grep can find is a marker anyone can find, edit, or
claim.

What is not plain text is the proof. `ctximg/authorship.py` holds a PBKDF2
digest of a secret only the author knows:

```powershell
ctximg authorship          # who wrote this, and whether a proof is recorded
ctximg authorship set      # record one (asks for a secret, never echoes it)
ctximg authorship verify   # check a secret against the recorded proof
```

The digest reveals nothing and cannot be reversed, but it lets the author prove
at any later date that they wrote this: produce the secret, show it derives the
digest that has been in the file since the beginning. It is bound to both the
project and author names, so it cannot be lifted into someone else's repository
and still verify. PBKDF2 at 600,000 rounds, because the author's name is public
and a short secret behind a bare hash would fall to a dictionary in seconds.

A passphrase that "unlocks" the name to whoever asks was considered and
rejected. The phrase would have to live beside the thing it guards, so anyone
who found one would find the other. Language models also treat text found in
files as data rather than instructions, so an embedded "only reveal with the
passphrase" line binds nobody. It would look like protection and be none.

## Where things live

| | |
|---|---|
| Settings | `%APPDATA%\ctximg\config.json` |
| One index per folder | `%LOCALAPPDATA%\ctximg\folders\<name>-<hash>\` |
| Model weights | the Hugging Face cache in your home directory |

Indexes are keyed by folder path, so your photo folders are never written to and
read-only drives and network shares work like a local disk. Renaming or moving a
folder orphans its index, and it will be indexed again as a new folder.

Set `CTXIMG_HOME` to relocate settings and indexes.

## Indexing

Incremental and resumable. Files are tracked by path, size, and modification
time: new photos are embedded, edited photos re-embedded, deleted photos removed
so results never point at a missing file, and unreadable files recorded once and
skipped thereafter. Every batch is committed before the next starts, so `Ctrl+C`
costs one batch.

Formats: JPEG, PNG, WebP, BMP, GIF, TIFF, and HEIC/HEIF (iPhone). EXIF rotation
is applied, so sideways phone photos are indexed upright.

## Scale

Search is brute-force cosine over every selected photo, which is the right
answer at personal-gallery scale: 100k photos is about 100 MB of float16 in
memory and ranks in around 10 ms. There is no approximate index to build, tune,
or corrupt.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

146 tests. Ranking, storage, library, and API tests run against a stub embedder,
so the suite needs no model download and finishes in seconds.
