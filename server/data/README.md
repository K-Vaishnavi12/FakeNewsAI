# Datasets

**These files are intentionally excluded from Git.** They total roughly 408 MB,
and the largest single file (`WELFake_Dataset.csv`, 234 MB) exceeds GitHub's
100 MB hard limit — a commit containing it is rejected on push. Everything in
this directory except this README is ignored via `.gitignore`.

Do not commit datasets, `.env` files, API keys, tokens, or any private
credentials to this repository.

## Source

The datasets are published as assets on the **`datasets-v1` release**:

<https://github.com/K-Vaishnavi12/FakeNewsAI/releases/tag/datasets-v1>

Release assets allow up to 2 GB per file, so the full set fits, and they do not
bloat the Git history the way committed files would.

A mirror of the original folder is also on Google Drive:
<https://drive.google.com/drive/folders/1vuV3ALf6JmHLR8HVGENPPK9LG9bcWSnC>

> The repository is **private**, so release downloads are authenticated. You
> need the [GitHub CLI](https://cli.github.com) and `gh auth login`. If you do
> not have repo access, use the Drive mirror instead.

## Where the files go

Everything belongs directly in this directory — `server/data/` — with no
subfolders. This path is defined once, as `DATA_DIR` in
[`server/paths.py`](../paths.py); the training pipeline, the Flask API and the
sync tool all read it from there.

## Getting the files

From the repository root, with the virtualenv active:

```bash
pip install -r server/requirements.txt

# Default — pull the GitHub release assets (needs `gh auth login`)
python -m server.datasets sync

# Fallback — the public Google Drive mirror
python -m server.datasets sync --from drive

# You already downloaded the folder by hand
python -m server.datasets sync --source "/path/to/downloaded/Data"

# Verify
python -m server.datasets check
```

`check` exits non-zero if any required file is missing, so it is safe to use in
a setup script. It also flags leftover `.part` files, which are **incomplete
downloads, not data** — delete them and re-run `sync`.

Equivalent raw command, if you prefer not to use the Python entry point:

```bash
gh release download datasets-v1 --repo K-Vaishnavi12/FakeNewsAI \
   --dir server/data --pattern "*"
```

## Required for training

`python -m server.ml.train_model` needs these five files. The loader skips any
that are absent, so a partial set silently produces a weaker model — run
`check` first.

| File | Size | Columns used | Label convention |
| --- | ---: | --- | --- |
| `WELFake_Dataset.csv` | 234 MB | `title`, `text`, `label` | `1` = fake, `0` = real (remapped on load) |
| `Fake.csv` (ISOT) | 60 MB | `title`, `text`, `subject`, `date` | entire file is fake |
| `True.csv` (ISOT) | 51 MB | `title`, `text`, `subject`, `date` | entire file is real |
| `BuzzFeed_fake_news_content.csv` | 0.6 MB | `id`, `title`, `text` | entire file is fake |
| `BuzzFeed_real_news_content.csv` | 0.6 MB | `id`, `title`, `text` | entire file is real |

Internally the project uses `0` = fake, `1` = real; WELFake's inverted labels
are remapped in `server/ml/train_model.py`.

## Also published (not used for training)

`FACTors.csv`, `author_stats.csv`, `org_stats.csv`,
`PolitiFact_fake_news_content.csv`, `PolitiFactNews.txt`,
`PolitiFactNewsUser.txt`, `PolitiFactUser.txt`, `PolitiFactUserUser.txt`,
`BuzzFeedNews.txt`, `BuzzFeedNewsUser.txt`, `BuzzFeedUser.txt`,
`BuzzFeedUserUser.txt`.

These are downloaded by `sync` but no code path reads them.

> **Warning — `PolitiFact_fake_news_content.csv` is mislabelled.** Despite its
> name, its rows carry ids of the form `Real_1-Webpage`: it contains the *real*
> article set. It is deliberately absent from `DATASET_REGISTRY` in
> `server/paths.py`. Training on it would label ~120 real articles as fake.
> The Drive mirror ships no correct PolitiFact fake set either.

## Training

```bash
python -m server.ml.train_model
```

Optional flags: `--data-dir` to point elsewhere, `--download-data` to sync
first, `--output` to choose the artefact name. The model is written to
`server/ml/models/` and is also gitignored — rebuild it rather than commit it.
