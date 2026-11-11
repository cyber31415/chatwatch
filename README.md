# chatwatch

A local dashboard for Twitch moderators. It reads one channel's public chat anonymously and shows, live, which
messages are toxic and which chatters are worth a look, with the reasons. It never posts and never acts: every flag is
a prompt for a human to look. Everything runs on your own machine, including the AI models.

chatwatch is not affiliated with or endorsed by Twitch.

> **Content warning.** To recognise slurs and hate codes, `src/chatwatch/assets/coded_lexicon.json`,
> `coded_glossary.json`, the keyword cues in `src/chatwatch/cues.py` and the tests (`tests/`) contain offensive terms in
> plain text.

## What it shows
- **Live chat:** every message with its toxicity level (none, low, medium, high) and why. Filters: All,
  First-timers, Flagged.
- **Watchlist:** the chatters worth a look right now ("likely" / "possible"), each with the messages behind it.
- **Review** (in the 12B + 31B Hybrid setup): a stronger model re-checks the flagged chatters on demand.
- **The AI panel:** what runs on this machine, how accurate each option is, and how busy a chat it keeps up with.

## How it works
1. A fast first pass on every message: word and pattern lists (slurs count in any spelling) and a small toxicity model
   (an ONNX encoder plus a head trained for this tool).
2. A local language model (Gemma 4, through Ollama) re-reads messages and reads each flagged chatter as a whole. Its
   reads decide the Watchlist.
3. The queue never drops work: when chat is busier than the model, it catches up after (the page says how far behind).

## Requirements
- **Linux with podman 4.0 or later**, and the usual command-line tools (bash, curl, coreutils, util-linux).
  Everything else runs in containers; nothing is installed on the host.
- **A GPU is recommended:** Nvidia (with podman's CDI set up: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`)
  or AMD (ROCm). It also runs on the CPU, slowly.
- **Disk:** about 4-6 GB for the containers (Ollama's image is most of it), plus the models you install (the 31B
  about 20 GB, the 12B about 8 GB).
- **RAM:** 16 GB is the lowest option; the 31B on a 12 GB card needs about 14 GB free.

Tested on an Nvidia GPU + AMD CPU system and an AMD GPU + AMD CPU system. The dashboard reads your GPU memory and
system RAM itself; other hardware uses estimates until you press **Check Speed**. macOS and Windows are untested.

## Quick start
```
git clone https://github.com/cyber31415/chatwatch.git && cd chatwatch
./start.sh                                     # builds the image, starts the AI container; open http://127.0.0.1:8765
bin/ollama.sh install gemma4mtp:31b-n3-text    # in a second terminal: the model the page recommends (example)
```
On the first start no AI is chosen yet. The AI panel lists the setups that fit your machine; one that is not
installed shows its install command. Then:
1. Type the channel and press **Measure the channel's chat rate** (or type it).
2. Press **Check Speed** on the models you might use (1-3 minutes each). Speeds in italics are estimates until then.
3. Pick one and press **Use this**, then **Start**. Up to about 40 msgs/min the Gemma 4 31B is the most accurate;
   busier chat suits the 12B + 31B Hybrid.

The first start downloads the two small encoder models (about 150 MB, pinned and checksum-verified). An MTP model's
drafter is downloaded by `bin/ollama.sh install` (pinned, sha256-checked: `models/drafts/drafters.json`).

```
./start.sh --no-llm          # without the AI: word lists and the small model only
./start.sh --log             # keep a session log of this run (see Privacy)
CW_PORT=8766 ./start.sh      # another port
```
Ctrl+C stops the dashboard. The AI container keeps its model in memory for the next start; `bin/ollama.sh stop`
frees the GPU and RAM. One dashboard at a time.

## Settings (environment variables)
| variable | default | what it does |
|---|---|---|
| `CW_PORT` | 8765 | the dashboard's port (on 127.0.0.1) |
| `CW_GPU` | detected | `nvidia`, `amd` or `cpu` for the AI container |
| `CW_OLLAMA_PORT` | 11435 | the AI container's port (on 127.0.0.1) |
| `CW_OLLAMA_STORE` | `models/ollama` | where the AI models are kept |
| `CW_KEEP_CORES` | 2 | CPU cores the AI leaves free for the rest of the machine (0: none) |
| `CW_VRAM_RESERVE_GB` | 1.5 | on a GPU that drives your screen: GPU memory a model that fills the card leaves free |
| `CW_SESSION_KEEP_DAYS` | 14 | session logs older than this are deleted at start (0: keep) |

They can also go in `~/.config/chatwatch/gpu.env` (one `NAME=value` per line); a variable set in the environment
wins.

## Privacy and responsible use
- It logs in anonymously (read-only), sees only what any viewer sees, and never posts.
- Besides Twitch's chat connection and the one-time model downloads, it fetches the watched channel's emote sets from
  7TV, BetterTTV and FrankerFaceZ (sending the channel's Twitch id), so it can tell emotes from words.
- **Session logs are off by default.** With `--log`, a run's raw chat lines (with usernames) and the tool's reads are
  kept in `data/sessions/`, and deleted after `CW_SESSION_KEEP_DAYS`.
- The running state (`data/state/`) holds per-chatter sums for the channels you watched. To forget a chatter:
  `bin/cw python -m chatwatch.forget USER_ID --go` (without `--go` it only reports).
- Use it for your own channel or one you moderate, for moderators only. Never show its reads to chatters or on stream:
  they are probabilistic first impressions of what someone is doing in chat, not judgements of who they are.
- See [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md) for the defaults, the known bias risks and why.

## Accuracy
Measured on held-out real Twitch chat against the consensus of two independent AI judges (Claude Opus); details and
caveats in [docs/QUALITY.md](docs/QUALITY.md).

| setup | accuracy (gain over No AI) | msgs/min it keeps up with (12 GB GPU) |
|---|---|---|
| Gemma 4 31B | +66% | ~40 |
| 12B + 31B Hybrid (Review covering all flagged chatters) | +62% | ≥150 live |
| Gemma 4 12B | +50% | ≥150 |
| No AI (word lists and the small model) | +0% | any |

With the 31B on quiet chat: about 3 in 4 toxic messages caught (recall 0.74) at precision 0.64.

## Files
- `start.sh` the launcher; `bin/` the container wrapper (`cw`), the AI container (`ollama.sh`), its log copier.
- `src/chatwatch/` the dashboard (Python); `web/index.html` its page.
- `data/models/` the small models trained for this tool (the toxicity head, the cold-start and chatter-type models).
- `models/drafts/` the MTP Modelfiles and the drafter download list.
- `tests/` the tests (`pytest`), and `tests/web/` the page's browser checks.

## Licence
Apache-2.0 ([LICENSE](LICENSE)). The models it downloads keep their own licences ([NOTICE](NOTICE)).
