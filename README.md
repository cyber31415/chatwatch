# chatwatch

**A local dashboard for Twitch moderators.** It reads one channel's public chat and shows, live, which messages are
toxic and which chatters are worth a look, with the reasons.

![The chatwatch dashboard while watching a channel](docs/images/overview.png)

- **Runs on your machine.** The AI models run locally; no chat or chatter data leaves your computer.
- **Flags with reasons.** Every flagged message says what it is (an insult, a threat, mockery, spam...) and who it is
  aimed at; every chatter on the Watchlist says why they are there.
- **Never acts.** It reads chat anonymously and cannot post, time out or ban. Every flag is a prompt for a human.

chatwatch is not affiliated with or endorsed by Twitch.

> **Content warning.** To recognise slurs and hate codes, `src/chatwatch/assets/coded_lexicon.json`,
> `coded_glossary.json`, the keyword cues in `src/chatwatch/cues.py` and the tests (`tests/`) contain offensive terms in
> plain text.

**Contents:** [Quick start](#quick-start) · [A tour of the dashboard](#a-tour-of-the-dashboard) ·
[How accurate it is](#how-accurate-it-is) · [How it works](#how-it-works) · [Requirements](#requirements) ·
[Everyday use](#everyday-use) · [Settings](#settings) · [Privacy and responsible use](#privacy-and-responsible-use)

## Quick start
```
git clone https://github.com/cyber31415/chatwatch.git && cd chatwatch
./start.sh                                     # builds its image, starts the AI container
```
Open **http://127.0.0.1:8765**. On the first start no AI is chosen yet: the AI panel lists the setups that fit your
machine, and one that is not installed shows the command that installs it, for example:
```
bin/ollama.sh install gemma4mtp:31b-n3-text    # in a second terminal
```
Then pick a setup, press **Use this**, type a channel and press **Start**. The first start also downloads two small
encoder models (about 150 MB, pinned and checksum-verified).

## A tour of the dashboard
The screenshots use invented chat (every chatter is `demo_...`) run through the real engine. The text of each flagged
message is covered with a black box.

### 1. Choose the AI
![Choosing the AI](docs/images/ai-panel.png)

While stopped, the AI panel shows what this machine has and every setup that fits it:
- **Accuracy** is how well a setup marks toxic messages and picks out bad-faith chatters, as a gain over No AI.
- **msgs/min** is the busiest chat it keeps up with. Speeds in *italics* are estimates; press **Check Speed** (1-3
  minutes) to measure a model on your machine. The ones above were measured.
- **Measure the channel's chat rate** (or type it) and the panel marks the setup that suits that chat. Up to about 40
  msgs/min the Gemma 4 31B is the most accurate; busier chat suits the **12B + 31B Hybrid**.

### 2. The AI bar
![The AI bar](docs/images/ai-bar.png)

Once running, the panel folds into one line: the setup in use and its speed. Here the Hybrid's fast 12B reads every
message live and the 31B reviews on demand; both models are in memory, so a review's model swap takes about 8 seconds.

### 3. Live chat
![Live chat](docs/images/live-chat.png)

Every message, as it arrives:
- **Level and reason:** low, medium or high, what kind (insult, threat, mockery, hostile...) and who it is aimed at
  (the streamer, a chatter, everyone).
- **Watchlist tag:** "likely" or "possible" beside anyone on the Watchlist.
- **Warning chips:** a first message (`1st`), a `template` posted by several new accounts, a `ban-lookalike` of an
  account that was just banned.
- **Moderator actions** (bans, timeouts) appear as context.
- The **⋯** beside a message lets you reclassify it by hand; your correction steers your Watchlist only.

### 4. The Flagged filter
![The Flagged filter](docs/images/flagged.png)

**Flagged** keeps only the messages with a level or a warning chip, and everything said by people on the Watchlist.
**First-timers** keeps the first messages of new chatters.

### 5. The Watchlist
![The Watchlist](docs/images/watchlist.png)

The chatters worth a look right now:
- **likely** (red): sure content, such as a threat or a slur.
- **possible** (amber): a pattern, such as several toxic messages in a few minutes or a bad-faith way of talking.

Each card shows its reasons and the message behind them; open the reasons for every message that counted.

### 6. Review (12B + 31B Hybrid)
![The Review box](docs/images/review-box.png)

In the Hybrid, **Review** has the stronger 31B re-check the flagged chatters. The box says how long it takes and who
it covers (the Watchlist plus everyone else with a toxic message). While it runs, the live reading pauses; chat waits
in the queue and is caught up after. Tick **Review Watchlist only** for fewer chatters and a shorter pause. The **(i)**
buttons explain both.

### 7. A chatter's card
![A chatter's card](docs/images/chatter-card.png)

Click any name for their card: why they are listed, the AI's read of them as a person, their toxic messages, and tools
(a warning draft for you to copy, ask the AI again, mark as a bot). Nothing is ever posted for you.

## How accurate it is
Measured on held-out real Twitch chat against the consensus of two independent AI judges (Claude Opus). Details and
caveats: [docs/QUALITY.md](docs/QUALITY.md).

| setup | accuracy (gain over No AI) | msgs/min it keeps up with (12 GB GPU) |
|---|---|---|
| Gemma 4 31B | +66% | ~40 |
| 12B + 31B Hybrid (Review covering all flagged chatters) | +62% | ≥150 live |
| Gemma 4 12B | +50% | ≥150 |
| No AI (word lists and the small model) | +0% | any |

With the 31B on quiet chat it catches about 3 in 4 toxic messages (recall 0.74) at precision 0.64.

## How it works
1. **A fast first pass** on every message: word and pattern lists (slurs count in any spelling) and a small toxicity
   model (an ONNX encoder plus a head trained for this tool).
2. **A local language model** (Gemma 4, through Ollama) re-reads messages and reads each flagged chatter as a whole.
   Its reads decide the Watchlist.
3. **Nothing is dropped:** when chat is busier than the model, the queue catches up after (the page says how far
   behind it is).

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

## Everyday use
```
./start.sh                   # start (then open http://127.0.0.1:8765)
./start.sh --no-llm          # without the AI: word lists and the small model only
./start.sh --log             # keep a session log of this run (see Privacy)
CW_PORT=8766 ./start.sh      # another port
bin/ollama.sh stop           # free the GPU and RAM the AI container holds
```
Ctrl+C stops the dashboard; the AI container keeps its model in memory for the next start. One dashboard at a time,
one channel per session: to watch another channel, Stop, then Start.

While it runs it leaves room for the rest of your machine: two CPU cores, and on a GPU that drives your screen 1.5 GB
of its memory (see Settings).

## Settings
Environment variables, or lines in `~/.config/chatwatch/gpu.env` (`NAME=value`; the environment wins):

| variable | default | what it does |
|---|---|---|
| `CW_PORT` | 8765 | the dashboard's port (on 127.0.0.1) |
| `CW_GPU` | detected | `nvidia`, `amd` or `cpu` for the AI container |
| `CW_OLLAMA_PORT` | 11435 | the AI container's port (on 127.0.0.1) |
| `CW_OLLAMA_STORE` | `models/ollama` | where the AI models are kept |
| `CW_KEEP_CORES` | 2 | CPU cores the AI leaves free for the rest of the machine (0: none) |
| `CW_VRAM_RESERVE_GB` | 1.5 | on a GPU that drives your screen: GPU memory a model that fills the card leaves free |
| `CW_SESSION_KEEP_DAYS` | 14 | session logs older than this are deleted at start (0: keep) |

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
- More in [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md): the defaults, the known bias risks and why.

## Files
- `start.sh` the launcher; `bin/` the container wrapper (`cw`), the AI container (`ollama.sh`), its log copier.
- `src/chatwatch/` the dashboard (Python); `web/index.html` its page.
- `data/models/` the small models trained for this tool (the toxicity head, the cold-start and chatter-type models).
- `models/drafts/` the MTP Modelfiles and the drafter download list.
- `tests/` the tests (`pytest`), and `tests/web/` the page's browser checks.
- `docs/` accuracy details, responsible use, and the screenshots.

## Licence
Apache-2.0 ([LICENSE](LICENSE)). The models it downloads keep their own licences ([NOTICE](NOTICE)).
