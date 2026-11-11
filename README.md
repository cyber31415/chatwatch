# chatwatch

**A local dashboard for Twitch moderators.** It reads one channel's public chat and shows, live, which messages are
toxic and which chatters are worth a look, with the reasons. It reads English chat.

![The chatwatch dashboard while watching a channel](docs/images/overview.png)

**Built on AI models that run on your own computer:**
- **Gemma 4** (Google), a language model, re-reads the chat and judges each chatter worth a look. Pick the size that
  suits your machine: **31B**, **12B** or **E4B**, or the **12B + 31B Hybrid**.
- **Two small models** check every message the instant it arrives: one scores toxicity (unbiased-toxic-roberta, tuned
  for this tool), the other recognises styles of talking (all-MiniLM-L6-v2).
- **No AI** also works: word lists and the small models only, on any machine.

**Why use it:**
- **Private.** Everything runs on your computer; no chat or chatter data leaves it.
- **Flags with reasons.** Every flagged message says what it is (an insult, a threat, mockery, hate...) and who it is
  aimed at; every chatter on the Watchlist says why they are there.
- **Never acts.** It reads chat anonymously and cannot post, time out or ban. Every flag is a prompt for a human.

chatwatch is not affiliated with or endorsed by Twitch.

> **Content warning.** To recognise slurs and hate codes, `src/chatwatch/assets/coded_lexicon.json`,
> `coded_glossary.json`, the keyword cues in `src/chatwatch/cues.py` and the tests (`tests/`) contain offensive terms in
> plain text.

**Contents:** [Install](#install) · [A tour of the dashboard](#a-tour-of-the-dashboard) ·
[Everyday use](#everyday-use) · [Settings](#settings) · [How it works](#how-it-works) ·
[How accurate it is](#how-accurate-it-is) · [Privacy and responsible use](#privacy-and-responsible-use) ·
[Files](#files)

## Install
Downloads for each system are on the [Releases](https://github.com/cyber31415/chatwatch/releases) page. macOS is not
supported.

### Windows 10 or 11
Download `chatwatch-<version>-windows.zip`, extract it, and double-click **Start chatwatch.cmd**. If Windows warns about
a file from the internet, choose **More info**, then **Run anyway**.

chatwatch runs inside Ubuntu in WSL2 (Windows' built-in Linux), and the page opens in your normal Windows browser.
WSL itself must be installed once: if it is not, the launcher says so; open PowerShell as administrator, run
`wsl --install --no-distribution` and restart the PC. Keep WSL up to date with `wsl --update`.

Steps 1-3 happen only the first time:
1. **WSL's memory.** WSL's default is half your RAM, too little for the AI models, so the launcher raises WSL's limit in
   `C:\Users\<you>\.wslconfig` to your RAM minus 3 GB. It asks first and keeps a copy of the old file. Windows keeps
   whatever WSL does not use.
2. **Ubuntu 24.04**, where you choose a username and password.
3. **podman and chatwatch** inside Ubuntu, in `~/chatwatch` (it asks for that password).
4. **It starts chatwatch** and opens the page. **Ctrl+C** in its window stops chatwatch, then shuts WSL down (unless
   something else runs in it), so all the memory goes back to Windows. If it then asks "Terminate batch job (Y/N)?",
   either answer is fine.

Tested on a laptop with no GPU. An Nvidia GPU under WSL is untested: without the GPU setup described under Linux (done
inside Ubuntu), chatwatch uses the CPU. An AMD GPU under WSL is not supported.

<details><summary>Setting it up by hand instead</summary>

1. Put `[wsl2]` and `memory=12GB` (your RAM minus 3 GB; 12GB on a 16 GB PC) on two lines in
   `C:\Users\<you>\.wslconfig`, then run `wsl --shutdown` in PowerShell.
2. `wsl --install -d Ubuntu-24.04`
3. In Ubuntu: `sudo apt update && sudo apt install -y podman git curl iptables uidmap dbus-user-session`
4. `cd ~ && git clone https://github.com/cyber31415/chatwatch.git && cd chatwatch && ./start.sh`. Keep it on Ubuntu's
   own disk, not in a Windows folder under `/mnt/c`, where models load slowly.
5. Open **http://127.0.0.1:8765** in your Windows browser. Ctrl+C stops chatwatch and its AI. `wsl --shutdown` in
   PowerShell gives the rest of WSL's memory back to Windows.

</details>

### Linux
1. **Install podman 4.0 or later** and the tools chatwatch uses (`start.sh` does not install them):
   - Ubuntu 24.04 or later, Debian 12 or later: `sudo apt install podman iptables git curl`
   - Fedora: `sudo dnf install podman git curl`
   - Arch: `sudo pacman -S podman git curl`
2. **GPU (optional, much faster):**
   - **Nvidia:** install the
     [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
     then run `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` once. Without it, chatwatch uses the CPU
     and says so when it starts.
   - **AMD:** nothing to install (the AI container brings ROCm); your user must be in the `video` and `render` groups
     (`sudo usermod -aG video,render $USER`, then log out and back in).
3. **Get chatwatch and start it** (or extract a release's `chatwatch-<version>-linux.tar.gz` instead of cloning):
   ```
   git clone https://github.com/cyber31415/chatwatch.git && cd chatwatch
   ./start.sh
   ```
   then open **http://127.0.0.1:8765**.

Everything else runs in containers, and the models chatwatch downloads stay in its `models/` folder.

Tested on an Nvidia GPU + AMD CPU system (Ubuntu 24.04 and 26.04) and an AMD GPU + AMD CPU system (CachyOS). On Ubuntu
26.04 podman may print `rootless netns: kill network process: permission denied` when chatwatch stops: it is
harmless.

### First start
The first start uses **No AI**, which works straight away: type a channel and press **Start**. The AI models are big
downloads (6-20 GB each), so none is downloaded until you choose one: the AI panel lists the setups that fit your
machine; press **Download** on one (the page shows its progress; for the Hybrid, download the 12B and the 31B), then
**Use this**. chatwatch remembers your choice.

### Removing it
- **Linux:** run these, then delete the folder (and `~/.config/chatwatch/` if you made it):
  ```
  podman rm -f cw-ollama
  podman rmi docker.io/ollama/ollama:0.40.11      # AMD: docker.io/ollama/ollama:0.40.11-rocm
  podman images --filter reference=localhost/chatwatch -q | xargs -r podman rmi
  podman rmi docker.io/library/python:3.12-slim    # unless something else uses it
  podman network rm cw-net
  ```
- **Windows:** if Ubuntu 24.04 was installed for chatwatch, `wsl --unregister Ubuntu-24.04` in PowerShell deletes it
  and everything in it. If you had it before chatwatch, keep it: in Ubuntu, run the Linux steps above and delete
  `~/chatwatch`. Then delete the extracted folder and `C:\Users\<you>\.chatwatch`. In `C:\Users\<you>\`, restore `.wslconfig` from its
  `.wslconfig.bak-...` copy if there is one; otherwise remove the `memory=` line the launcher added.

## A tour of the dashboard
The screenshots use invented chat (every chatter is `demo_...`) run through the real engine. The text of each flagged
message is covered with a black box.

### 1. Choose the AI
![Choosing the AI](docs/images/ai-panel.png)

While stopped, the AI panel shows what this machine has and the setups suited to it:
- **Accuracy** is how well a setup marks toxic messages and picks out bad-faith chatters, as a gain over No AI (+66%
  scores 1.66 times No AI's score).
- **msgs/min** is the busiest chat a setup keeps up with: flagged messages and chatter reads stay under 10 minutes
  behind. Speeds in *italics* are estimates; press **Check Speed** (1-3 minutes) to measure a model on your machine.
  All the speeds in this screenshot were measured.
- **Measure the channel's chat rate** (or type it) and the panel marks the setup that suits that chat. On a 12 GB GPU,
  the 31B is the most accurate up to about 40 msgs/min, and the Hybrid suits busier chat.

**What each setup needs.** A GPU is fastest but not required. A model that does not fit in the GPU's memory is split
between the GPU and system RAM (slower); with no GPU it runs from system RAM on the CPU (slowest). The panel offers only
what fits your machine.

| setup | download | how it runs | offered on |
|---|---|---|---|
| Gemma 4 31B | 19.8 GB | split between GPU and RAM (on a 12 GB GPU, about half is in RAM) | a GPU of 8 GB or more, with 32 GB RAM or more |
| 12B + 31B Hybrid | 27.8 GB | the 12B live and the 31B for reviews, one in memory at a time | as the 31B |
| Gemma 4 12B | 8.0 GB | all on a 12 GB GPU, split on an 8 GB GPU, or on the CPU | a GPU of 8 GB or more; or no GPU, 32 GB RAM or more |
| Gemma 4 E4B | 6.3 GB | on the CPU: the setup for machines without a GPU | no GPU, 16 GB RAM or more |
| No AI | none | on the CPU | any machine |

A GPU with less than 8 GB counts as no GPU.

### 2. The AI bar
![The AI bar](docs/images/ai-bar.png)

Once running, the panel folds into one line: the setup in use and its speed. Here the Hybrid's fast 12B does the live
re-reading and the 31B reviews when you press Review. The bar shows how long this machine takes to swap the two models
(about 8 seconds here, with plenty of free RAM; without it, a swap loads them from disk and takes longer).

### 3. Live chat
![Live chat](docs/images/live-chat.png)

Every message, as it arrives:
- **Level and reason:** low, medium or high, what kind (insult, threat, mockery, hostile...) and who it is aimed at
  (the streamer, a chatter, a group, someone outside chat, or no one in particular).
- **Watchlist tag:** "likely" or "possible" beside anyone on the Watchlist.
- **Chips:** `1st` (their first message ever in this channel), `new acct` (an account under a week old), `coded vocab`
  and `possible distress`. Warning chips: `template` (posted by several new accounts), `ban-lookalike` (a name like a
  just-banned account's), `evasion` (disguised spelling) and `twitch flag` (Twitch's own filter).
- **Moderator actions** (bans, timeouts) appear as context.
- The **⋯** beside a message lets you reclassify it by hand; your correction steers your Watchlist only.

### 4. The Flagged filter
![The Flagged filter](docs/images/flagged.png)

**Flagged** keeps toxic messages, messages with a warning chip, and everything said by people on the Watchlist.
**First-timers** keeps each chatter's first message ever in this channel.

### 5. The Watchlist
![The Watchlist](docs/images/watchlist.png)

The chatters worth a look right now:
- **likely** (red): clear-cut content, such as a slur, or a threat (amber instead if the AI reads the chatter as not
  acting in bad faith).
- **possible** (amber): the AI reads them as acting in bad faith. With No AI: 3 or more toxic messages in 10 minutes,
  or a bad-faith way of talking.

Each card shows its reasons and the message behind them; open the reasons for every message that counted.

### 6. Review (12B + 31B Hybrid)
![The Review box](docs/images/review-box.png)

In the Hybrid, **Review** has the stronger 31B re-check the flagged chatters. The box says how long it takes and who
it covers: everyone on the Watchlist, plus anyone else with a toxic message, who has posted since the last review.
While it runs, the AI's live re-reading pauses (new
messages still get the instant check) and catches up after. Tick **Review Watchlist only** for fewer chatters and a
shorter pause. The **(i)** buttons explain both.

### 7. A chatter's card
![A chatter's card](docs/images/chatter-card.png)

Click any name for their card: why they are listed, the AI's read of them as a person, their toxic messages, and tools
(a warning draft for you to copy, ask the AI again, mark as a bot). Nothing is ever posted for you.

## Everyday use
```
./start.sh                   # start (then open http://127.0.0.1:8765)
./start.sh --no-llm          # without the language model: word lists and the small models only
./start.sh --log             # keep a session log of this run (see Privacy)
CW_PORT=8766 ./start.sh      # another port
bin/ollama.sh stop           # free the memory the AI holds
```
Ctrl+C stops it. On Linux the AI keeps its model in memory so the next start is quick; `bin/ollama.sh stop` frees that
memory. On Windows, Ctrl+C also stops the AI and shuts WSL down. One dashboard at a time, one channel per session: to
watch another channel, Stop, then Start. While it runs, chatwatch leaves room for the rest of your machine (see
`CW_KEEP_CORES` and `CW_VRAM_RESERVE_GB` below).

On Windows the launcher takes no options. For `--log` or the commands above, open **Ubuntu 24.04** from the Start menu,
run `cd ~/chatwatch`, and use them there. Settings in that Ubuntu's `~/.config/chatwatch/gpu.env` apply to the launcher
too, except `CW_PORT` (the launcher always opens 8765).

## Settings
Environment variables, or lines in `~/.config/chatwatch/gpu.env` (`NAME=value`; the environment wins):

| variable | default | what it does |
|---|---|---|
| `CW_PORT` | 8765 | the dashboard's port (on 127.0.0.1) |
| `CW_GPU` | detected | `nvidia`, `amd` or `cpu` for the AI |
| `CW_OLLAMA_PORT` | 11435 | the AI container's port (on 127.0.0.1) |
| `CW_OLLAMA_STORE` | `models/ollama` | where the AI models are kept |
| `CW_KEEP_CORES` | 2 | CPU cores the AI leaves free for the rest of the machine (0: none) |
| `CW_VRAM_RESERVE_GB` | 1.5 | on a GPU that drives your screen: memory a model that fills the GPU leaves free |
| `CW_SESSION_KEEP_DAYS` | 14 | session logs older than this are deleted at start (0: keep) |
| `CW_KEEP_AI` | (unset) | Windows, started by hand in Ubuntu: 1 keeps the AI loaded after Ctrl+C |

## How it works
1. **A fast first pass** checks every message as it arrives: word and pattern lists (they see through common
   disguises such as l33t, look-alike letters and spaced-out letters) and the small models. Its flags show at once.
2. **The language model** (Gemma 4, run by Ollama) then re-reads most messages (not bot commands or very short
   harmless ones), flagged ones first, and confirms or clears each flag. A slur caught by the word lists stays flagged
   whatever the model says. It also reads each chatter worth a look as a whole; that read decides who is listed as
   "possible".
3. **When chat is busier than the model,** the work waits in a queue and is caught up. Flagged messages and chatter
   reads always get their turn; an ordinary message still waiting after 15 minutes is not re-read and keeps its
   first-pass result. The page shows how far behind it is.

## How accurate it is
Measured on held-out real English Twitch chat against the consensus of two independent AI judges (Claude Opus; not human
moderators). Details and caveats: [docs/QUALITY.md](docs/QUALITY.md).

| setup | accuracy (gain over No AI) | msgs/min it keeps up with* |
|---|---|---|
| Gemma 4 31B | +66% | ~40 |
| 12B + 31B Hybrid† | +62% | ≥150 live |
| Gemma 4 12B | +50% | ≥150 |
| Gemma 4 E4B | +37% | ≥150 (no GPU: ~20, or ~10 on a 16 GB laptop) |
| No AI (word lists and the small models) | +0% | any |

\* Tested on a machine with a 12 GB Nvidia GPU and 96 GB of system memory. The E4B is offered only on machines without
a GPU.

† With Review pressed about every 10 minutes; between reviews it reads at the 12B's level. Above about 75 msgs/min a
press cannot cover every chatter (about +59% at 100, +55% at 150).

With the 31B, about 3 in 4 toxic messages are caught, and about 2 in 3 of its flags are right (recall 0.74, precision
0.64, on quiet chat).

## Privacy and responsible use
- It logs in anonymously (read-only), sees only what any viewer sees, and never posts.
- Besides Twitch's chat connection and the one-time model downloads, it fetches the watched channel's emote sets from
  7TV, BetterTTV and FrankerFaceZ (sending the channel's Twitch id), so it can tell emotes from words.
- **Session logs are off by default.** With `--log`, a run's raw chat lines (with usernames) and the tool's reads are
  kept in `data/sessions/`, and deleted after `CW_SESSION_KEEP_DAYS`.
- The running state (`data/state/`) holds, for the channels you watched, per-chatter sums and each chatter's recent
  and flagged messages (with usernames).
- To forget a chatter, stop chatwatch and run `bin/cw python -m chatwatch.forget USER_ID --go` (on Windows, in Ubuntu,
  in `~/chatwatch`). USER_ID is their numeric Twitch user id, not their name; chatwatch does not show it, so look the
  name up with a Twitch username-to-ID tool. Without `--go` it only reports. It also lists any session logs that
  mention them, for you to delete.
- Use it for your own channel or one you moderate, for moderators only. Never show its reads to chatters or on stream:
  they are probabilistic first impressions of what someone is doing in chat, not judgements of who they are.
- More in [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md): the defaults, the known bias risks and why.

## Files
- `start.sh` the launcher; `bin/` the container wrapper (`cw`), the AI container (`ollama.sh`), its log copier, and the
  helper that runs the page's model downloads (`install_helper.sh`).
- `windows/` the Windows launcher (`Start chatwatch.cmd`, `chatwatch.ps1`, and `setup_wsl.sh`, its part inside WSL): a
  release's Windows zip holds these three files.
- `src/chatwatch/` the dashboard (Python); `web/index.html` its page.
- `data/models/` the small models trained for this tool (the toxicity layer, the cold-start and chatter-type models).
- `models/drafts/` the speed-up (MTP) model files and their download list.
- `tests/` the tests (`pytest`), and `tests/web/` the page's browser checks.
- `docs/` accuracy details, responsible use, and the screenshots.

## Licence
Apache-2.0 ([LICENSE](LICENSE)). The models it downloads keep their own licences ([NOTICE](NOTICE)).
