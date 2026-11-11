# Responsible use

A live per-chatter characterizer is a profiling tool by construction. These are the defaults chatwatch ships with, and
why.

## Defaults in the code
- **Read-only and anonymous.** It logs in as `justinfanNNNNN`, so it cannot post, is not tied to an account, and sees
  only what any viewer sees.
- **Never acts.** There are no automatic timeouts or bans. Flags are prompts to look, and they show their reasons.
- **One channel at a time, per-channel profiles.** There is no cross-channel tracking.
- **The same standard for everyone.** Moderators, VIPs and the broadcaster are scored like everyone else; their cards
  show their role.
- **Registers, not psychology.** Labels describe what someone is doing in chat ("reads as a provocateur so far"), never
  who they are.
- **Local only.** The dashboard listens on 127.0.0.1; the AI runs in a local container. Besides Twitch's chat
  connection and the one-time model downloads, the only outside requests are for the watched channel's emote sets
  (7TV, BetterTTV, FrankerFaceZ: the channel's Twitch id is sent); no chat and nothing about chatters leaves the
  machine.
- **No session logs unless you ask.** `./start.sh --log` keeps a run's raw chat (with usernames) and reads in
  `data/sessions/`; logs older than `CW_SESSION_KEEP_DAYS` (14) are deleted at start.

## Before using it on a real channel
- **Use it for your own channel, or one you moderate, for moderators only.** Never show its labels to chatters or on
  stream.
- **What it keeps.** `data/state/engine.pkl` holds, for the channels you watched, running per-chatter sums and each
  chatter's recent and flagged messages, with usernames (and `engine.pkl.prev`, the state before the last save).
  With `--log`, each session folder also keeps that run's raw chat lines (`raw/`), the tool's outputs
  (`outputs.jsonl`) and a copy of the state at its start; delete them when no longer needed.
- **Your marks and corrections.** `data/sessions/owner_roles.json` holds the user id, login and display name of each
  chatter you marked as a bot; `data/sessions/owner_overrides.jsonl` holds the ids, levels and times of messages you
  reclassified (no text). They steer your dashboard only and are never used to train anything.
- **Opt-out.** If a chatter asks to be forgotten, stop the dashboard and run
  `bin/cw python -m chatwatch.forget USER_ID --go` (without `--go` it only reports). It removes them from the saved
  state and every session's state copy, from your corrections and your bot marks, and lists the session logs that
  mention them, to delete.
- **The state is a pickle.** Anyone who can write this folder can make the dashboard run code when it loads
  `engine.pkl`. Keep the folder private to you.
- **Twitch's terms.** Twitch's Developer Agreement limits keeping chat logs "to collect information about Twitch's end
  users". Use beyond your own channel's moderation may need Twitch's agreement.

## Known bias risks
- Toxicity classifiers over-flag African-American English, reclaimed slurs, non-native English and gaming trash talk.
  In chatwatch the small model's scores count only in narrow cases (a very sure score together with a hostile cue or an
  aimed insult, an identity attack, a threat aimed at a person, very sure sexual content), and the head can lift a
  message to low or medium on its own, never to high. Only sure content (a slur, or a threat, sexual or harassing phrase
  from the keyword lists) or the local AI's re-read puts someone at "likely" on the Watchlist. The AI's instructions
  tell it not to flag non-native English, slang or dialect; a slur counts in any spelling.
- A moderator's timeouts and bans are biased labels: they reflect one channel's rules and the moderators' attention.
  chatwatch does not use them as evidence.
- A false "watch" is cheap only if a human reviews it. Wired to an automatic action, the false-alarm cost changes
  completely: at a 5% false-positive rate and a 2-6% base rate, precision is about 20%.

## Reclassifying a message
You can correct a message's level by hand on the dashboard (the ⋯ beside it). The correction steers your Watchlist
only; it is recorded with ids, levels and times, never the message text; and `chatwatch.forget` removes a chatter's
corrections too.
