# How accurate it is

All figures come from the development project's held-out test sets: real public Twitch chat collected in October
2026 (English-speaking channels), split by set before any model was run, and never used for tuning.

## What "correct" means
Each message and chatter in the test sets was labelled by two independent AI judges (Claude Opus, the same written
rubric, blind to the tool's output); the label is their consensus. "Correct" therefore means "agrees with those
judges", not "agrees with a human moderator". The judges' own rubric counts a slur as at least "medium" however the
chat uses it.

## The setups on the page
**Accuracy** is the mean of two F1 scores against the judges, shown as the gain over No AI:
- messages: toxic (low or worse) versus not, on 2,446 test lines (152 toxic), each line at the level the tool gives it
  with the setup's AI re-reading the lines it queues;
- people: "on the Watchlist" versus bad faith, on 1,490 test chatters (126 bad faith).

| setup | accuracy (gain over No AI) | msgs/min it keeps up with (12 GB GPU) |
|---|---|---|
| Gemma 4 31B (MTP build) | +66% | ~40 |
| 12B + 31B Hybrid, Review covering all flagged chatters | +62% | ≥150 live |
| 12B + 31B Hybrid, Review covering the Watchlist only | +54% | ≥150 live |
| Gemma 4 12B (MTP build) | +50% | ≥150 |
| Gemma 4 E4B (MTP build, no GPU) | +37% | ~20 on 8 CPU cores |
| No AI | +0% | any |

The hybrid's figures assume Review is pressed; between reviews it reads at the 12B's level. "Keeps up with" is the
busiest chat at which the AI's reads of flagged lines and chatters stay under 10 minutes behind.

## The default (Gemma 4 31B) on quiet chat
- Messages: recall 0.74 [0.64, 0.83] (about 3 in 4 toxic messages caught) at precision 0.64; 30 of 36 medium-or-worse
  messages caught.
- People: a Watchlist card is a bad-faith chatter 0.40 [0.28, 0.52] of the time, and the Watchlist shows 0.71 of the
  bad-faith chatters.

Intervals are 95% bootstrap intervals over test sets.

## Known limits
- Most remaining false alarms are the model calling a mocking or dismissive message "low" where the judges said none;
  most misses are one- or two-word messages.
- English only; measured on chat rates up to about 40 msgs/min. Busier chat is read in batches, which costs some
  accuracy on mild messages.
- Speeds are from the two test machines; elsewhere the page estimates them until Check Speed measures them.
- The test sets themselves are not published (they hold other people's chat).
