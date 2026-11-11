"""The local LLM tier (T3): a policy model that reads people and messages.

"Bad faith" is a custom policy, not a generic toxicity label, so this uses a
policy model chosen on the page (the default here: gemma4mtp:31b-n3-text) run
by Ollama in its own container (cw-ollama, bin/ollama.sh). The policy below is
the same taxonomy as archetypes.py, written for a model. Live (./start.sh) it re-reads messages
for toxicity in windows and reads candidates as people for the watchlist; it also reviews a
chatter when a mod clicks "review" (and alerts, with --llm-review but without --person-review),
and it labels silver for the learning rounds (rate_toxicity_window). A person read without one
of the four labels, or a one-message toxicity read without one of the four levels, raises
ValueError: the caller records an error, never a read.

    review(profile_dict, room_context) -> {"label", "confidence", "rules", "rationale"}
"""
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("CW_OLLAMA", "http://cw-ollama:11434")   # same podman network (cw-net)
MODEL = os.environ.get("CW_LLM_MODEL", "gemma4mtp:31b-n3-text")
# One context size for every call: Ollama reloads the model (~1 min from the
# share) whenever a request asks for a different num_ctx than the loaded one, and
# a caller whose timeout is shorter than the load aborts it, so mixed sizes can
# thrash (seen 2026-10-26: window calls 8192 vs reviews at the 4096 default).
NUM_CTX = 8192

POLICY = """\
# Policy: bad-faith participation in a Twitch live chat

You review one chatter for a channel's moderators. Decide whether their
messages show BAD-FAITH participation. You see their recent messages, the
surrounding chat, and context signals. Output JSON only.

## Definitions
Bad faith = participating mainly to provoke, demean, derail or exploit the
stream or its community, rather than to take part in it.

R1 Provocation: insults, dismissal or baiting aimed at the streamer or chat
   ("who even watches this", "L streamer", "you're trash", viewbot claims),
   especially when the room was calm and nothing prompted it.
R2 Concern trolling / JAQing: hostility wrapped in politeness or questions
   ("no offense but...", "genuine question, why is he always...", "is it true
   that...") where the question carries a hostile premise.
R3 Topic injection: dropping divisive politics, religion or drama into a room
   that was not discussing it, to derail or start fights.
R4 Harassment: slurs, sexual remarks about the streamer's body, threats,
   targeting a specific chatter, or "kys".
R5 Spam / scam / bot: self-promotion, follower selling, crypto, templated
   messages shared with other new accounts, link drops.

## Twitch chat conventions (read before judging)
- Tokens written as :Name: are EMOTES (small pictures). Repeating emotes, or
  posting only emotes, is normal participation, never spam by itself.
- Messages starting with "!" are bot commands (!link, !discord, !charm). Normal.
- Gaming slang is not an insult to a person: "dogwater", "mid", "washed",
  "ez", "L", "W", "cope", "ratio" are mostly banter about play or events.
- Many channels ARE about politics or news. If the surrounding chat is
  already discussing politics, a political message is the conversation, not
  topic injection. Only flag R3 when the room was clearly on something else.
- Opinions you find wrong or edgy are not bad faith unless aimed at people.

## NOT bad faith (do not flag)
- Gaming banter and trash talk inside a shared joke, chat-wide copypasta,
  emote spam, hype, swearing that is not aimed at anyone (but a slur, hate or
  a threat is never excused by being a joke, a copypasta or common here).
- Honest criticism, disagreement, or questions asked in good faith.
- Backseating (annoying, but usually well meant) -> label "mild".
- A one-off insult or swear at someone ("u idiot", "fuck you") from a chatter
  who is otherwise taking part -> label "mild" (toxic, not bad faith). Bad
  faith needs a pattern of hostility, OR slurs, hate, threats, harassment or
  sexual remarks aimed at people, OR a chatter who does nothing but provoke.
- Non-native English, slang, dialect. (The n-word in any spelling, the "-a"
  form included, is a slur, not dialect.)
- The standard is fixed: slurs, hate, threats and harassment count the same
  however many others in the room are saying them, and however normal they
  are in that chat. Harmless chants and copypasta (emotes, "L", "W", hype)
  are still participation.

## Output
{"label": "good_faith" | "mild" | "bad_faith" | "unclear",
 "rules": ["R1".."R5"], "confidence": "low" | "medium" | "high",
 "rationale": "1-2 sentences a moderator can read in five seconds"}
Prefer "unclear" over guessing when there is too little text.
"""

# The same rules with the boundaries the judges drew in one judged session
# spelled out (FINDINGS "LLM reviewer"): on its 101 judged chatters precision
# 0.41 -> 0.55, recall 1.00 both (DEV; confirm on a fresh judged session).
# tools/eval_reviewer.py --policy previous runs the old wording.
POLICY_PREVIOUS = POLICY
POLICY = POLICY.replace(
    "## NOT bad faith (do not flag)",
    """## Boundaries
- R5 means self-promotion, scams, bots and templated text across accounts.
  Repeating your own joke, taunt or reaction a few times is flooding, not spam:
  judge what it says, not that it repeats.
- Crude jokes about the game, its characters or on-screen events are not
  harassment of a person ("cuck ending", "nice cock" about a character). The
  fixed standard still applies: a slur, or sexual remarks or threats aimed at
  the streamer or a chatter, count wherever they appear.
- Telling a chatter to "go outside" / "touch grass" or mocking how someone plays
  is banter unless it comes with insults, slurs or a pattern of targeting.

## NOT bad faith (do not flag)""")

SCHEMA = {"type": "object", "properties": {
    "label": {"type": "string", "enum": ["good_faith", "mild", "bad_faith", "unclear"]},
    "rules": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    "rationale": {"type": "string"}},
    "required": ["label", "rules", "confidence", "rationale"]}
LABELS = tuple(SCHEMA["properties"]["label"]["enum"])
LEVELS = ("none", "low", "medium", "high")


def _checked(res, key, allowed, what):
    """The parsed answer, or ValueError when it is not an object with one of the allowed values: a
    reply outside the schema is an error, never a read (docs/review/tests_docs.md G4)."""
    if not isinstance(res, dict) or res.get(key) not in allowed:
        raise ValueError(f"the LLM's {what} is not one of {'/'.join(allowed)}: {str(res)[:120]}")
    return res


# Twitch sub-emote naming: lowercase channel prefix + capitalised code
# (abcdeHi, wxyzDance). Used when exact emote positions aren't available.
SUB_EMOTE = re.compile(r"^[a-z][a-z0-9]{2,}[A-Z][A-Za-z0-9]*$")


def mark_emotes(text, emotes=frozenset()):
    """Wrap emote tokens as :Name: so the model can't mistake them for words."""
    out = []
    for tok in text.split():
        bare = tok.strip(".,!?")
        if bare and (bare in emotes or SUB_EMOTE.match(bare)):
            out.append(tok.replace(bare, f":{bare}:"))
        else:
            out.append(tok)
    return " ".join(out)


def _content(p, ctx, emotes=frozenset()):
    msgs = "\n".join(f"- {mark_emotes(m['text'], emotes)}" for m in p["recent_messages"][-12:])
    room = "\n".join(f"{who}: {mark_emotes(text, emotes)}" for _, who, text in ctx[-20:])
    age = p.get("account_age_days")
    signals = [f"messages in channel: {p['messages']}",
               f"first-ever message in channel seen: {p['first_msg_seen']}",
               f"account age days: {age:.0f}" if age is not None else "account age unknown",
               f"subscribed months: {p['sub_months']}",
               f"started friction in calm room: {p['friction']['instigated']}, joined heated room: {p['friction']['joined_heated_room']}"]
    # no mod-action count: mod actions are never evidence (BASELINE 8.1, audit LB-2), and this read decides
    # "possible" (D7) and the A4h/S11 demotions
    return (f"## Chatter: {p['display']}\n### Their recent messages\n{msgs}\n\n"
            f"### Surrounding chat (most recent last)\n{room}\n\n### Signals\n" + "\n".join(signals))


def _post(body, timeout):
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def loaded(model):
    """Is the model in GPU/CPU memory right now?"""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as r:
            return any(m.get("name") == model or m.get("model") == model for m in json.load(r).get("models", []))
    except OSError:
        return False


def reachable(timeout=5):
    """Does the Ollama server answer at all (GET /api/version)? The dashboard's work queue waits, without
    spending a retry, while it does not (a restarting runner or container)."""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/version", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def transient(e):
    """A failure worth retrying: the server or its runner failed (HTTP 5xx: a runner that aborted on CUDA out of
    memory answers 500), the connection failed or timed out, or the reply was empty or cut off. A reply outside the
    schema (ValueError from _checked) or a refused request (HTTP 4xx: an unknown model) is final."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500
    if isinstance(e, json.JSONDecodeError):
        return True
    return isinstance(e, (OSError, TimeoutError, KeyError, http.client.HTTPException))    # IncompleteRead: cut off


def warm(model, timeout=600):
    """Load the model (an empty chat only loads it). A cold load from disk can take
    longer than one rating's timeout, and a timed-out request cancels the load, so
    every request after a model swap timed out (audit 2026-10-25: 30 of 150 lost).
    keep_alive -1: resident until unloaded (the picker swaps models by unloading them, docs/DASHBOARD.md)."""
    _post({"model": model, "messages": [], "keep_alive": -1, "options": _load_options(model)}, timeout)


def unload(model, timeout=60):
    """Take the model out of memory now (keep_alive 0). The picker and the hybrid review use it before loading another
    one: cw-ollama holds one model at a time (OLLAMA_MAX_LOADED_MODELS=1)."""
    _post({"model": model, "messages": [], "keep_alive": 0}, timeout)


def running_models(timeout=10):
    """The models in memory now (GET /api/ps), as names; OSError when the server does not answer."""
    with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=timeout) as r:
        return [m.get("name") or m.get("model") for m in json.load(r).get("models", [])]


def installed_models(timeout=10):
    """The models cw-ollama has (GET /api/tags), as names; OSError when the server does not answer."""
    with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=timeout) as r:
        return [m.get("name") or m.get("model") for m in json.load(r).get("models", [])]


def version(timeout=5):
    """Ollama's version (GET /api/version): the MTP builds need 0.40.11 or later."""
    with urllib.request.urlopen(OLLAMA + "/api/version", timeout=timeout) as r:
        return json.load(r).get("version")


# Per-model request settings (docs/HARDWARE_TIERS.md, Appendix B): each model in its own vendor format, tuned on dev and
# scored once on held-out test in hardware study round 2. gpt-oss keeps the live settings (temperature 0, reasoning
# THINK); Gemma 4 (gemma4:* and the MTP builds gemma4mtp:*) uses Google's sampling with thinking off, as frozen in
# data/study/hw_tiers2/configs/gemma31_final(_mtp3_h3).json; the small Gemmas (E4B, E2B) re-read lines with thinking on
# (configs/gemmae4b_final.json: num_predict 6000), their person read stays thinking off; Qwen runs with thinking off at
# Qwen's non-thinking sampling (configs/qwen36q4_final.json; Qwen 3.8: presence 0).
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
QWEN_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 1.5}
SMALL_GEMMA = re.compile(r"^gemma4(mtp)?:e[24]b")
# A cap on the layers Ollama puts on the GPU, per machine (CW_LLM_NUM_GPU; unset = Ollama's own fit). An MTP build needs
# one: Ollama sizes the split without the drafter, and the runner then aborts on CUDA out of memory (H3; 26 for
# gemma4mtp:31b-n3 on a 12 GB card, 15 on 8 GB, 36 for the 12B on 8 GB; 30 for the text-only gemma4mtp:31b-n3-text on
# 12 GB, data/study/quant/FACTS.md section 4).
NUM_GPU = os.environ.get("CW_LLM_NUM_GPU", "").strip()
# The model picker (picker.py) sets each model's cap on this machine here, so a hybrid's two models keep their own
# (the 12B live uncapped, the 31B reviewer at 26): model -> layers, or None for Ollama's own fit. A model in this map
# takes its own value; any other model takes NUM_GPU, as before.
NUM_GPU_BY_MODEL = {}


def num_gpu(model=None):
    """The GPU layer cap a call to this model carries on this machine, or None (Ollama's own fit)."""
    model = model or MODEL
    if model in NUM_GPU_BY_MODEL:
        v = NUM_GPU_BY_MODEL[model]
        return None if v in (None, "") else int(v)
    return int(NUM_GPU) if NUM_GPU else None


def num_thread():
    """CPU threads for a model's CPU part (owner 2026-11-10: leave room for the rest of the machine): CW_LLM_NUM_THREAD,
    which ./start.sh sets to the physical cores less CW_KEEP_CORES (default 2); unset, Ollama's own (every core).
    Sent with every load and call alike: a request whose load options differ would reload the model."""
    try:
        n = int(os.environ.get("CW_LLM_NUM_THREAD") or 0)
    except ValueError:
        n = 0
    return n if n > 0 else None


def _load_options(model=None):
    """The options that decide how a model is loaded: a request with others makes Ollama reload it."""
    ng = num_gpu(model)
    nt = num_thread()
    return {"num_ctx": NUM_CTX, **({"num_gpu": ng} if ng is not None else {}), **({"num_thread": nt} if nt else {})}


def _tune(body, model, sampling=True, reread=False):
    """A request body in the model's own format (see GEMMA_SAMPLING). sampling=False keeps the call's own temperature
    (the warning draft, the one-line rating): only the thinking switch and the GPU placement follow the model.
    reread: a window re-read (the small Gemmas think there)."""
    if model.startswith("gpt-oss"):
        body["think"] = THINK
    elif model.startswith("qwen"):
        body["think"] = False
        if sampling:
            body["options"] = {**QWEN_SAMPLING, "num_ctx": NUM_CTX,
                               **({"presence_penalty": 0} if model.startswith("qwen3.8") else {})}
    elif model.startswith("gemma4"):
        body["think"] = bool(reread and SMALL_GEMMA.match(model))
        if sampling:
            body["options"] = {**GEMMA_SAMPLING, "num_ctx": NUM_CTX,
                               **({"num_predict": 6000} if body["think"] else {})}
    ng = num_gpu(model)
    if ng is not None:
        body["options"]["num_gpu"] = ng
    nt = num_thread()
    if nt:
        body["options"]["num_thread"] = nt
    return body


_call_info = threading.local()


def last_load_s():
    """Seconds the last _chat call in this thread spent loading the model rather than answering (QL1, audit 2026-11-10):
    Ollama's `load_duration` in its reply, plus all of a call that found the model unloaded (the failed try and the
    reload). The dashboard's queue keeps it out of the measured call cost."""
    return getattr(_call_info, "load_s", 0.0)


def _chat(body, timeout):
    _call_info.load_s = 0.0
    t0 = time.time()
    try:
        out = _post(body, timeout)
    except OSError:                    # timeouts and connection errors
        if loaded(body["model"]):
            raise
        warm(body["model"])
        _call_info.load_s = time.time() - t0
        out = _post(body, timeout)
    try:
        _call_info.load_s += max(0.0, float(out.get("load_duration") or 0) / 1e9)
    except (AttributeError, TypeError, ValueError):
        pass
    return out


GUARDIAN_QUESTION = ("Using the policy above as the risk definition: is this chatter participating in "
                     "BAD FAITH (any of R1-R5)? Answer Yes or No.")


def review(profile, room_context, timeout=90, emotes=frozenset(), model=None):
    """Policy read of one chatter. `model` overrides CW_LLM_MODEL (bake-offs).
    Guardian-style models answer Yes/No; they are mapped onto the same output."""
    model = model or MODEL
    content = _content(profile, room_context, emotes)
    if "guardian" in model:
        body = {"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": NUM_CTX},
                "messages": [{"role": "system", "content": POLICY + "\n\n" + GUARDIAN_QUESTION},
                             {"role": "user", "content": content}]}
        out = _chat(body, timeout)
        txt = out["message"]["content"].strip()
        yes = txt.lower().lstrip("<score> ").startswith("yes") or "<score> yes" in txt.lower()
        return {"label": "bad_faith" if yes else "good_faith", "rules": [], "confidence": "medium",
                "rationale": txt[:200], "model": model}
    body = {"model": model, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": NUM_CTX},
            "messages": [{"role": "system", "content": POLICY},
                         {"role": "user", "content": content}]}
    _tune(body, model)
    out = _chat(body, timeout)
    res = _checked(json.loads(out["message"]["content"]), "label", LABELS, "person read")
    res["model"] = model
    return res


WARN_POLICY = """\
You help a Twitch moderator who wants to send a short warning to a chatter.
Write ONE message the moderator could paste into chat, addressed with @name.

Rules for the message:
- Calm, plain and respectful. No sarcasm, no insults, no lecturing.
- Name the specific behaviour (e.g. "insults aimed at other chatters",
  "comments about the streamer's body"), never judge the person.
- If it fits, state the expectation positively ("please keep it civil").
- Under 200 characters. English. No emojis, no hashtags, no threats beyond a
  plain mention that continuing may lead to a timeout, and only if the
  behaviour was serious.
- MOST chatters do NOT need a warning. Set "warranted": false unless a
  specific message clearly breaks common chat rules: insults or harassment
  aimed at someone, slurs, sexual comments about a person, threats, spam
  (the same message repeated), or disruptive provocation. Banter, jokes,
  opinions, criticism of the stream and complaints about moderation are
  not reasons on their own.
- The warning must name what that specific message did. Never invent a
  vague reason ("political insults") that the messages don't show.

Return JSON: {"warning": "<the message, or empty>", "warranted": true|false,
"why": "<one short line for the moderator>"}
"""

WARN_SCHEMA = {"type": "object", "properties": {
    "warning": {"type": "string"}, "warranted": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["warning", "warranted", "why"]}


def draft_warning(profile, room_context, timeout=60, emotes=frozenset(), model=None):
    """A ready-to-paste, calm warning for a mod. Never sent automatically."""
    model = model or MODEL
    reasons = "; ".join(r["reason"] for r in profile.get("reasons", [])[:5]
                        if r.get("family") in ("badfaith", "mild"))
    content = _content(profile, room_context, emotes) + (
        f"\n\n### What the tool noticed\n{reasons or 'nothing specific'}"
        f"\nThe tool's overall read: {profile.get('verdict', '')}"
        f"\n\nThe chatter's name is @{profile.get('display', 'chatter')}.")
    body = {"model": model, "stream": False, "format": WARN_SCHEMA, "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
            "messages": [{"role": "system", "content": WARN_POLICY}, {"role": "user", "content": content}]}
    _tune(body, model, sampling=False)
    out = _chat(body, timeout)
    res = json.loads(out["message"]["content"])
    res["model"] = model
    return res


TOX_POLICY = """\
You rate ONE Twitch chat message for toxicity, using the preceding chat lines
only as context (is it a reply, a quote, a shared joke, game talk?).

TOXIC = insulting, demeaning, hateful, sexually harassing, threatening or
mocking language, INCLUDING when aimed at a third party or a group outside
chat ("those guys never touch grass" is toxic mockery). NOT toxic: plain
swearing with nobody insulted ("holy shit", "fuck yeah"), self-deprecation
("I'm dumb af"), game talk ("shoot him", "kill the boss"), sincere criticism
or disagreement, quoting something to condemn it. Tokens written as :Name:
are emotes.

Levels: none | low (mild jab, crude) | medium (clear insult or mockery) |
high (hate, slur, threat, sexual harassment, "kys", smearing insinuation).
Return JSON {"toxicity": level, "target": "streamer|chatter|third_party|group|general|",
"kind": "short word or empty"}.
"""

TOX_SCHEMA = {"type": "object", "properties": {
    "toxicity": {"type": "string", "enum": ["none", "low", "medium", "high"]},
    "target": {"type": "string"}, "kind": {"type": "string"}},
    "required": ["toxicity", "target", "kind"]}


def rate_toxicity(message, room_lines=(), timeout=60, emotes=frozenset(), model=None):
    """LLM read of one message's toxicity (same rubric the judges used)."""
    model = model or MODEL
    room = "\n".join(mark_emotes(l, emotes) for l in room_lines[-6:])
    content = f"### Chat before\n{room}\n\n### Message to rate\n{mark_emotes(message, emotes)}"
    body = {"model": model, "stream": False, "format": TOX_SCHEMA, "options": {"temperature": 0, "num_ctx": NUM_CTX},
            "messages": [{"role": "system", "content": TOX_POLICY}, {"role": "user", "content": content}]}
    _tune(body, model, sampling=False)
    out = _chat(body, timeout)
    res = _checked(json.loads(out["message"]["content"]), "toxicity", LEVELS, "toxicity level")
    res["model"] = model
    return res


WINDOW_POLICY = TOX_POLICY.replace(
    "You rate ONE Twitch chat message for toxicity, using the preceding chat lines\nonly as context",
    "You rate EVERY numbered Twitch chat line below for toxicity. Earlier lines\nmarked [context] are not rated; use all lines as context") + """
Rate each numbered line on its own (who wrote it, what it means in this
conversation). Do not under-rate mild hostility: "low" INCLUDES dismissive
taunts and put-downs aimed at the streamer or a person ("nobody likes you",
"get a real job", "you're washed", "cry about it"), mocking someone's looks,
voice or beliefs, sarcastic jabs, and joining a pile-on against someone. A
line can be short and still be "low". Plain hype, emotes, jokes about the
game, questions and sincere disagreement stay "none". Return JSON {"ratings": [{"n": <line number>, "toxicity": level,
"target": ..., "kind": ...}, ...]} with one entry per numbered line.
"""

# E2 (2026-11-10, data/study/e_checks/RESULT.md; BASELINE section 30): the live 31B re-reads with the pile-on clause
# narrowed to lines that themselves demean or insult. Tested only on the 31B (gemma4mtp:31b-n3-text; the projector
# build has the same weights), against today's wording under the frozen 10:1 rule on Route A's pool P: 21 false alarms
# cut beyond noise, no toxic, medium+ or severe line lost. Every other model keeps WINDOW_POLICY, so the pipeline's
# teacher (t2, gpt-oss-safeguard:20b) and its labels do not change.
PILE_ON_TODAY = "sarcastic jabs, and joining a pile-on against someone."
PILE_ON_E2 = "sarcastic jabs, and joining a pile-on against someone when the line\nitself demeans or insults them."
GEMMA31 = re.compile(r"^gemma4(mtp)?:31b")


def window_policy(model):
    """The re-read's policy text for this model: WINDOW_POLICY with the E2 pile-on clause for the 31B, else as is."""
    if GEMMA31.match(model or ""):
        assert WINDOW_POLICY.count(PILE_ON_TODAY) == 1, "the pile-on clause moved: re-check E2"
        return WINDOW_POLICY.replace(PILE_ON_TODAY, PILE_ON_E2)
    return WINDOW_POLICY


# Version of the toxicity wording the teacher labels with (recorded on relabelled
# rows; bump whenever WINDOW_POLICY / TOX_POLICY or the glossary change).
#   t1  WINDOW_POLICY
#   t2  (2026-10-25) t1 + the coded-language reference block ("glossary_rule",
#       glossary g2) where a glossary term appears; everywhere else t1's exact
#       prompt. Coded test set, test half: recall 0.566 -> 0.763, precision
#       0.754 -> 0.817, sign test p = 0.0009 (docs/FINDINGS "coded language")
TEACHER_VERSION = "t2"
TEACHER_CODED, TEACHER_GLOSSARY = "glossary_rule", "g2"
# gpt-oss reasoning budget for every call ("low" = fastest; tools/eval_teacher.py
# compares budgets against the judges)
THINK = "low"

# Candidate (tools/eval_teacher.py): the judges' "low" spelled out. The teacher
# caught only 54-77% of judge-"low" messages (FINDINGS "teacher"), mostly these
# kinds. Examples drawn from two judged sets, so the clean check
# is messages_all and the capture gold sets.
WINDOW_POLICY_CANDIDATE = WINDOW_POLICY.replace(
    "A\nline can be short and still be \"low\".",
    """Also "low": mocking how someone plays or performs ("so bad",
"try aiming", "EU aim", "not even hardest difficulty", "skill issue"),
trash-talking a team, company or group ("United are dog shit", "devs are
clowns"), crude sexual jokes ("nice cock", "small hole"), and impatient insults
("hurry the fuck up", "slow as shit"). A slur is never lower than "medium".
Frustration at the game itself with nobody put down ("worst fucking ending",
"shit ui") stays "none". A line can be short and still be "low".""")

WINDOW_SCHEMA = {"type": "object", "properties": {"ratings": {"type": "array", "items": {
    "type": "object", "properties": {
        "n": {"type": "integer"},
        "toxicity": {"type": "string", "enum": ["none", "low", "medium", "high"]},
        "target": {"type": "string"}, "kind": {"type": "string"}},
    "required": ["n", "toxicity", "target", "kind"]}}}, "required": ["ratings"]}


# Coded-language reference for the teacher (tools/eval_teacher.py tests it; not
# used by the pipeline until a teacher version adopts it): definitions of the
# hate codes and coded slurs the 20B doesn't know (data/scratch/foid_probe.py,
# dogwhistle_probe.py), added only when a line in the window matches a trigger.
GLOSSARY_FILE = os.environ.get("CW_GLOSSARY") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                                               "coded_glossary.json")
_GLOSSARY = None


def glossary_version():
    return json.load(open(GLOSSARY_FILE))["version"].split()[0]


def same_prompt(version, lines):
    """Would the current teacher send the prompt that `version` sent for these
    lines (so its label stands)? t1 -> t2 differs only where a glossary term
    appears (jobs.job_relabel re-rates just those windows)."""
    return version == TEACHER_VERSION or (version in (None, "t1") and TEACHER_VERSION == "t2"
                                          and not glossary_lookup(lines))


def glossary_lookup(lines):
    """The glossary entries whose trigger matches the text of any line (the
    "name: " prefix is skipped, so a login can't trigger one)."""
    global _GLOSSARY
    if _GLOSSARY is None:
        g = json.load(open(GLOSSARY_FILE))
        _GLOSSARY = [{**e, "rx": re.compile(e["trigger"], re.I)} for e in g["entries"]]
    from .text import normalize
    texts = [normalize(l.partition(": ")[2] if ": " in l else l) for l in lines]
    return [e for e in _GLOSSARY if any(e["rx"].search(t) for t in texts)]


# Rule for coded hate ("glossary_rule"): the rubric says what a slur is worth,
# not that a code counts as one
CODED_RULE = """
Coded hate counts as hate: a hate code, coded slur or dog whistle used in its
coded sense is rated like the open slur or hateful statement it stands for
("medium" or "high"), also as a joke or meme and however common it is in the
room. The same word or number used innocently stays "none"."""

# Worked examples (policy "glossary_rule_examples"), written for this purpose:
# not taken from any judged set
CODED_EXAMPLES = """
Examples: "1488 gang" = high (neo-Nazi code); "he went 14-88 in ranked" = none.
"check his early life section" about a person = high (insinuates he is Jewish);
"the boss's early life cutscene" = none. "despite being 13%" = high (racist
statistic); "13/50 health, heal up" = none. "go join the 41%" to a trans
streamer = high (suicide taunt); "41% battery" = none."""
CODED_MODES = ("glossary", "glossary_rule", "glossary_rule_examples")


def reference_text(entries, mode="glossary"):
    """The block added to the teacher's instructions, only when a coded term
    appears (so ordinary chat gets exactly the current prompt)."""
    if not entries:
        return ""
    rows = [f"- {e['means']}" + (f" Not coded: {e['innocent']}." if e["innocent"] else "") for e in entries]
    return ("\n\n### Reference: coded terms that appear in these lines\n" + "\n".join(rows) +
            "\nA reference says what a term means when it is used as a code; decide from the line and"
            " the chat whether it is used that way." + (CODED_RULE if mode != "glossary" else "") +
            (CODED_EXAMPLES if mode == "glossary_rule_examples" else ""))


def rate_toxicity_window(context_lines, lines, timeout=180, emotes=frozenset(), model=None,
                         policy=None, coded=None):
    """Rate many consecutive chat lines in one call (shared context = far higher
    throughput). lines: ["name: text", ...]. Returns a list aligned with lines
    (None where the model skipped a line; the dashboard re-queues it once, alone: serve.py). policy / coded (one of CODED_MODES,
    "" for none, i.e. teacher t1): teacher candidates (tools/eval_teacher.py,
    eval_coded.py); the defaults are the current teacher (the 31B: with the E2 pile-on clause, window_policy)."""
    model = model or MODEL
    system = policy or window_policy(model)
    if coded is None:
        coded = TEACHER_CODED
        if glossary_version() != TEACHER_GLOSSARY:
            raise RuntimeError(f"the glossary is {glossary_version()}, teacher {TEACHER_VERSION} uses "
                               f"{TEACHER_GLOSSARY}: bump TEACHER_VERSION (and relabel) when the glossary changes")
    if coded:
        system += reference_text(glossary_lookup(list(context_lines) + list(lines)), coded)
    ctx = "\n".join(f"[context] {mark_emotes(l, emotes)}" for l in context_lines)
    body_lines = "\n".join(f"{i}. {mark_emotes(l, emotes)}" for i, l in enumerate(lines, 1))
    body = {"model": model, "stream": False, "format": WINDOW_SCHEMA, "options": {"temperature": 0, "num_ctx": NUM_CTX},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"{ctx}\n\n### Lines to rate\n{body_lines}"}]}
    _tune(body, model, reread=True)
    out = _chat(body, timeout)
    got = {}
    for r in json.loads(out["message"]["content"]).get("ratings", []):
        try:
            n = int(r.get("n"))          # "1" counts as line 1 (the schema asks for a number)
        except (TypeError, ValueError):
            continue
        if r.get("toxicity") in LEVELS:     # an unknown level is a skipped line, never a label
            got[n] = r
    return [got.get(i) for i in range(1, len(lines) + 1)]
