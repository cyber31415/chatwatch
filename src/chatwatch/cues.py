"""Inspectable cue lexicon for reading intent from very little text.

Each cue is a regex over the *normalised* message (see text.normalize), a
family, and a one-line reason that is shown to mods when it fires. Nothing
here is learned; the weights that turn cues into archetype evidence live in
archetypes.py, and evaluate.py checks them against real mod actions.

Families
  friction   - aimed at the streamer, chat or a chatter: dismissal, insults,
               mockery, hedged attacks, loaded questions, comparisons
  injection  - divisive topics dropped into a room that wasn't discussing them
  creep      - comments on the streamer's body / sexual remarks
  spam       - promotion, scams, follower selling, links to strangers
  benign     - greetings, newcomer self-identification, honest questions,
               support, hype, backseating (annoying, not bad faith)
  callout    - other chatters reacting to someone: "mods", "ratio", "who asked"
               (used as *reaction* evidence about the person they answer)

The patterns are deliberately conservative. Gaming banter ("ur trash at this
lol") overlaps with real hostility; a single cue is weak evidence and the
archetype layer needs several, or context, before it calls anything bad faith.
"""
import re

# who a remark can be aimed at
STREAMER = r"(?:streamer|this (?:guy|dude|girl|woman|man|clown|kid|streamer)|he|she|him|her|bro|bruh|homie|u|you|ya|ur|youre|you're|your|this (?:stream|channel|content))"
CHAT = r"(?:chat|you guys|yall|y'all|everyone here|this chat|these people|mods|viewers|simps)"
NEG_ADJ = (r"(?:trash|garbage|dogshit|shit|ass|bad|terrible|awful|horrible|stupid|dumb|"
           r"idiot(?:ic)?|braindead|brain ?dead|retarded|clown(?:ish)?|loser|cringe|"
           r"washed(?: up)?|mid|annoying|ugly|fat|fake|boring|unfunny|pathetic|"
           r"irrelevant|overrated|insufferable|unwatchable|a joke|dogwater|"
           r"talentless|cringy|weird|creepy|lame|so bad|actually bad|the worst)")

_CUES = [
    # ---------------------------------------------------------- friction
    ("insult_target", "friction",
     rf"\b{STREAMER}\b(?: (?:is|are|r|s|look|looks|sound|sounds|so|really|actually|just|literally|kinda|lowkey|such an?|being))* (?:an? )?{NEG_ADJ}\b",
     "calls the streamer/addressee something negative"),
    ("insult_chat", "friction",
     rf"\b{CHAT}\b(?: (?:is|are|r|so|really|actually|just|literally))* (?:an? )?{NEG_ADJ}\b",
     "insults chat as a whole"),
    ("you_are_x", "friction",
     r"\b(?:ur|youre|you're|you are|u r|u are) (?:an? )?(?:idiot|moron|clown|loser|joke|fraud|bum|nerd|bot|npc|simp|sellout|hack)\b",
     "direct name-calling"),
    ("insult_word", "friction",
     r"(?:^|\byou(?:s|'re| are| r)? (?:a |an |such a )?|\bur (?:a |an )?|\bthis |\bbro |\bl )"
     r"(?:fat ?ass|fatty|bitch|asshole|dick ?rider|cuck|bozo|clown|loser|idiot|moron|dumbass|jackass|"
     r"dipshit|scumbag|degenerate|virgin|neckbeard|pussy|simp|bum|fraud|"
     r"cunt|twat|prick|wanker|dork|weirdo|freak|creep)s?\b",
     "personal insult word"),
    # body/appearance put-downs need an explicit "you look like / you are" frame:
    # bare "egg" or "pig" is far more often a chat meme than an insult
    ("appearance_insult", "friction",
     r"\b(?:you|u|ur|he|she|this guy) (?:look|looks|r|are|is) (?:like )?(?:a |an )?(?:egg|pig|whale|potato|thumb|goblin|troll|homeless (?:man|guy|person)|fat \w+)\b|"
     r"\bwhy is there an? (?:homeless|fat|ugly|bald) \w+ on my screen\b",
     "appearance put-down"),
    # intolerance (Rossini): dehumanising a group, the more harmful kind of incivility
    ("dehumanize", "creep",
     r"\b(?:they|they're|theyre|these|those|all|them)\b[^.?!]{0,30}?\b(?:animals|vermin|subhuman|parasites|cockroaches|savages|rats|"
     r"invaders|degenerates|filth|scum|apes|monkeys)\b",
     "dehumanising a group"),
    # smearing by insinuation (a pile-on staple): accusations of predation or
    # crimes, no slur or swear needed. Needs a person as subject/object.
    ("insinuation", "creep",
     r"\b(?:p[-. ]?d[-. ]?f(?:ile)?|pedo(?:phile)?s?|groomer|grooming|predator|nonce|diddler|chomo)\b|"
     r"\b(?:you|u|he|bro|this guy|streamer|him)\b[^.?!]{0,25}\b(?:talks? to|likes?|around|near|with|manipulat\w*|touch\w*) (?:kids|children|minors|little (?:kids|boys|girls))\b|"
     r"\b(?:check|seize) (?:his|your|ur) (?:hard ?drive|laptop|computer|basement)|\b(?:chris hansen|have a seat|on (?:a|the) (?:list|registry))\b|"
     r"\bcreepy (?:youth pastor|uncle|old man|vibes|ass)\b",
     "insinuates the person is a predator / criminal"),
    ("ban_evasion_claim", "friction",
     r"\b(?:(?:you|u) banned me|got banned on my (?:main|other)|(?:new|alt|another|second) account(?:s)?\b.*\bban|"
     r"ban(?:ned)? me.*\b(?:new|alt) account|ban evad(?:e|ing)|can'?t ban me|you can'?t stop me)\b",
     "says they are evading a ban"),
    ("threat", "creep",
     r"\b(?:(?:got|have) a bullet for|(?:i'?ll|i will|gonna|going to|i'?m gonna) (?:kill|shoot|stab|find|hurt|beat) (?:you|u|him|her|them|all)|"
     r"(?:shoot|kill|burn|hang) (?:them|em|you|u) all|gas and (?:a )?match|i know where (?:you|u) live|watch your back|"
     r"swat(?:ting)? (?:you|u|him|her))\b",
     "threat of violence"),
    # wished or threatened sexual violence against a person; not past-tense game talk ("got raped by the boss")
    ("sexual_threat", "creep",
     r"\b(?:(?:hope|wish|pray)\b[^.?!]{0,25}\b(?:gets?|getting|be|is) (?:raped|molested)|"
     r"(?<!dont )(?<!don't )(?<!doesnt )(?<!doesn't )(?<!not )(?<!never )(?<!shouldnt )"
     r"(?:deserves?|should|needs? to) (?:to )?(?:get|be) (?:raped|molested)|go (?:get|be) raped|"
     r"(?:i'?ll|i will|gonna|going to|i'?m gonna|let me|lemme|imma) (?:rape|molest) (?:you|u|him|her|them|ur|your))\b",
     "wishes or threatens sexual violence"),
    ("dismiss_stream", "friction",
     r"\b(?:who (?:even |tf |the fuck )?(?:watches|watching|likes|supports) (?:this|him|her|u|you)|"
     r"nobody (?:asked|cares)|no one (?:asked|cares)|unwatchable|(?:you|u|he|she|streamer|this (?:guy|stream|channel)|bro|this) (?:really |actually |so )?fell off|"
     r"dead (?:stream|channel)|this (?:stream|content|channel) (?:sucks|blows|is (?:so |actually |literally )?(?:trash|garbage|boring|dead|ass|mid|cringe|unwatchable))|"
     r"(?:so|this is so|very|super) boring|snooze ?fest|get a (?:real )?job|touch grass|"
     r"go outside|(?:you|u|he|she) (?:have|has) no (?:life|friends|talent)|"
     r"waste of (?:time|my time)|worst (?:stream|streamer|content)|end the stream|turn off the stream|"
     r"(?:when does|when will) (?:it|he|she) (?:stop|shut up)|stop talking|nobody wants to (?:hear|see|watch))\b",
     "dismisses the stream or streamer"),
    ("l_streamer", "friction",
     r"(?:^|\s)(?:huge |massive |big )?l (?:streamer|stream|content|chat|bozo)\b|\bl ?\+ ?ratio\b",
     "\"L streamer\" style put-down"),
    ("viewbot_claim", "friction",
     r"\b(?:viewbot(?:s|ted|ting|ter)?|fake (?:viewers|chat|followers|subs)|bought (?:viewers|followers)|"
     r"chat (?:is|are) (?:all )?(?:bots|fake|dead))\b",
     "accuses the channel of fake viewers"),
    ("mockery", "friction",
     r"\b(?:imagine (?:being|thinking|watching|paying|subbing)|cope|seethe|mald(?:ing)?|"
     r"cry about it|stay mad|skill issue|ratio(?:ed)?|ok boomer|nerd emoji|keep crying|"
     r"didn'?t ask|you mad|u mad|cry more|cringe ?lord)\b",
     "mockery / taunting"),
    ("hedged_attack", "friction",
     r"\b(?:no offen[cs]e|not to be (?:rude|mean)|not trying to (?:hate|be (?:rude|mean))|"
     r"no hate but|with all (?:due )?respect|i'?m just saying|just saying|"
     r"i don'?t mean to be rude|just being honest|respectfully|not gonna lie but|"
     r"im not hating but|i'?m not hating but)\b",
     "politeness hedge that usually precedes criticism"),
    ("loaded_question", "friction",
     r"\b(?:why (?:is|are|does|do|did) (?:he|she|you|u|streamer|this guy|chat|the mods?) (?:so|always|never|even|still|literally)\b|"
     r"is it true (?:that )?(?:he|she|you|u|streamer)|"
     r"(?:isn'?t|wasn'?t|didn'?t) (?:he|she|you|u) (?:the one|get|got|say)|"
     r"why do (?:people|you guys|yall|u guys|you all) (?:even )?(?:watch|like|support|sub)|"
     r"how (?:is|are) (?:he|she|you|u|this) (?:so )?(?:bad|popular|famous|allowed)|"
     r"how do (?:you|u) (?:not|still) |"
     r"genuine(?:ly)? (?:question|asking)|honest question|serious question|just asking)\b",
     "question with a hostile premise (JAQing)"),
    ("comparison", "friction",
     r"\b(?:go watch \w+|watch \w+ instead|\w+ (?:is|was) (?:way |so much )?better than (?:you|u|him|her|this)|"
     r"(?:you|u|he|she) (?:could|should) learn from \w+|(?:even )?my (?:little )?(?:brother|sister|mom|dog|grandma) (?:plays|could do|is) better)\b",
     "unfavourable comparison"),
    ("leaving_threat", "friction",
     r"\b(?:unfollow(?:ed|ing)?|unsub(?:bed|bing|scribed|scribing)?|i'?m (?:leaving|done watching)|"
     r"bye (?:loser|clown|nerd)|lost a (?:viewer|follower|sub))\b",
     "announces leaving in protest"),
    ("swear_at", "friction",
     r"\b(?:(?<!will )(?<!gonna )(?<!going to )fuck (?:you|u)(?! money| up)|fuck off|fuck (?:this (?:guy|dude|streamer|stream)|chat|the mods)|stfu|shut (?:the fuck )?up|kys|kill yourself|"
     r"go die|die in a|hope (?:you|u|he|she) (?:die|dies|lose|loses|fail))\b",
     "hostile directive"),
    ("mod_complaint", "friction",
     r"\b(?:(?:power ?tripping|powertrip|abusive|cringe|corrupt|weirdo|loser) mods?|mods? (?:are|r) (?:trash|losers|weird|sensitive|power ?tripping|corrupt|soft)|"
     r"free speech|censorship|cant say anything|can'?t say anything|got timed out for)\b",
     "complains about moderation"),

    # --------------------------------------------------------- injection
    ("divisive_politics", "injection",
     r"\b(?:trump|biden|kamala|maga|liberals?|libtards?|leftists?|conservatives?|democrats?|"
     r"republicans?|woke(?:ness)?|dei|antifa|commies?|communists?|nazis?|fascists?|zionists?|"
     r"israel|palestine|gaza|hamas|immigrants?|illegals?|feminists?|incels?|sjws?|"
     r"abortion|pro ?life|pro ?choice|vaccine|vaxx(?:ed)?|great replacement|globalists?|soros|"
     r"trans(?:gender)?s?|pronouns|lgbtq?\+?|groomers?|alpha males?|red ?pill)\b",
     "divisive political / culture-war topic"),
    ("divisive_religion", "injection",
     r"\b(?:jesus (?:saves|loves|is (?:lord|king|coming))|repent|accept (?:jesus|christ|god)|"
     r"allah|islam(?:ic)?|muslims?|christians?|atheists?|bible says|god hates|end times)\b",
     "religious proselytising / topic"),
    ("drama_bait", "injection",
     r"\b(?:did (?:you|u) (?:see|hear) what \w+ (?:said|did)|thoughts on \w+|what do (?:you|u) think (?:of|about) \w+|"
     r"react to \w+|(?:talk|speak) (?:about|on) the \w+ (?:drama|situation|allegations)|allegations|exposed)\b",
     "fishing for drama"),

    # Slurs and dehumanising terms. Matched on the normalised text, so leet /
    # homoglyph / spaced spellings land here too. Reclaimed in-group use is a
    # known false positive (Sap et al. 2019): this is evidence, not a verdict.
    ("slur", "creep",
     # Every form of the n-word counts, the "-a" form included (owner decision
     # 2026-10-25, docs/FINDINGS: the same root slur; the tool cannot know who
     # is saying it or whether the streamer allows it). This reverses the
     # dialect-fairness exemption of FINDINGS §8 on purpose: it raises flag
     # rates in chats where the in-group form is common.
     r"\b(?:nigg(?:ers?|uh|a+h?|az)s?|fag(?:got|gots|s)?|trann(?:y|ies)|troons?|kikes?|chinks?|spics?|"
     r"wetbacks?|beaners?|gooks?|ragheads?|towelheads?|sand ?n\w+|coons?|f[e]?moids?|foids?|"
     r"retard(?:s|ed|ation)?|mongoloids?|subhuman|untermensch|(?:these|those|you|such|fucking|absolute|bunch of|all) re ?gard(?:s|ed)?(?! to)|re ?gards (?:in|of) (?:chat|here))\b",
     "slur / dehumanising term"),

    # ------------------------------------------------------------- creep
    ("appearance_sexual", "creep",
     r"\b(?:(?:you|u|ur|she|her|shes|she's|you're|youre) (?:are |is |look |looks |so |r )*(?:hot|sexy|thicc|thick|fat|ugly|bald)\b|"
     r"(?:show|pics? of|send|see|flash) (?:us |me )?(?:your |ur |her |the |them )?(?:feet|toes|body|ass|boobs|tits|thighs|dick|dih|cock|pussy|nudes)|feet pics|boobs|tits|booba|"
     r"twerk (?:for|on) (?:me|us)|(?:your|ur|her|signed|used) (?:underwear|panties|bra|socks)|(?:your|ur|her) smell|sugar tits|"
     r"marry me|be my (?:gf|girlfriend|wife)|send (?:pics|nudes|feet)|onlyfans|only fans|"
     r"step on me|sit on (?:my|me)|are you single|r u single)\b",
     "remark about body / sexual comment"),

    # -------------------------------------------------------------- spam
    ("promo", "spam",
     r"\b(?:check (?:out )?my (?:channel|stream|page|profile|music|song|bio)|follow (?:me|my)|"
     r"sub(?:scribe)? to (?:me|my)|come watch me|i stream too|visit my|f4f|follow for follow|"
     r"promot(?:e|ion|ing) (?:your|ur|my)|(?:grow|boost) (?:your|ur) (?:channel|stream))\b",
     "self-promotion"),
    ("scam", "spam",
     r"\b(?:(?:buy|cheap|best|real) (?:viewers|followers|primes?|subs|views)|viewers on|"
     r"crypto|bitcoin|nft|airdrop|free (?:skins|robux|vbucks|v-bucks|nitro|gift ?cards?)|"
     r"giveaway|dm me for|telegram|whatsapp|cashapp|cash app|paypal me|"
     r"(?:wanna|want to) (?:become|get|be) famous|streamboo|dogehype|viewerboost)\b",
     "scam / follower-selling pattern"),

    # ------------------------------------------------------------ benign
    ("newcomer", "benign",
     r"\b(?:first time (?:here|chatting|watching|in (?:chat|the stream)|catching)|new here|"
     r"just found (?:you|your|this)|hello from|greetings from|watching from|"
     r"long ?time lurker|been lurking|first (?:message|msg|chat)|new viewer|new follower|"
     r"just followed|found (?:you|ur|your) (?:on|from|through|via))\b",
     "identifies as new"),
    ("greeting", "benign",
     r"^(?:hi+|hey+|hello+|heyo|yo+|sup|hiya|howdy|evening|morning|good (?:morning|evening|afternoon)|o7|gm)\b(?: (?:chat|everyone|all|guys|yall|there|streamer|bro|\w+))?[\s!.]*$|\bhow(?:'?s| is| are) (?:everyone|chat|you|u|it going|the stream)\b",
     "greeting"),
    ("honest_question", "benign",
     r"\b(?:what (?:game|is this|keyboard|mouse|headset|mic|chair|monitor|settings|sens|rank|song|music)|"
     r"what'?s the (?:song|game|name)|how long (?:have|has|is|until)|when (?:is|does|do|will) (?:the|you|u)|"
     r"is this (?:the|a|your) |how do (?:i|you) (?:get|unlock|beat|do)|which (?:one|game|character))\b",
     "ordinary question"),
    ("support", "benign",
     r"\b(?:love (?:the|your|ur|this) (?:stream|streams|content|channel|vibe|vibes)|you'?re (?:so )?(?:good|great|funny|amazing)|"
     r"keep it up|thank(?:s| you) for (?:the|streaming)|w streamer|w stream|goat(?:ed)?|"
     r"best streamer|have a (?:good|great|nice) (?:stream|day|night)|great stream|good stream|gg)\b",
     "supportive"),
    ("hype", "benign",
     r"^(?:w+|lets? go+|let'?s go+|pog\w*|clip (?:it|that)|insane|no way+|holy+ (?:shit|moly)|lmf?ao+|lol+|o+mg+|gg+|ez+|"
     r"wow+|nice+|clutch|goated|banger|sheesh|based|real|true+|facts|yes+|no+|what+|huh+|\?+|!+)[\s!?.]*$",
     "short reaction / hype"),
    ("backseat", "benign",
     r"\b(?:you should (?:have |go|use|try|do|pick|buy|get)|just (?:go|use|jump|run|press|click|do)|"
     r"wrong way|go (?:left|right|back|upstairs|downstairs)|use (?:the|your) \w+|"
     r"why (?:didn'?t|don'?t) (?:you|u) (?:just )?|behind (?:you|u)|you missed|u missed)\b",
     "backseating / unsolicited directions"),

    # ----------------------------------------------------------- callout
    # strong: an appeal to moderation; can be attributed by proximity
    ("callout_mods", "callout",
     r"(?:^|\s)@?mods+\b(?! are)|\bmodcheck\b|\b(?:ban|timeout|time out|report) (?:him|her|them|this (?:guy|dude|clown|kid))\b|"
     r"\bget (?:him|her|them) out\b|\bdon'?t feed (?:the )?trolls?\b|\bignore (?:him|her|them|the troll)\b",
     "chat appeals to mods"),
    # weak: social pushback; only counts when it explicitly replies to / @s someone
    ("callout_social", "callout",
     r"\b(?:who asked|nobody asked|ratio|weirdo|weird(?:champ|ge)|bait(?:ed)?|troll(?:ing)?|"
     r"touch grass|cringe|yikes|bro what|what is wrong with (?:you|u|him|her))\b",
     "chat pushes back"),
]

CUES = [(n, fam, re.compile(rx, re.I), why) for n, fam, rx, why in _CUES]

# threat V1 (data/study/threat_v1/RULE_THREAT_V1.md): "gonna/going to kill|shoot|... you|him|..." counts only
# with the speaker as the agent: I or we, or a bare "gonna/going to" opening the message (after @mentions).
# "SHES GONNA KILL YOU", "hes gonna find you", "you're gonna beat him" are warnings, narration and coaching.
# The other alternatives (a bullet for, the "kill/shoot/burn/hang them all" chants, gas and a match, i know
# where you live, watch your back, swatting) are unchanged. The cue fires only when both its own regex and
# this one match, so this can only remove hits (data/scratch/threatdev/threat_variants.py V1, same regex).
_SPEAKER = (r"(?:\b(?:i'?ll|i will|i'?m (?:gonna|going to)|i am (?:gonna|going to)|imma|ima|i (?:gonna|going to)|"
            r"we'?ll|we will|we're (?:gonna|going to)|we are (?:gonna|going to)|"
            r"(?<!they )(?<!you )(?<!u )(?<!he )(?<!she )(?<!it )were (?:gonna|going to)|"
            r"we (?:gonna|going to))|^(?:@\w+[ ,]*)*(?:gonna|going to))")
THREAT_SPEAKER = re.compile("|".join([
    r"\b(?:got|have) a bullet for\b",
    rf"{_SPEAKER} (?:kill|shoot|stab|find|hurt|beat) (?:you|u|him|her|them|all)\b",
    r"\b(?:shoot|kill|burn|hang) (?:them|em|you|u) all\b", r"\bgas and (?:a )?match\b",
    r"\bi know where (?:you|u) live\b", r"\bwatch your back\b", r"\bswat(?:ting)? (?:you|u|him|her)\b"]), re.I)
CUE_FAMILY = {n: fam for n, fam, _, _ in CUES}
CUE_REASON = {n: why for n, _, _, why in CUES}
FAMILIES = ("friction", "injection", "creep", "spam", "benign", "callout")


def match(norm_text, raw_lower=None):
    """Return {cue_name: matched_text} over the normalised text.

    If raw_lower is given, also report which cues *only* matched after
    normalisation; that is filter-evasion evidence.
    """
    hits, evaded = {}, []
    for name, _, rx, _ in CUES:
        if name == "threat" and not THREAT_SPEAKER.search(norm_text):
            continue
        found = list(dict.fromkeys(m.group(0) for m in rx.finditer(norm_text)))
        if found:
            # several distinct matches of one cue ("you look so hot" + "show
            # us your feet") are joined with " | " and weigh a bit more
            hits[name] = " | ".join(found[:3])
            # disguising only means something for hostile words: "L0L" or
            # "Gggg" turning into lol/gg is not filter evasion
            if raw_lower is not None and not rx.search(raw_lower) and CUE_FAMILY[name] not in ("benign", "callout"):
                evaded.append(name)
    return hits, evaded


# Emotes whose meaning carries the same signal as words. Fixes lesson #1 from
# the handoff (LUL/KEKW/monkaS never reached the lexicon). Keys are exact
# emote codes; the value is a lexicon category from lexicon.py.
EMOTE_CATEGORIES = {
    # laughter
    "LUL": "laugh", "LULW": "laugh", "KEKW": "laugh", "OMEGALUL": "laugh",
    "ICANT": "laugh", "KEK": "laugh", "xdd": "laugh", "LuL": "laugh",
    "pepeLaugh": "laugh", "4Head": "laugh", "EleGiggle": "laugh",
    "LOLW": "laugh", "forsenLUL": "laugh", "Kappa": "laugh", "KappaPride": "laugh",
    # positive / hype
    "PogChamp": "posemo", "Pog": "posemo", "POGGERS": "posemo", "PogU": "posemo",
    "POGGIES": "posemo", "PogBones": "posemo", "EZ": "posemo", "Clap": "posemo",
    "peepoHappy": "posemo", "FeelsGoodMan": "posemo", "catJAM": "posemo",
    "widepeepoHappy": "posemo", "<3": "posemo", "VoHiYo": "posemo", "HeyGuys": "greeting",
    # anxiety
    "monkaS": "anx", "monkaW": "anx", "monkaGIGA": "anx", "monkaH": "anx",
    "monkaEyes": "anx", "WutFace": "anx", "DansGame": "negemo",
    # sadness
    "Sadge": "sad", "BibleThump": "sad", "PepeHands": "sad", "FeelsBadMan": "sad",
    "peepoSad": "sad", "widepeepoSad": "sad", "NotLikeThis": "sad", "Sadeg": "sad",
    # negative / disapproval
    "WeirdChamp": "negemo", "Weirdge": "negemo", "ResidentSleeper": "negemo",
    "NotLikeThis_": "negemo", "Madge": "anger", "BabyRage": "anger",
    "SwiftRage": "anger", "pepeMeltdown": "anx", "Clueless": "laugh",
}

# Emotes that, aimed at the streamer, read as mockery/boredom rather than
# shared laughter. Only counted as friction when the message also has a
# friction cue or targets the streamer.
MOCKING_EMOTES = {"OMEGALUL", "LULW", "ResidentSleeper", "WeirdChamp", "Weirdge",
                  "Clueless", "Kappa", "4Head", "DansGame", "Jebaited", "BabyRage",
                  "cmonBruh", "TriHard", "NotLikeThis"}
