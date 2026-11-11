// Smoke test of web/index.html in headless Chromium (playwright 1.48.2), against the record shapes a real
// server sends (fixture from tests/web/make_fixture.py) plus hand-made records for rare states. No backend,
// no network: the websocket and the API are mocked in the browser. Run with tests/web/run.sh.
//   node smoke.js FIXTURE.json WEB_DIR [SHOTS_DIR]
// Prints one line per check and exits 1 when any check fails.
const { chromium } = require("playwright");
const fs = require("fs");
const [FIX, WEB, SHOTS] = process.argv.slice(2);
const fx = JSON.parse(fs.readFileSync(FIX, "utf8"));
const HTML = fs.readFileSync(WEB + "/index.html", "utf8");
let failed = 0;
const check = (name, ok, detail = "") => { console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "  (" + detail + ")" : ""}`); if (!ok) failed++; };
const shot = async (p, n) => { if (SHOTS) await p.screenshot({ path: `${SHOTS}/${n}.png` }); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const T0 = fx.batches[0].find(r => r.type === "msg").ts;
let seq = 0;
const tox = (level, kinds = ["insult"], extra = {}) => ({ level, score: .6, kinds, target: level === "none" ? "" : "streamer", ...extra });
const msg = (ch, uid, display, text, o = {}) => ({ type: "msg", channel: ch, ts: T0 + 3e6 + (seq++) * 1000, id: "h" + seq, user_id: uid,
  display, color: "#1e90ff", text, badfaith: .1, top: [["ordinary", .6], ["reactor", .3], ["provocateur", .1]], flags: [], toxicity: tox("none", []), ...o });
const status = o => ({ type: "status", mode: "live", channels: ["cozy"], connected: true, running: true, session: "s", stopped_at: null,
  stats: { msgs: 1 }, ...o });

async function open(b, w = 1500, h = 950, api = {}, clock = null) {
  const p = await b.newPage({ viewport: { width: w, height: h } });
  p.errs = []; p.posts = [];
  p.on("pageerror", e => p.errs.push("pageerror: " + e.message));
  if (clock !== null) await p.clock.install({ time: clock });         // a fake clock the check moves itself (LF6-1)
  await p.route("http://127.0.0.1:8765/", r => r.fulfill({ status: 200, contentType: "text/html", body: HTML }));
  await p.route(/\/api\//, async r => {
    const u = r.request().url();
    if (r.request().method() === "POST") p.posts.push([u.replace(/.*\/api\//, ""), r.request().postData(), r.request().headers()["x-chatwatch"]]);
    for (const k of Object.keys(api)) if (u.includes(k)) return api[k](r);
    if (u.includes("/api/profile")) {
      const q = new URL(u).searchParams, key = `${q.get("channel")}:${q.get("uid")}`;
      const pr = fx.profiles[key] || Object.values(fx.profiles)[0];
      return r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...pr, channel: q.get("channel"), user_id: q.get("uid") }) });
    }
    if (u.includes("/api/llm_review")) return r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ label: "mild", confidence: "low", rationale: "jabs", rules: [] }) });
    if (u.includes("/api/draft_warning")) return r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ warranted: true, warning: "@x please keep it civil", why: "insult" }) });
    r.fulfill({ status: 404, contentType: "application/json", body: '{"error":"unknown chatter or channel"}' });
  });
  p.refuse = false;                                           // true: the server is down, every reconnect is closed at once
  await p.routeWebSocket(/\/ws$/, ws => { if (p.refuse){ ws.close(); return; } p.sock = ws; });
  await p.goto("http://127.0.0.1:8765/");
  for (let i = 0; i < 50 && !p.sock; i++) { if (clock !== null) await p.clock.runFor(20); await sleep(20); }
  p.feed = async (recs, wait = 120) => { p.sock.send(JSON.stringify(recs)); await sleep(wait); };
  // the server goes away (down: true keeps refusing); without `down` the page's next try reaches a new socket
  p.drop = async (down = false) => { p.refuse = down; const s = p.sock; p.sock = null; await s.close(); await sleep(300); };
  p.back = async () => { p.refuse = false; for (let i = 0; i < 200 && !p.sock; i++) await sleep(20); };
  return p;
}

(async () => {
  const b = await chromium.launch();
  // ---------------------------------------------------------------- the real server's stream
  let p = await open(b);
  for (const batch of fx.batches) await p.feed(batch, 15);
  await sleep(400);
  const rows = await p.$$eval("#rows .line", e => e.length);
  const cards = await p.$$eval("#heads .hu", e => e.length);
  const tabs = await p.$$eval("#tabs .tab", e => e.map(x => x.textContent.trim()));
  check("real records: chat rows, watchlist cards, tabs", rows > 50 && cards > 0 && tabs.length === 1, `rows ${rows}, cards ${cards}, tabs ${tabs.join(" ")}`);
  check("no page errors on the real stream", !p.errs.length, p.errs.join("; "));
  const slur = await p.$$eval("#heads .hu", e => e.map(x => x.textContent).join(" | "));
  check("a slur card reads 'likely'", /likely/i.test(slur));
  check("badge has words for the colours (LF-18)", !!(await p.$("#wlCount [aria-label*='likely']")));
  await shot(p, "real_stream");

  // ---------------------------------------------------------------- LF-1: clicks while chat is moving
  const cur = await p.evaluate("channel");
  let feeding = true, k = 0;
  const filler = (async () => { while (feeding) { await p.feed([msg(cur, "f" + (k % 9), "filler" + (k % 9), "moving chat line " + k++)], 0); await sleep(90); } })();
  let ok = 0; const N = 12;
  for (let i = 0; i < N; i++) {
    await p.evaluate(() => { document.getElementById("drawer").classList.remove("open"); });
    const el = await p.$(`#rows .line:nth-last-child(${3 + (i % 5)}) .who`);
    const bx = await el.boundingBox();
    await p.mouse.move(bx.x + 5, bx.y + 5); await p.mouse.down(); await sleep(150); await p.mouse.up();
    await sleep(250);
    if (await p.$eval("#drawer", d => d.classList.contains("open"))) ok++;
  }
  check("LF-1: presses on chat names while chat moves open the card", ok === N, `${ok} of ${N}`);
  await p.keyboard.press("Escape");
  ok = 0;
  for (let i = 0; i < 6; i++) {
    const el = await p.$(`#heads .hu[data-uid]:nth-child(${1 + (i % 2)})`) || await p.$("#heads .hu[data-uid]");
    const bx = await el.boundingBox();
    await p.mouse.move(bx.x + 20, bx.y + 8); await p.mouse.down(); await sleep(150); await p.mouse.up(); await sleep(250);
    if (await p.$eval("#drawer", d => d.classList.contains("open"))) ok++;
    await p.keyboard.press("Escape"); await sleep(50);
  }
  check("LF-1: presses on watchlist cards open the card", ok === 6, `${ok} of 6`);
  // a selection inside chat survives new batches
  await p.evaluate(() => { const el = document.querySelector("#rows .line:nth-last-child(4) .msg"); const r = document.createRange(); r.selectNodeContents(el); getSelection().removeAllRanges(); getSelection().addRange(r); });
  const sel0 = await p.evaluate(() => getSelection().toString().length);
  await sleep(1200);
  const sel1 = await p.evaluate(() => getSelection().toString().length);
  check("LF-1: a text selection in chat survives new lines", sel0 > 0 && sel1 === sel0, `${sel0} -> ${sel1}`);
  await p.evaluate(() => getSelection().removeAllRanges());

  // ---------------------------------------------------------------- LF-2: scrolled up holds its place, jump bar, older lines
  await p.mouse.move(5, 5);
  await p.$eval("#chat", c => { c.scrollTop = Math.max(0, c.scrollHeight / 3); c.dispatchEvent(new Event("scroll")); });
  await sleep(200);
  const top0 = await p.evaluate(() => { const c = document.getElementById("chat"); const r = c.getBoundingClientRect();
    return document.elementFromPoint(r.left + 60, r.top + 20).closest(".line").dataset.id; });
  await sleep(1500);
  const top1 = await p.evaluate(() => { const c = document.getElementById("chat"); const r = c.getBoundingClientRect();
    return document.elementFromPoint(r.left + 60, r.top + 20).closest(".line").dataset.id; });
  const jump = await p.$eval("#jump", j => ({ hidden: j.hidden, text: j.textContent }));
  check("LF-2: scrolled-up chat holds the reader's line", top0 === top1, `${top0} / ${top1}`);
  check("LF-2: a 'new messages, jump to latest' bar shows", !jump.hidden && /new message/.test(jump.text), jump.text);
  await shot(p, "scrolled_up");
  await p.click("#jump"); await sleep(300);
  check("LF-2: jump to latest follows again", await p.$eval("#chat", c => c.scrollHeight - c.scrollTop - c.clientHeight < 40) && await p.$eval("#jump", j => j.hidden));
  feeding = false; await filler;
  const before = await p.$$eval("#rows .line", e => e.length);
  const older = await p.$("#olderBtn");
  if (older) { await older.click(); await sleep(200); }
  const after = await p.$$eval("#rows .line", e => e.length);
  check("LF-2: older lines can be reached", !!older && after > before, `${before} -> ${after}`);

  // ---------------------------------------------------------------- LF-3, LF-10: the card's tools use the card's channel
  await p.click("#heads .hu[data-uid] .top"); await sleep(300);
  await p.$eval("#drawer", d => { d.scrollTop = 400; });
  await p.click("#warnBtn"); await p.click("#llmBtn"); await sleep(300);
  const tools = p.posts.filter(x => /draft_warning|llm_review/.test(x[0]));
  check("LF-3/LB-10: tools POST the card's channel with X-Chatwatch", tools.length === 2 && tools.every(x => JSON.parse(x[1]).channel === cur && x[2] === "1"),
        JSON.stringify(tools));
  check("LF-3: the card names its channel", new RegExp("#" + cur).test(await p.textContent("#drawer .note")));
  const second = await p.$("#rows .line .who") || await p.$("#heads .hu[data-uid]");
  await second.click(); await sleep(300);
  check("LF-10: a second card opens at its heading", await p.$eval("#drawer", d => d.scrollTop) === 0);
  await p.keyboard.press("Escape"); await sleep(100);
  check("LF-5: the closed card is out of the tab order", await p.$eval("#drawer", d => d.hasAttribute("inert")));
  check("no page errors during interaction", !p.errs.length, p.errs.join("; "));
  await p.close();

  // ---------------------------------------------------------------- LF-5 keyboard
  p = await open(b);
  for (const batch of fx.batches.slice(0, 20)) await p.feed(batch, 5);
  await sleep(300);
  await p.focus("#rows .line .who");
  await p.keyboard.press("Enter"); await sleep(300);
  const kb = await p.evaluate(() => ({ open: document.getElementById("drawer").classList.contains("open"), focus: document.activeElement.className }));
  await p.keyboard.press("Escape"); await sleep(100);
  const back = await p.evaluate(() => document.activeElement.classList.contains("who"));
  check("LF-5: Enter on a name opens the card, focus goes to Close and back", kb.open && kb.focus === "close" && back, JSON.stringify(kb));
  check("LF-5: the input has a label; filters say which is on", !!(await p.$("#chanInput[aria-label]")) && !!(await p.$('#filters button[aria-pressed="true"]')));

  // ---------------------------------------------------------------- LF-8, LF-17: stop and a new start
  await p.feed([status({ running: false, connected: false, channels: [], stopped_at: Date.now() })], 300);
  const stopped = { st: await p.textContent("#status"), strip: await p.textContent("#strip"), tab: !!(await p.$("#tabs .tab.stopped")) };
  check("LF-8/LF-17: Stop shows 'stopped' and frozen tabs", /stopped/.test(stopped.st) && /stopped at/.test(stopped.strip) && stopped.tab, JSON.stringify(stopped));
  await p.feed([status({ running: true, connected: false, channels: ["gamma"] })], 300);
  const t2 = await p.$$eval("#tabs .tab", e => e.map(x => x.dataset.ch));
  check("LF-8: a new Start replaces the old tabs", t2.join() === "gamma", t2.join());
  check("LF-17: running before chat says connecting", /connecting/.test(await p.textContent("#status")));
  await p.feed([status({ running: false, connected: false, channels: [], stopped_at: null, session: null })], 200);
  check("LF-17: idle live mode says idle", /idle/.test(await p.textContent("#status")));
  // ---------------------------------------------------------------- the lossless queue: rate, backlog, lag, Stop now
  const q = o => ({ reads: 2, lines: 10, behind_s: 40, behind_hi_s: 40, catchup_s: 75, aged_out: 0, retrying: null,
    held: false, finishing: false, open: true, ...o });
  // the stats line (owner 2026-11-08): the status first, then msgs/min, the AI buffer, chatters and first-timers this session
  const groom = { type: "room", channel: "gamma", msgs_per_min: 48, chatters: 10, first_timers_this_min: 0, first_timers_session: 7, notices: { items: [], counts: {} } };
  const bufOf = async () => ({ txt: (await p.textContent("#strip")).replace(/\s+/g, " "), title: await p.getAttribute("#strip [data-ai-buffer]", "title"),
    late: !!(await p.$("#strip [data-ai-buffer].late")) });
  await p.feed([groom, status({ running: true, channels: ["gamma"], started_at: Date.now() - 65 * 60000, chat_per_min: { gamma: 48 }, llm_queue: q() })], 300);
  await shot(p, "queue");
  const qs = await bufOf();
  check("stats line: status and run time first, then msgs/min, the AI buffer (catch-up in its tooltip), chatters and first-timers this session",
    /^ ?live for 1 h 5 min 48 ?msgs \/ min ?12 ?AI buffer · 40 s behind · caught up in ~75 s 10 ?chatters this session 7 ?first-timers this session/.test(qs.txt)
    && /caught up in about 75 s/.test(qs.title) && !qs.late && !/#gamma|this min/.test(qs.txt), JSON.stringify(qs));
  await p.feed([status({ running: true, channels: ["gamma"], chat_per_min: { gamma: 48 }, llm_queue: q({ behind_s: 900, behind_hi_s: 720, catchup_s: null, aged_out: 31 }) })], 300);
  await shot(p, "queue_late");
  const ql = await bufOf();
  check("queue: a person read or flagged line over 10 min late is marked; aged-out re-reads are counted", ql.late && /15 min behind · falling behind/.test(ql.txt)
    && /falling behind/.test(ql.title) && /31 low-priority re-reads aged out/.test(ql.title), JSON.stringify(ql));
  // the AI buffer ticking down after a review batch
  await p.feed([groom, status({ running: true, channels: ["gamma"], chat_per_min: { gamma: 48 }, llm_queue: q({ reads: 3, lines: 120, behind_s: 300, held: true }) })], 300);
  const b1 = await p.textContent("#strip");
  await p.feed([status({ running: true, channels: ["gamma"], chat_per_min: { gamma: 48 }, llm_queue: q({ reads: 0, lines: 40, behind_s: 60, catchup_s: 30 }) })], 300);
  const b2 = await p.textContent("#strip");
  await shot(p, "ai_buffer");
  check("the strip shows the AI buffer beside msgs/min; paused during a review, then ticking down", /123\s*AI buffer · 5 min behind · live AI paused/.test(b1)
    && /40\s*AI buffer · 60 s behind · caught up in ~30 s/.test(b2) && /48\s*msgs \/ min/.test(b2), JSON.stringify([b1, b2]));
  await p.feed([status({ running: true, channels: ["gamma"], llm_queue: q({ reads: 0, lines: 0 }) })], 300);
  check("queue: an empty queue says up to date", /0\s*AI buffer · up to date/.test(await p.textContent("#strip")));
  await p.feed([status({ running: true, channels: ["gamma"], llm_queue: q({ retrying: "the local LLM is not answering: refused" }) })], 300);
  const qr = await bufOf();
  check("queue: a call being retried shows and is marked", /AI buffer · 40 s behind · retrying/.test(qr.txt) && qr.late && /not answering/.test(qr.title), JSON.stringify(qr));
  check("the top bar holds no status: CHATWATCH, the tab, then bots / channel / Start on the right",
    !(await p.$("header #status")) && !(await p.$("header .aiSpeed, #aiSpeed")) && !(await p.$("header #sessName"))
    && /^CHATWATCH/.test((await p.textContent("header")).trim()), (await p.textContent("header")).replace(/\s+/g, " "));
  await p.feed([status({ running: false, stopping: true, connected: false, channels: [], llm_queue: q({ finishing: true, reads: 1, lines: 4 }) })], 300);
  await shot(p, "queue_finishing");
  await p.setViewportSize({ width: 390, height: 800 }); await sleep(200);
  await shot(p, "queue_finishing_phone");
  check("stopping/finishing at 390 px: no sideways scroll", await p.evaluate(() => document.documentElement.scrollWidth) <= 390);
  await p.setViewportSize({ width: 1500, height: 950 }); await sleep(100);
  const fin = { st: await p.textContent("#status"), btn: await p.textContent("#startBtn"), dis: await p.$eval("#startBtn", e => e.disabled) };
  await p.click("#startBtn"); await sleep(200);
  const stopNow = p.posts.find(x => x[0] === "stop");
  check("queue: Stop finishing the queue says so, and its button is Stop now", /finishing 5 AI reads/.test(fin.st) && fin.btn === "Stop now" && !fin.dis
    && !!stopNow && /"now":true/.test(stopNow[1] || ""), JSON.stringify({ ...fin, stopNow }));
  await p.close();

  // ---------------------------------------------------------------- LF-6, LF-7, LF-9, LF-12, LF-13, LF-14, LF-15, LF-22, LB-3 banner, XSS
  p = await open(b);
  const X = `<img src=x onerror="window.__xss=(window.__xss||0)+1">`;
  const card = (uid, display, level, o = {}) => ({ user_id: uid, display, level, why: ["slur or dehumanising language"], n: 4, severity: 3, toxic: 1,
    last_seen: T0, since: T0 + 3e6, quote: "q", first: false, review: null, role: null, ...o });
  const many = Array.from({ length: 40 }, (_, i) => card("c" + i, "Card" + i, i < 10 ? "act" : "watch"));
  const room = { type: "room", channel: "cozy", msgs_per_min: 40, chatters: 100, first_timers_this_min: 1,
    raid: { from: "rf" + X, viewers: "5" + X, ts: T0 }, notices: { items: many, counts: { act: 10, watch: 37 }, more: 7 }, notices_earlier: [] };
  const recs = [status({ stats: { msgs: 1, save_errors: 0, llm_person_errors: 0, llm_person_reads: 3 } }), room,
    msg("cozy", "u1", "Name" + X, "text " + X + " ‮unterminated", { top: [["newkind", .7]], flags: ["first"], toxicity: tox("low", [], { source: "llm", layer: {} }) })];
  for (let i = 0; i < 6; i++) recs.push({ type: "alert", kind: "coordinated", channel: "cozy", ts: Date.now() - 1000 + i, text: "tmpl", accounts: 4 + i });
  recs.push({ type: "alert", kind: "pileon", channel: "cozy", ts: Date.now() - 5000, first_timer_share: .3, first_timer_msgs: 12, hostile_first_msgs: 4, window_min: 10 });
  recs.push({ type: "alert", kind: "wave", channel: "cozy", ts: Date.now() - 4000, new_accounts: 7, declared_raid: false });
  recs.push({ type: "alert", kind: "mystery", channel: "cozy", ts: Date.now() - 3000, text: "?" });
  await p.feed(recs, 300);
  await p.feed([status({ stats: { msgs: 2, save_errors: 1, llm_person_errors: 2, llm_person_reads: 3 } })], 400);
  const ban = await p.$$eval("#banners .banner", e => e.map(x => x.textContent.replace(/\s+/g, " ").trim()));
  check("LF-6: one banner per kind, with a count", ban.filter(t => /coordinated/.test(t)).length === 1 && ban.some(t => /×6/.test(t)), ban.join(" | "));
  check("LF-12: no instruction in the pile-on banner; the wave is hedged", !ban.some(t => /Shield Mode|may help/.test(t)) && ban.some(t => /Possible wave/.test(t)));
  check("LF-22: an unknown room warning is labelled as such", ban.some(t => /Possible room warning \(mystery\)/.test(t)));
  check("LB-3: a banner says the LLM reads are failing", ban.some(t => /LLM reads failing/.test(t)));
  check("LF-7: a rising error counter shows in the header", /state not saved/.test(await p.textContent("#errs")));
  check("LF-14: an unknown archetype does not stop the page", !p.errs.length && (await p.$$eval("#rows .line", e => e.length)) >= 1, p.errs.join("; "));
  check("LF-15/XSS: nothing injected", !(await p.evaluate(() => window.__xss)));
  check("LF-22: message text is bidi-isolated", !!(await p.$("#rows .line .msg bdi")));
  check("LF-22: a re-read with no kind has a clean chip", !/·\s+→|·\s+·/.test(await p.textContent("#rows .line .tc")), await p.textContent("#rows .line .tc"));
  await p.click("#showAllBtn"); await sleep(200);
  const more = { n: await p.$$eval("#heads .hu", e => e.length), btn: await p.textContent("#showAllBtn"), note: await p.textContent("#headsMore") };
  check("LF-9: expanded list says 'Show fewer' and how many are not listed", more.n === 40 && /Show fewer/.test(more.btn) && /7 more not listed/.test(more.note), JSON.stringify(more));
  await p.feed([{ ...room, raid: null, notices: { items: [card("g1", "G1", "act", { group: [{ user_id: "g1", display: "G1", level: "act", why: ["slur or dehumanising language"] },
    { user_id: "g2", display: "G2", level: "watch", why: ["LLM read: bad faith"] }] })], counts: { act: 1, watch: 1 }, more: 0 } },
    msg("cozy", "g2", "G2", "you are a nigger lol")], 300);
  const gtip = await p.$eval('#rows .line[data-u="g2"] .lvl', e => e.title);
  check("LF-13: a group member's tag gives their own reason", /LLM read: bad faith/.test(gtip), gtip);
  await shot(p, "banners");
  await p.close();

  // ---------------------------------------------------------------- LF-4: phone width
  p = await open(b, 390, 800);
  for (const batch of fx.batches.slice(0, 25)) await p.feed(batch, 5);
  await p.feed([msg("cozy", "long", "Long", "https://example.com/" + "a".repeat(300) + " " + "W".repeat(120))], 300);
  const docW = await p.evaluate(() => document.documentElement.scrollWidth);
  check("LF-4: no sideways scroll at 390 px", docW <= 390, `scrollWidth ${docW}`);
  await shot(p, "phone");
  await p.close();

  // ---------------------------------------------------------------- LF-16: drawer errors and races
  p = await open(b, 1500, 950, { "/api/profile": r => r.abort() });
  for (const batch of fx.batches.slice(0, 10)) await p.feed(batch, 5);
  await sleep(200);
  await p.click("#rows .line .who"); await sleep(300);
  check("LF-16: a dead server says so in the card", /not reachable/.test(await p.textContent("#drawer")) && !p.errs.length, p.errs.join("; "));
  await p.close();

  // ---------------------------------------------------------------- reclassify a message (the owner's correction)
  p = await open(b, 1500, 950, { "/api/override": r => {
    const q = JSON.parse(r.request().postData() || "{}");
    const tox = q.level === null ? { level: "medium", score: 0.6, kinds: ["insult"], source: "layer" }
      : { level: q.level, score: 0.5, kinds: q.kind ? [q.kind] : [], source: "owner", auto: { level: "medium", kinds: ["insult"], source: "layer" } };
    return r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ type: "msg_update", channel: q.channel, user_id: q.uid, id: q.id, toxicity: tox }) });
  } });
  for (const batch of fx.batches.slice(0, 15)) await p.feed(batch, 5);
  await sleep(200);
  const rowId = await p.$eval("#rows .line[data-id]:last-child", e => e.dataset.id);
  const rowCh = await p.evaluate("channel");
  await p.click(`#rows .line[data-id="${rowId}"] .rc`); await sleep(150);
  check("reclassify: ⋯ opens the menu with its choices, focus on the first", await p.$$eval("#rcMenu [data-rcopt]", e => e.length) === 7
    && await p.evaluate(() => document.activeElement && document.activeElement.dataset.rcopt === "none"));
  await p.keyboard.press("Escape"); await sleep(100);
  check("reclassify: Escape closes the menu and returns focus to ⋯", !(await p.$("#rcMenu"))
    && await p.evaluate(() => document.activeElement && document.activeElement.classList.contains("rc")));
  await p.click(`#rows .line[data-id="${rowId}"] .rc`); await sleep(150);
  await shot(p, "reclassify_menu");
  await p.click('#rcMenu [data-rcopt="none"]'); await sleep(300);
  const post = p.posts.find(x => x[0] === "override");
  const body = post ? JSON.parse(post[1]) : {};
  check("reclassify: the POST names the row's channel, chatter and message, with the header", !!post && body.id === rowId
    && body.channel === rowCh && body.level === "none" && post[2] === "1", post ? post[1] : "no POST");
  check("reclassify: the row shows the correction as set by you", /not toxic · set by you/.test(await p.textContent(`#rows .line[data-id="${rowId}"]`))
    && !(await p.$("#rcMenu")));
  await p.click(`#rows .line[data-id="${rowId}"] .rc`); await sleep(150);
  check("reclassify: a corrected message offers 'Back to automatic'", !!(await p.$('#rcMenu [data-rcopt="auto"]')));
  await p.click('#rcMenu [data-rcopt="auto"]'); await sleep(300);
  check("reclassify: back to automatic shows the tool's read again", !/set by you/.test(await p.textContent(`#rows .line[data-id="${rowId}"]`)));
  await p.click(`#rows .line[data-id="${rowId}"] .rc`); await sleep(150);
  await p.mouse.click(5, 5); await sleep(100);
  check("reclassify: a click elsewhere closes the menu", !(await p.$("#rcMenu")) && !p.errs.length, p.errs.join("; "));
  await shot(p, "reclassify");
  await p.close();

  // ---------------------------------------------------------------- audit 2026-10-29b (live_frontend.md N-n)
  // a mock server for /api/override: answers like the real one (owner toxicity, "unchanged" on a repeat), after `delay` ms
  const ovr = (o = {}) => async r => {
    const q = JSON.parse(r.request().postData() || "{}");
    if (o.delay) await sleep(o.delay);
    if (o.fail) return r.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify(o.fail) });
    const tox = q.level === null ? { level: "medium", score: 0.6, kinds: ["insult"], source: "layer" }
      : { level: q.level, score: 0.5, kinds: q.kind ? [q.kind] : [], source: "owner", auto: { level: "medium", kinds: ["insult"], source: "layer" } };
    return r.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ type: "msg_update", channel: q.channel, user_id: q.uid, id: q.id, toxicity: tox, recorded: true }) });
  };
  const lineRec = (ch, i, T) => ({ type: "msg", channel: ch, ts: T + i * 100, id: "bg" + i, user_id: "u" + (i % 900), display: "chatter" + (i % 900),
    color: "#1e90ff", text: "background line " + i, flags: [], toxicity: { level: "none", kinds: [] }, top: [["ordinary", .7]], badfaith: .05 });

  // N-1: a background tab (no animation frames) while 6,000 lines arrive
  p = await open(b);
  for (const batch of fx.batches.slice(0, 15)) await p.feed(batch, 5);
  await sleep(200);
  let ch1 = await p.evaluate("channel");
  await p.evaluate(() => { window.__raf = window.requestAnimationFrame; window.__held = []; window.requestAnimationFrame = cb => { __held.push(cb); return 0; }; });
  const TB = Date.now();
  for (let k = 0; k < 6000; k += 500) await p.feed(Array.from({ length: 500 }, (_, i) => lineRec(ch1, k + i, TB)), 0);
  await sleep(500);
  const bg = await p.evaluate(() => ({ pending: pending.length, overflow: typeof overflow === "undefined" ? null : overflow }));
  await p.evaluate(() => { window.requestAnimationFrame = window.__raf; for (const cb of __held.splice(0)) cb(performance.now()); });
  await sleep(400);
  const bgBack = await p.evaluate(() => ({ rows: document.querySelectorAll("#rows .line").length, shown: shown.length,
    last: document.querySelector("#rows .line:last-child").dataset.id }));
  check("N-1: a background tab's queue stays capped and the chat rebuilds on return", bg.pending <= 300 && bg.overflow
    && bgBack.rows <= 300 && bgBack.shown === bgBack.rows && bgBack.last === "bg5999" && !p.errs.length, JSON.stringify({ bg, bgBack, errs: p.errs }));
  await p.close();

  // N-2, N-8: the menu holds the chat, quotes its message, closes on scroll; keyboard; N-4 owner chips; N-12
  p = await open(b, 1500, 950, { "/api/override": ovr() });
  for (const batch of fx.batches.slice(0, 15)) await p.feed(batch, 5);
  await sleep(200);
  ch1 = await p.evaluate("channel");
  const opener = await p.$eval("#rows .line[data-id]:last-child", e => ({ id: e.dataset.id, text: e.querySelector(".msg > bdi").textContent }));
  await p.click(`#rows .line[data-id="${opener.id}"] .rc`); await sleep(150);
  const y0 = await p.$eval(`#rows .line[data-id="${opener.id}"]`, e => e.getBoundingClientRect().top);
  const n0 = await p.$$eval("#rows .line", e => e.length);
  for (let i = 0; i < 25; i++) await p.feed([msg(ch1, "f" + (i % 5), "filler" + (i % 5), "fast chat line " + i)], 40);
  await sleep(300);
  const held = { y: await p.$eval(`#rows .line[data-id="${opener.id}"]`, e => e.getBoundingClientRect().top), n: await p.$$eval("#rows .line", e => e.length),
    quote: await p.textContent("#rcMenu .q"), expanded: await p.$eval(`#rows .line[data-id="${opener.id}"] .rc`, e => e.getAttribute("aria-expanded")) };
  check("N-2: an open menu holds the chat under it and quotes its message", held.y === y0 && held.n === n0
    && held.quote.includes(opener.text.slice(0, 40)) && held.expanded === "true", JSON.stringify({ y0, n0, held }));
  await p.keyboard.press("ArrowDown"); await p.keyboard.press("ArrowDown");
  const kb2 = await p.evaluate(() => document.activeElement.dataset.rcopt);
  await p.keyboard.press("End");
  const kbEnd = await p.evaluate(() => document.activeElement.dataset.rcopt);
  check("N-8: arrow keys and End move through the choices; they are radio items", kb2 === "medium" && kbEnd === "high"
    && await p.$$eval('#rcMenu [role="menuitemradio"]', e => e.length) === 7);
  await p.keyboard.press("Tab"); await sleep(100);
  check("N-8: Tab out of the menu closes it", !(await p.$("#rcMenu")));
  await p.click(`#rows .line[data-id="${opener.id}"] .rc`); await sleep(150);
  await p.$eval("#chat", c => { c.scrollTop = 0; }); await sleep(150);
  check("N-2: scrolling the chat closes the menu", !(await p.$("#rcMenu")));
  await p.$eval("#chat", c => { c.scrollTop = c.scrollHeight; }); await sleep(400);
  const lastTxt = await p.$eval("#rows .line:last-child", e => e.textContent);
  check("N-2: the chat moves again once the menu is closed", /fast chat line 24/.test(lastTxt), lastTxt.replace(/\s+/g, " ").trim());
  // N-4: a card made by the owner's correction says so, on the card and on the chat tag
  await p.mouse.move(5, 5);                                    // not aiming at the chat (LF-1 holds redraws then)
  const roomOwn = { type: "room", channel: ch1, msgs_per_min: 4, chatters: 10, first_timers_this_min: 0, raid: null, notices_earlier: [],
    notices: { items: [{ user_id: "own1", display: "Own1", level: "act", why: ["slur or dehumanising language"], n: 1, severity: 3, toxic: 1, last_seen: T0,
      since: T0, quote: "politics out of this chat", first: false, review: null, role: null, owner: true, quote_owner: true }],
      counts: { act: "1", watch: "<img src=x onerror=\"window.__xss=1\">" }, more: "2<img src=x onerror=\"window.__xss=1\">" } };
  await p.feed([roomOwn, msg(ch1, "own1", "Own1", "lets keep politics out of this chat please",
    { toxicity: { level: "high", kinds: ["hate"], source: "owner", auto: { level: "none", kinds: [] } } })], 1500);
  const ownCard = await p.$eval('#heads .hu[data-uid="own1"]', e => e.textContent);
  const ownTag = await p.$eval('#rows .line[data-u="own1"] .lvl', e => e.title);
  check("N-4: a card made by the owner's correction says 'set by you', and so does its chat tag", /set by you/.test(ownCard) && /your own correction/.test(ownTag),
    `${ownCard.replace(/\s+/g, " ").trim()} | ${ownTag}`);
  check("N-12: server numbers in the counts are not HTML", !(await p.evaluate(() => window.__xss)) && /1/.test(await p.textContent("#wlCount")));
  await p.close();

  // N-5: one POST at a time, with a "Saving…" note; N-3: a failed register write shows in the menu and as a pill
  p = await open(b, 1500, 950, { "/api/override": ovr({ delay: 600 }) });
  for (const batch of fx.batches.slice(0, 15)) await p.feed(batch, 5);
  await sleep(200);
  let rid = await p.$eval("#rows .line[data-id]:last-child", e => e.dataset.id);
  await p.click(`#rows .line[data-id="${rid}"] .rc`); await sleep(150);
  await p.click('#rcMenu [data-rcopt="low"]');
  const saving = await p.textContent("#rcErr");
  await p.evaluate(() => { for (const b of document.querySelectorAll('#rcMenu [data-rcopt]')) { b.disabled = false; } });   // a click that slips through
  await p.click('#rcMenu [data-rcopt="medium"]', { force: true }).catch(() => {});
  await sleep(900);
  check("N-5: a correction in flight blocks a second one and says 'Saving…'", p.posts.filter(x => x[0] === "override").length === 1 && /Saving/.test(saving),
    `${p.posts.filter(x => x[0] === "override").length} POSTs, "${saving}"`);
  await p.close();
  p = await open(b, 1500, 950, { "/api/override": ovr({ fail: { error: "not applied: the correction could not be recorded in data/sessions/owner_overrides.jsonl", not_recorded: true } }) });
  for (const batch of fx.batches.slice(0, 15)) await p.feed(batch, 5);
  await sleep(200);
  rid = await p.$eval("#rows .line[data-id]:last-child", e => e.dataset.id);
  await p.click(`#rows .line[data-id="${rid}"] .rc`); await sleep(150);
  await p.click('#rcMenu [data-rcopt="high"][data-rck="hate"]'); await sleep(300);
  const errTxt = await p.textContent("#rcErr");
  const stillAuto = !/set by you/.test(await p.textContent(`#rows .line[data-id="${rid}"]`));
  await p.keyboard.press("Escape");
  await p.feed([status({ stats: { msgs: 1, override_log_errors: 0 } })], 100);
  await p.feed([status({ stats: { msgs: 2, override_log_errors: 1 } })], 400);
  check("N-3: a correction the server could not record says so in the menu and in a header pill", /not applied/.test(errTxt) && stillAuto
    && /corrections not recorded/.test(await p.textContent("#errs")), `${errTxt} | ${await p.textContent("#errs")}`);
  await p.close();

  // N-6, N-7, N-13: a correction from the open card keeps its place and its ⋯; the server's echo is no news; the card is modal
  p = await open(b, 1500, 450, { "/api/override": ovr() });
  for (const batch of fx.batches.slice(0, 20)) await p.feed(batch, 5);
  await sleep(200);
  await p.click("#heads .hu[data-uid] .top"); await sleep(400);
  check("N-13: the open card is a modal dialog", await p.$eval("#drawer", d => d.getAttribute("aria-modal")) === "true");
  const cardRc = await p.$$eval("#drawer section .rmsg .rc", e => e.map(x => x.dataset.rc));
  const low = cardRc[cardRc.length - 1];
  await p.$eval(`#drawer section .rc[data-rc="${low}"]`, e => e.scrollIntoView());
  const dTop0 = await p.$eval("#drawer", d => d.scrollTop);
  await p.click(`#drawer section .rc[data-rc="${low}"]`); await sleep(150);
  await p.click('#rcMenu [data-rcopt="none"]'); await sleep(500);
  const cu = await p.evaluate("card");
  const echo = { type: "msg_update", channel: cu.ch, user_id: cu.uid, id: low,
    toxicity: { level: "none", score: 0.5, kinds: [], source: "owner", auto: { level: "medium", kinds: ["insult"], source: "layer" } } };
  await p.feed([echo], 300);
  const after7 = { top: await p.$eval("#drawer", d => d.scrollTop), focus: await p.evaluate(() => document.activeElement.dataset.rc), news: !!(await p.$("#dUpd")) };
  check("N-7: a correction from the card keeps its place and focus on its ⋯", dTop0 > 0 && Math.abs(after7.top - dTop0) < 30 && after7.focus === low, JSON.stringify({ dTop0, after7 }));
  check("N-6: the server's copy of this page's own correction is no 'new information'", !after7.news);
  await p.feed([{ ...echo, toxicity: { level: "high", score: .85, kinds: ["harassment"], source: "llm", layer: {} } }], 300);
  check("N-6: a later update for the chatter still offers a reload", !!(await p.$("#dUpd")));
  await p.focus("#drawer .close");
  await p.keyboard.press("Shift+Tab");
  check("N-13: Shift+Tab from the first control wraps inside the card", await p.evaluate(() => document.getElementById("drawer").contains(document.activeElement)));
  await p.close();

  // N-10: another page's correction updates an open menu; under the Flagged filter a cleared line leaves
  p = await open(b, 1500, 950, { "/api/override": ovr() });
  for (const batch of fx.batches.slice(0, 15)) await p.feed(batch, 5);
  await sleep(200);
  ch1 = await p.evaluate("channel");
  rid = await p.$eval("#rows .line[data-id]:last-child", e => e.dataset.id);
  const ruid = await p.$eval(`#rows .line[data-id="${rid}"]`, e => e.dataset.u);
  await p.click(`#rows .line[data-id="${rid}"] .rc`); await sleep(150);
  await p.feed([{ type: "msg_update", channel: ch1, user_id: ruid, id: rid,
    toxicity: { level: "high", kinds: ["threat"], source: "owner", auto: { level: "none", kinds: [] } } }], 300);
  check("N-10: an open menu follows another page's correction", !!(await p.$('#rcMenu [data-rcopt="auto"]'))
    && await p.$eval('#rcMenu [data-rck="threat"]', e => e.getAttribute("aria-checked")) === "true");
  await p.keyboard.press("Escape");
  // polish 2026-11-10: Flagged holds every toxic line (the Toxic tab is gone); a line flagged only for its toxicity
  check("polish: the chat filters are All, First-timers and Flagged (no Toxic tab)",
    (await p.$$eval("#filters button", e => e.map(x => x.dataset.f))).join() === "all,first,flagged");
  await p.click('#filters [data-f="flagged"]'); await sleep(300);
  const tid = await p.evaluate(() => { const NM = noticeMap(channel), ls = chan(channel).lines;
    const el = [...document.querySelectorAll("#rows .line[data-id]")].reverse().find(e => { const m = ls.find(x => String(x.id) === e.dataset.id);
      return m && isTox(m.toxicity) && !NM[m.user_id] && !(m.flags || []).some(f => FLAGGED.includes(f)); });
    return el ? el.dataset.id : null; });
  if (tid) {
    const tuid = await p.$eval(`#rows .line[data-id="${tid}"]`, e => e.dataset.u);
    await p.feed([{ type: "msg_update", channel: ch1, user_id: tuid, id: tid, toxicity: { level: "none", kinds: [], source: "owner", auto: { level: "medium", kinds: [] } } }], 300);
  }
  // audit polish F6: a toxic line that also carries a warning chip stays under Flagged when it is reclassified 'Not toxic'
  await p.feed([msg(ch1, "twoway", "twoway", "a line with two reasons", { id: "two1", toxicity: tox("medium"), flags: ["template"] })], 300);
  await p.feed([{ type: "msg_update", channel: ch1, user_id: "twoway", id: "two1", toxicity: { level: "none", kinds: [], source: "owner", auto: { level: "medium", kinds: [] } } }], 300);
  check("audit polish F6: under Flagged a line flagged for two reasons stays when one goes", !!(await p.$('#rows .line[data-id="two1"]')));
  check("N-10: under the Flagged filter a line flagged only as toxic, reclassified 'Not toxic', leaves the list", !!tid && !(await p.$(`#rows .line[data-id="${tid}"]`)) && !p.errs.length, p.errs.join("; "));
  await p.close();


  // test gaps: a dead server while correcting; data-attribute XSS through id, uid and channel round-trips inert
  p = await open(b, 1500, 950, { "/api/override": r => r.abort() });
  for (const batch of fx.batches.slice(0, 10)) await p.feed(batch, 5);
  await sleep(200);
  ch1 = await p.evaluate("channel");
  const XA = `x"><img src=x onerror="window.__xss=1">'`;
  await p.feed([msg(ch1, XA + "u", "Xss", "plain line", { id: XA + "id" })], 1500);
  await p.click(`#rows .line:last-child .rc`); await sleep(150);
  await p.click('#rcMenu [data-rcopt="low"]'); await sleep(400);
  const dead = await p.textContent("#rcErr");
  const xpost = p.posts.find(x => x[0] === "override");
  const xb = xpost ? JSON.parse(xpost[1]) : {};
  check("test gap: a dead server while correcting says so in the menu", /not reachable/.test(dead), dead);
  check("test gap: id and uid with HTML in them round-trip exactly and stay inert", xb.id === XA + "id" && xb.uid === XA + "u" && xb.channel === ch1
    && !(await p.evaluate(() => window.__xss)) && !p.errs.length, JSON.stringify(xb));
  await p.close();

  // N-9 at touch size; LB2-12/N-11 the menu says a replay's correction is not recorded
  p = await b.newPage({ viewport: { width: 390, height: 800 }, hasTouch: true, isMobile: true });
  p.errs = []; p.on("pageerror", e => p.errs.push(e.message));
  await p.route("http://127.0.0.1:8765/", r => r.fulfill({ status: 200, contentType: "text/html", body: HTML }));
  await p.routeWebSocket(/\/ws$/, ws => { p.sock = ws; });
  await p.goto("http://127.0.0.1:8765/");
  for (let i = 0; i < 50 && !p.sock; i++) await sleep(20);
  for (const batch of fx.batches.slice(0, 10)) { p.sock.send(JSON.stringify(batch)); await sleep(5); }
  p.sock.send(JSON.stringify([status({ mode: "replay", running: undefined, channels: ["cozy"] })]));      // after the fixture's live status
  await sleep(300);
  const rcBox = await p.$eval("#rows .line .rc", e => { const r = e.getBoundingClientRect(); return { w: r.width, h: r.height, o: getComputedStyle(e).opacity }; });
  check("N-9: the ⋯ is a 24 px target at full strength on a touch screen", rcBox.h >= 24 && rcBox.w >= 24 && Number(rcBox.o) >= 0.9, JSON.stringify(rcBox));
  await p.tap("#rows .line .rc"); await sleep(150);
  check("LB2-12/N-11: on a replay the menu says the correction is not recorded", /not recorded/.test(await p.textContent("#rcMenu")));
  await shot(p, "phone_menu");
  await p.close();

  // ---------------------------------------------------------------- the owner features of 2026-11-01
  // the messages behind a listing (a watchlist card's reason opens them; an insult older than the 40 recent messages is
  // corrected from there) and the owner's bot role. The mock server keeps state, as the real one does.
  const OLD = fx.uids.oldtimer, BOT = fx.uids.gamebot;
  const evState = JSON.parse(JSON.stringify(fx.evidence)), roleState = {};
  const json = (r, body, status = 200) => r.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
  const ownerApi = {
    "/api/evidence": r => { const q = new URL(r.request().url()).searchParams, d = evState[`${q.get("channel")}:${q.get("uid")}`];
      return d ? json(r, d) : json(r, { error: "unknown chatter or channel" }, 404); },
    "/api/override": r => { const q = JSON.parse(r.request().postData() || "{}");
      const t = q.level === null ? { level: "medium", score: 0.6, kinds: ["insult"], source: "layer" }
        : { level: q.level, score: 0.5, kinds: q.kind ? [q.kind] : [], source: "owner", auto: { level: "medium", kinds: ["insult"], source: "layer" } };
      for (const d of Object.values(evState)) for (const it of d.items) if (String(it.id) === String(q.id)) { it.toxicity = t; it.counted = it.counted && q.level !== "none"; }
      return json(r, { type: "msg_update", channel: q.channel, user_id: q.uid, id: q.id, toxicity: t, recorded: true }); },
    "/api/roles": r => json(r, { roles: Object.entries(roleState).map(([user_id, e]) => ({ user_id, ...e })), saves: true }),
    "/api/role": r => { const q = JSON.parse(r.request().postData() || "{}");
      if (q.role) roleState[q.uid] = { role: q.role, login: "gamebot", display: "gamebot", set: "2026-11-01T01:30:00Z" }; else delete roleState[q.uid];
      return json(r, { user_id: q.uid, role: q.role, entry: roleState[q.uid] || null, recorded: true }); },
    "/api/profile": r => { const q = new URL(r.request().url()).searchParams, pr = fx.profiles[`${q.get("channel")}:${q.get("uid")}`] || Object.values(fx.profiles)[0];
      return json(r, { ...pr, channel: q.get("channel"), user_id: q.get("uid"), owner_role: roleState[q.get("uid")] || null }); },
  };
  p = await open(b, 1500, 950, ownerApi);
  for (const batch of fx.batches) await p.feed(batch, 5);
  await sleep(400);
  await p.click('#tabs .tab[data-ch="cozy"]'); await sleep(300);           // the owner-feature chatters are in cozy
  await p.mouse.move(5, 5);
  const oldCard = `#heads .hu[data-uid="${OLD}"]`;
  check("evidence: the chatter with an older insult is on the watchlist", !!(await p.$(oldCard)));
  await p.click(`${oldCard} [data-ev]`); await sleep(400);
  const evRows = await p.$$eval(`${oldCard} .evl .rmsg`, e => e.map(x => x.textContent.replace(/\s+/g, " ").trim()));
  // FLAGS_PERSIST (owner, 2026-11-07): a kept line counts at any age, so no row is tagged until a review or the owner clears it
  check("evidence: the reason opens the messages behind it, not the user card, older ones counted (untagged)",
    evRows.length >= 2 && evRows.every(t => !/older · not counted|not toxic now/.test(t))
    && /Untagged messages count now/.test(await p.textContent(`${oldCard} .evl`))
    && !(await p.$eval("#drawer", d => d.classList.contains("open"))) && await p.$eval(`${oldCard} [data-ev]`, e => e.getAttribute("aria-expanded")) === "true",
    evRows.join(" | "));
  await shot(p, "evidence_dropdown");
  const olderItem = evState[`cozy:${OLD}`].items.find(it => it.older);
  await p.click(`${oldCard} .evl .rc[data-rc="${olderItem.id}"]`); await sleep(150);
  check("evidence: an older message's ⋯ opens the reclassify menu", await p.$$eval("#rcMenu [data-rcopt]", e => e.length) === 7);
  await p.click('#rcMenu [data-rcopt="none"]'); await sleep(500);
  const opost = p.posts.filter(x => x[0] === "override").map(x => JSON.parse(x[1]));
  const fixed = await p.$eval(`${oldCard} .evl .rc[data-rc="${olderItem.id}"]`, e => e.closest(".rmsg").textContent.replace(/\s+/g, " ")).catch(() => "");
  check("evidence: correcting a message older than the 40 posts its id and the list shows it as set by you, still open",
    opost.length === 1 && opost[0].id === olderItem.id && opost[0].uid === OLD && opost[0].level === "none"
    && /not toxic · set by you/.test(fixed) && /not toxic now/.test(fixed)
    && await p.evaluate(id => document.activeElement && document.activeElement.dataset.rc === id, String(olderItem.id)), JSON.stringify(opost) + " | " + fixed);
  await shot(p, "evidence_corrected");
  await p.focus(`${oldCard} [data-ev]`); await p.keyboard.press("Enter"); await sleep(200);
  check("evidence: Enter on the reason folds the list again", !(await p.$(`${oldCard} .evl`)));
  // the user card lists them too, folded
  await p.click(`${oldCard} .top`); await sleep(400);
  const evd = await p.$eval("#drawer details.evd summary", e => e.textContent).catch(() => "");
  check("evidence: the user card lists the messages behind the listing", /Messages behind this \(4\)/.test(evd), evd);
  await p.click("#drawer details.evd summary"); await sleep(150);
  await shot(p, "card_evidence");
  await p.keyboard.press("Escape"); await sleep(100);

  // the bot role: a bot's line is tagged in chat; set and unset on the card; the header lists every role set
  const botLine = await p.$eval("#rows .line.bot", e => e.textContent.replace(/\s+/g, " ").trim()).catch(() => "");
  check("roles: a line from a chatter with the owner's bot role shows, tagged, without a pip colour or ⋯", /bot · set by you/.test(botLine)
    && !(await p.$("#rows .line.bot .rc")), botLine);
  await p.click("#rows .line.bot .who"); await sleep(400);
  const markBtn = await p.$('#drawer [data-role="bot"]');
  check("roles: the card offers 'Mark as a bot', and no mod role", !!markBtn && !(await p.$('#drawer [data-role="mod"]'))
    && /Not for mods/.test(await p.textContent("#drawer")));
  await markBtn.click(); await sleep(500);
  const rpost = p.posts.filter(x => x[0] === "role").map(x => JSON.parse(x[1]));
  const botCard = await p.textContent("#drawer");
  check("roles: setting it POSTs the card's chatter and the card then shows it with 'Not a bot'", rpost.length === 1 && rpost[0].uid === BOT
    && rpost[0].role === "bot" && rpost[0].channel === "cozy" && /Marked a bot by you/.test(botCard) && !!(await p.$('#drawer [data-role=""]')), JSON.stringify(rpost));
  await p.$eval("#drawer", d => { d.scrollTop = 0; }); await sleep(100);
  await shot(p, "bot_card");
  await p.keyboard.press("Escape"); await sleep(100);
  const rb = await p.$eval("#rolesBtn", e => ({ hidden: e.hidden, text: e.textContent }));
  check("roles: the header shows how many bots you set", !rb.hidden && /bots you set: 1/.test(rb.text), JSON.stringify(rb));
  await p.click("#rolesBtn"); await sleep(400);
  check("roles: the list names each one with a Remove button", /gamebot/.test(await p.textContent("#drawer")) && !!(await p.$(`#drawer [data-unrole="${BOT}"]`)));
  await shot(p, "roles_list");
  await p.click(`#drawer [data-unrole="${BOT}"]`); await sleep(500);
  const upost = p.posts.filter(x => x[0] === "role").map(x => JSON.parse(x[1]));
  check("roles: Remove unsets it and the list is empty", upost.length === 2 && upost[1].uid === BOT && upost[1].role === null
    && /None\./.test(await p.textContent("#drawer")) && await p.$eval("#rolesBtn", e => e.hidden), JSON.stringify(upost));
  check("no page errors with the owner features", !p.errs.length, p.errs.join("; "));
  await p.close();
  p = await open(b, 390, 800, ownerApi);                      // the dropdown at phone width
  for (const batch of fx.batches) await p.feed(batch, 5);
  await sleep(300);
  await p.click('#tabs .tab[data-ch="cozy"]'); await sleep(300);
  await p.click(`${oldCard} [data-ev]`); await sleep(400);
  await p.$eval(oldCard, e => e.scrollIntoView());
  const docW2 = await p.evaluate(() => document.documentElement.scrollWidth);
  check("evidence: no sideways scroll at 390 px with the list open", docW2 <= 390 && !!(await p.$(`${oldCard} .evl .rmsg`)), `scrollWidth ${docW2}`);
  await shot(p, "evidence_phone");
  await p.close();

  // ---------------------------------------------------------------- audit 2026-11-01 (live_frontend.md LF3-n and its test gaps)
  const XS = `<img src=x onerror="window.__xss=1">`;
  const S1 = { level: "none", score: 0, kinds: [], target: "", source: "llm", counted_level: "high",
    layer: { level: "high", score: .85, kinds: ["hate"], sure: ["hate"], source: "layer" } };
  const ev2 = JSON.parse(JSON.stringify(fx.evidence)), roles2 = {};
  let evGets = 0, roleDelay = 0, roleFail = false, evFail = false;
  const api2 = {
    "/api/evidence": r => { evGets++; const q = new URL(r.request().url()).searchParams, d = ev2[`${q.get("channel")}:${q.get("uid")}`];
      return evFail ? json(r, { error: "boom" }, 500) : d ? json(r, d) : json(r, { error: "unknown chatter or channel" }, 404); },
    "/api/override": ownerApi["/api/override"],
    "/api/roles": r => json(r, { roles: Object.entries(roles2).map(([user_id, e]) => ({ user_id, ...e })), saves: true }),
    "/api/role": async r => { const q = JSON.parse(r.request().postData() || "{}");
      if (roleDelay) await sleep(roleDelay);
      if (roleFail) return json(r, { error: "not applied: the role could not be saved in data/sessions/owner_roles.json. See the server's console.", not_recorded: true }, 500);
      if (q.role) roles2[q.uid] = { role: q.role, login: "gamebot", display: "gamebot", set: "2026-11-01T01:30:00Z" }; else delete roles2[q.uid];
      return json(r, { user_id: q.uid, role: q.role, entry: roles2[q.uid] || null, recorded: true, rev: Object.keys(roles2).length + 10 }); },
    "/api/profile": r => { const q = new URL(r.request().url()).searchParams, uid = q.get("uid");
      if (uid === "stub1") return json(r, roles2.stub1 ? { user_id: uid, login: "stubby", display: "Stub" + XS, channel: q.get("channel"), stub: true,
        owner_role: roles2.stub1, evidence: [], person_read: null } : { error: "unknown chatter or channel" }, roles2.stub1 ? 200 : 404);
      const pr = fx.profiles[`${q.get("channel")}:${uid}`] || Object.values(fx.profiles)[0];
      return json(r, { ...pr, channel: q.get("channel"), user_id: uid, owner_role: roles2[uid] || null }); },
  };
  p = await open(b, 1500, 950, api2);
  for (const batch of fx.batches) await p.feed(batch, 5);
  await sleep(300);
  await p.click('#tabs .tab[data-ch="cozy"]'); await sleep(300);
  await p.mouse.move(5, 5);

  // LF3-2: a keyword slur the LLM re-read as "none" still counts (S1): chip, pip and the Flagged filter show it
  await p.feed([msg("cozy", "s1u", "slurry", "an S1 line", { id: "s1msg", toxicity: S1 })], 300);
  const s1 = await p.$eval('#rows .line[data-id="s1msg"]', e => ({ chip: e.querySelector(".tc").textContent, pip: e.querySelector(".pip").style.background }));
  check("LF3-2: a keyword slur re-read as none shows the level it counts at in chat", /high · hate · keyword slur/.test(s1.chip) && /--bad/.test(s1.pip), JSON.stringify(s1));
  await p.click('#filters button[data-f="flagged"]'); await sleep(300);
  check("LF3-2: the Flagged filter keeps it", !!(await p.$('#rows .line[data-id="s1msg"]')));
  await p.click('#filters button[data-f="all"]'); await sleep(300);

  // LF3-5: the watchlist card is a group, not a button holding buttons
  const nested = await p.$$eval('#heads [role="button"] [role="button"], #heads [role="button"] button, #heads .hu[role="button"]', e => e.length);
  check("LF3-5: no control inside another; the card is a group with its name as the button", nested === 0
    && await p.$eval(oldCard, e => e.getAttribute("role")) === "group" && !!(await p.$(`${oldCard} .nm[role="button"][tabindex="0"]`)), `nested ${nested}`);

  // LF3-1: the open dropdown follows another page's correction through a re-render, and refreshes while open
  await p.click(`${oldCard} [data-ev]`); await sleep(400);
  const items2 = ev2[`cozy:${OLD}`].items, recentItem = items2.filter(it => it.counted).slice(-1)[0];
  const other = { level: "none", score: 0, kinds: [], target: "", source: "owner", auto: { level: "medium", kinds: ["insult"], source: "layer" } };
  await p.feed([{ type: "msg_update", channel: "cozy", user_id: OLD, id: recentItem.id, toxicity: other }], 200);
  await p.evaluate(u => { const el = document.querySelector(`#heads .hu[data-uid="${u}"]`); if (el) el._html = "stale"; renderWatch(); }, OLD);   // a re-render of that card
  await sleep(100);
  const after1 = await p.$eval(`${oldCard} .evl .rc[data-rc="${recentItem.id}"]`, e => ({ t: e.closest(".rmsg").textContent.replace(/\s+/g, " "), rco: e.dataset.rco })).catch(() => ({}));
  check("LF3-1: after a re-render the dropdown keeps another page's correction (no stale read)", /not toxic · set by you/.test(after1.t || "") && after1.rco === "none:", JSON.stringify(after1));
  items2.push({ id: "fresh1", ts: T0 + 9e6, text: "a brand new insult " + XS, toxicity: tox("medium"), flags: [], counted: true, older: false });
  const g0 = evGets;
  for (let i = 0; i < 4; i++) await p.feed([msg("cozy", OLD, "oldtimer", "line " + i, { toxicity: tox("none", []) })], 150);
  await sleep(2600);
  const fresh = await p.$eval(`${oldCard} .evl`, e => e.textContent).catch(() => "");
  check("LF3-1: an open dropdown refreshes when its chatter posts, at most once per 2 s", /a brand new insult/.test(fresh) && evGets - g0 === 1, `gets ${evGets - g0}`);
  check("test gap: evidence text with HTML stays inert", !(await p.evaluate(() => window.__xss)) && !(await p.$(`${oldCard} .evl img`)));

  // LF3-4: focus on a ⋯ in the dropdown survives a re-render of the card
  await p.focus(`${oldCard} .evl .rc[data-rc="fresh1"]`);
  await p.feed([{ type: "msg_update", channel: "cozy", user_id: OLD, id: items2[0].id, toxicity: tox("low") }], 300);
  const foc = await p.evaluate(() => document.activeElement && document.activeElement.dataset.rc);
  check("LF3-4: focus on a ⋯ inside the dropdown stays there when the card is redrawn", foc === "fresh1", String(foc));

  // LF3-6: Escape folds the dropdown and focus goes back to its reason
  await p.focus(`${oldCard} [data-ev]`); await p.keyboard.press("Escape"); await sleep(200);
  check("LF3-6: Escape folds the open dropdown", !(await p.$(`${oldCard} .evl`))
    && await p.evaluate(u => document.activeElement && document.activeElement.dataset.ev === u, OLD));

  // LF3-2 in the dropdown: an S1-held slur that left the recent messages still counts (FLAGS_PERSIST: no tag), never
  // "not toxic now"
  ev2[`cozy:${OLD}`].items = [{ id: "s1old", ts: T0 + 1e6, text: "an old S1 line", toxicity: S1, flags: [], counted: true, older: true }];
  await p.click(`${oldCard} [data-ev]`); await sleep(400);
  const s1row = await p.$eval(`${oldCard} .evl .rmsg`, e => e.textContent.replace(/\s+/g, " ")).catch(() => "");
  check("LF3-2: in the dropdown an S1-held slur that left the recent messages still counts (untagged), never 'not toxic now'", !/older · not counted/.test(s1row) && !/not toxic now/.test(s1row) && /keyword slur/.test(s1row), s1row);
  await p.click(`${oldCard} [data-ev]`); await sleep(200);

  // LF3-7: a hundred messages stay in a box, not a 10,000 px card
  ev2[`cozy:${OLD}`].items = Array.from({ length: 100 }, (_, i) => ({ id: "many" + i, ts: T0 + i * 1000, text: "insult number " + i, toxicity: tox("medium"), flags: [], counted: i > 96, older: i <= 96 }));
  await p.click(`${oldCard} [data-ev]`); await sleep(500);
  const hgt = await p.$eval(oldCard, e => e.getBoundingClientRect().height);
  check("LF3-7: a dropdown with 100 messages keeps the card under about half the screen", hgt < 950 * 0.5 + 160 && await p.$$eval(`${oldCard} .evl .rmsg`, e => e.length) === 100, `${Math.round(hgt)} px`);
  await shot(p, "evidence_many");
  await p.click(`${oldCard} [data-ev]`); await sleep(200);

  // test gap: /api/evidence fails
  evFail = true;
  await p.click(`${oldCard} [data-ev]`); await sleep(400);
  check("test gap: a failing evidence request says so in the dropdown", /Couldn't load them: boom/.test(await p.textContent(`${oldCard} .evl`).catch(() => "")));
  await p.click(`${oldCard} [data-ev]`); await sleep(200);
  evFail = false;

  // LF3-8: the roles pill is not a channel tab
  check("LF3-8: 'bots you set' has its own style, not a channel tab's", await p.$eval("#rolesBtn", e => e.className) === "rolesPill");

  // roles: double click, 500 and its pill, focus after a change (LF3-4)
  await p.click("#rows .line.bot .who"); await sleep(400);
  roleDelay = 300;
  const r0 = p.posts.filter(x => x[0] === "role").length;
  await p.click('#drawer [data-role="bot"]'); await p.click('#drawer [data-role="bot"]', { force: true }).catch(() => {});
  await sleep(800);
  roleDelay = 0;
  check("test gap: a double click on 'Mark as a bot' sends one POST", p.posts.filter(x => x[0] === "role").length - r0 === 1);
  check("LF3-4: after 'Mark as a bot' focus is on the card's new role button", await p.evaluate(() => document.activeElement && document.activeElement.dataset.role === ""));
  roleFail = true;
  await p.feed([status({ stats: { msgs: 2, role_save_errors: 0 } })], 50);
  await p.click('#drawer [data-role=""]'); await sleep(400);
  await p.feed([status({ stats: { msgs: 3, role_save_errors: 1 } })], 300);
  check("test gap: a role the server could not save says so on the card and in a header pill",
    /not applied/.test(await p.textContent("#roleErr")) && /roles not saved/.test(await p.textContent("#errs")));
  roleFail = false;
  await p.keyboard.press("Escape"); await sleep(150);

  // LF3-8: another page sets a role: the open card offers a reload
  delete roles2[BOT];
  await p.click("#rows .line.bot .who"); await sleep(400);
  await p.feed([status({ owner_roles: 0, owner_roles_rev: 20 })], 200);
  roles2[BOT] = { role: "bot", login: "gamebot", display: "gamebot", set: "2026-11-01T02:00:00Z" };
  await p.feed([status({ owner_roles: 1, owner_roles_rev: 21 })], 500);
  check("LF3-8: a role set on another page offers this card a reload", !!(await p.$("#drawer #dUpd")) && !!(await p.$('#drawer [data-role="bot"]')));
  await p.keyboard.press("Escape"); await sleep(150);

  // LF3-3: "Not a bot" on a stub card (a marked chatter with no profile here) ends in a plain note, not an error
  roles2.stub1 = { role: "bot", login: "stubby", display: "Stub" + XS, set: "2026-11-01T01:30:00Z" };
  await p.evaluate(() => openCard("stub1", "cozy")); await sleep(400);
  await p.click('#drawer [data-role=""]'); await sleep(500);
  const stubTxt = await p.textContent("#drawer");
  check("LF3-3: 'Not a bot' on a stub card says so instead of 'Couldn't load this chatter'", /Not a bot any more/.test(stubTxt) && !/Couldn't load/.test(stubTxt), stubTxt.replace(/\s+/g, " ").slice(0, 120));
  check("test gap: a stub's display and the roles list stay inert", !(await p.evaluate(() => window.__xss)) && !(await p.$("#drawer img")));
  await p.keyboard.press("Escape"); await sleep(150);

  // LF3-4: Close after the last Remove does not drop focus on <body>
  await p.feed([status({ owner_roles: 1, owner_roles_rev: 22 })], 200);
  await p.click("#rolesBtn"); await sleep(400);
  await p.click(`#drawer [data-unrole="${BOT}"]`); await sleep(500);
  await p.keyboard.press("Escape"); await sleep(200);
  const focTag = await p.evaluate(() => document.activeElement ? document.activeElement.tagName : "");
  check("LF3-4: closing the roles list after the last Remove keeps focus in the page", focTag !== "BODY" && !!focTag, focTag);

  // LB3-3: the status line says when Stop is still saving
  await p.feed([status({ running: false, stopping: true })], 200);
  check("LB3-3: the status says 'stopping…' while Stop saves", /stopping/.test(await p.textContent("#status")));
  check("no page errors in the audit checks", !p.errs.length, p.errs.join("; "));
  await p.close();

  // ---------------------------------------------------------------- audit 2026-11-01b (live_frontend.md LF4-n and its test gaps)
  const FXS = fx.batches.flat().find(r => r.type === "status").session;     // the fixture server's session
  const hdr = async p => ({ st: (await p.textContent("#status")).trim(), btn: await p.textContent("#startBtn"), dis: await p.$eval("#startBtn", e => e.disabled),
    dot: !!(await p.$("#status .dot.on")), pill: await p.$eval("#rolesBtn", e => e.hidden ? "" : e.textContent),
    strip: (await p.textContent("#strip")).replace(/\s+/g, " ").trim(), stale: await p.$$eval("#tabs .tab.stale", e => e.length),
    tabs: await p.$$eval("#tabs .tab", e => e.length), cards: await p.$$eval("#heads .hu", e => e.length), rows: await p.$$eval("#rows .line", e => e.length) });

  // LF4-1: the server goes away: the header says offline and stays so through renders; button, pill and lists keep the last
  // state, marked stale; Start/Stop cannot be pressed meanwhile
  for (const [w, h, tag] of [[1500, 950, ""], [390, 800, "_phone"]]) {
    p = await open(b, w, h);
    for (const batch of fx.batches.slice(0, 40)) await p.feed(batch, 5);
    await p.feed([status({ session: FXS, owner_roles: 2 })], 200);
    await p.drop(true);
    await p.mouse.click(5, 5); await sleep(400);                 // any click renders again
    await p.evaluate(() => render()); await sleep(100);
    const off = await hdr(p);
    if (!tag) {
      check("LF4-1: a lost server reads 'offline · reconnecting…' through renders, the dot off, Stop and the pill kept but Stop not pressable",
        /offline · reconnecting/.test(off.st) && !off.dot && off.btn === "Stop" && off.dis && /bots you set: 2/.test(off.pill), JSON.stringify(off));
      check("LF4-1: while it is lost the lists are marked as they were (strip note, dimmed tabs)",
        /server not reachable since \d\d:\d\d:\d\d: the lists below are as they were then/.test(off.strip) && off.stale === off.tabs && off.tabs === 1
        && off.cards > 0 && await p.evaluate(() => document.body.classList.contains("offline")), JSON.stringify(off));
      await p.click("#startBtn", { force: true }).catch(() => {}); await sleep(200);
      check("LF4-1: no Stop is sent while the server is lost", !p.posts.some(x => x[0] === "stop"));
    } else check("LF4-1: no sideways scroll at 390 px while offline", await p.evaluate(() => document.documentElement.scrollWidth) <= 390);
    await shot(p, "lf4_offline" + tag);
    await p.back(); await sleep(100);
    await p.feed([status({ session: FXS, owner_roles: 2 })], 300);
    const on = await hdr(p);
    if (!tag) check("LF4-1: back on the same run: 'live' again, lists kept and no longer marked (plus the gap line, LF5-1)", on.st === "live" && on.dot && !on.dis && !on.stale
      && on.cards === off.cards && on.rows === Math.min(off.rows + 1, 300) && (await p.$$eval("#rows .line.gap", e => e.length)) === 1 && !/not reachable/.test(on.strip), JSON.stringify(on));
    // the server restarts into idle: the old run's lists go, the header and the button agree
    await p.drop(); await p.back();
    await p.feed([status({ running: false, connected: false, channels: [], session: null, stopped_at: null })], 300);
    await p.mouse.click(5, 5); await sleep(300);
    const idle = await hdr(p);
    if (!tag) check("LF4-1: a server restarted into idle clears the old run: no tabs, cards, rows or strip; the status 'idle' alone on the stats line, with Start",
      idle.st === "idle" && idle.btn === "Start" && !idle.dis && !idle.tabs && !idle.cards && !idle.rows && idle.strip === "idle"
      && /Type a channel above/.test(await p.textContent("#chat")), JSON.stringify(idle));
    await shot(p, "lf4_restart_idle" + tag);
    check("no page errors when the server goes and comes back" + (tag ? " (390 px)" : ""), !p.errs.length, p.errs.join("; "));
    await p.close();
  }
  // a server restarted into a new run (another session) of the same channels: the old lines and cards go
  p = await open(b);
  for (const batch of fx.batches.slice(0, 40)) await p.feed(batch, 5);
  await p.drop(); await p.back();
  await p.feed([status({ session: "20261101-190000_cozy-other" })], 300);
  const rerun = await hdr(p);
  check("LF4-1: a server restarted into a new session clears the old lines and cards, keeps its tabs", rerun.st === "live" && rerun.tabs === 1
    && !rerun.cards && !rerun.rows && /waiting for chat/.test(rerun.strip), JSON.stringify(rerun));
  await p.feed([msg("cozy", "n1", "newrun", "first line of the new run")], 300);
  check("LF4-1: the new run's lines then show", (await p.$$eval("#rows .line", e => e.length)) === 1);
  await p.close();
  // LF4-1 boot id: a restart into the same channels with session logging off (no session to compare) is still found
  const FXB = fx.batches.flat().find(r => r.type === "status").boot;
  p = await open(b);
  for (const batch of fx.batches.slice(0, 40)) await p.feed(batch, 5);
  await p.drop(); await p.back();
  await p.feed([status({ session: null, boot: FXB })], 300);
  const same = await hdr(p);
  await p.drop(); await p.back();
  await p.feed([status({ session: null, boot: FXB + "x" })], 300);
  const reboot = await hdr(p);
  check("LF4-1 boot id: the server sends one, and back on the same process the lists stay", !!FXB && same.rows > 0 && same.cards > 0, JSON.stringify(same));
  check("LF4-1 boot id: a restart into the same channels without a session log clears the old lines and cards",
    reboot.st === "live" && reboot.tabs === 1 && !reboot.cards && !reboot.rows, JSON.stringify(reboot));
  check("no page errors in the boot id checks", !p.errs.length, p.errs.join("; "));
  await p.close();

  // ---------------------------------------------------------------- audit 2026-11-01b round 2 (live_frontend_r2.md LF5-n)
  // LF5-1: back on the same process, the lines it sent meanwhile never reach the page: each channel's chat gets one line
  // saying so, under every filter; it is no message (no count, no jump bar)
  const GAP = /^\d\d:\d\d:\d\d–\d\d:\d\d:\d\d not connected: lines from then aren't shown( \(your clock\))?$/;   // (a replay says whose clock)
  const rowsTxt = p => p.$$eval("#rows .line", e => e.map(x => x.classList.contains("gap") ? "GAP " + x.textContent.trim() : x.querySelector(".msg bdi").textContent.trim()));
  for (const [w, h, tag] of [[1500, 950, ""], [390, 800, "_phone"]]) {
    p = await open(b, w, h);
    // a replay may hold several channels (a live session watches one: owner 2026-11-07); the page treats both alike
    const RS = { boot: "A", mode: "replay", channels: ["cozy", "other"] };
    await p.feed([status(RS), msg("cozy", "g1", "g1", "before the drop 1", { flags: ["first"] }), msg("cozy", "g2", "g2", "before the drop 2", { toxicity: tox("medium") }),
      msg("other", "g3", "g3", "other before", { flags: ["first"] })], 200);
    await p.drop(); await p.back();
    await p.feed([status(RS)], 100);
    await p.feed([msg("cozy", "g4", "g4", "after the drop", { flags: ["first"] }), status(RS)], 300);
    const rt = await rowsTxt(p);
    if (!tag) {
      check("LF5-1: back on the same boot the chat marks the gap between the lines before and after it, as one line",
        rt.length === 4 && rt[0] === "before the drop 1" && rt[1] === "before the drop 2" && GAP.test(rt[2].slice(4)) && rt[3] === "after the drop", JSON.stringify(rt));
      const filt = {};
      for (const fl of ["first", "flagged"]) { await p.click(`[data-f="${fl}"]`); await sleep(150); filt[fl] = await rowsTxt(p); }
      await p.click('[data-f="all"]'); await sleep(150);
      check("LF5-1: the gap line shows under First-timers and Flagged too, in its place (Flagged holds the toxic line)",
        filt.flagged.length === 2 && filt.flagged[0] === "before the drop 2" && /^GAP /.test(filt.flagged[1])
        && filt.first.length === 3 && /^GAP /.test(filt.first[1]), JSON.stringify(filt));
      await p.click('#tabs .tab[data-ch="other"]'); await sleep(200);
      const ot = await rowsTxt(p);
      check("LF5-1: every channel the page held gets its gap line", ot.length === 2 && GAP.test(ot[1].slice(4)), JSON.stringify(ot));
      await p.click('#tabs .tab[data-ch="cozy"]'); await sleep(200);
      check("LF5-1: the gap line is no message: no 'Show older' count, no jump bar", !(await p.$("#olderBtn")) && await p.$eval("#jump", j => j.hidden));
    }
    await p.$eval("#rows .line.gap", e => e.scrollIntoView({ block: "center" }));
    await shot(p, "lf5_gap" + tag);
    if (tag) check("LF5-1: the gap line fits at 390 px (no sideways scroll)", await p.evaluate(() => document.documentElement.scrollWidth) <= 390
      && await p.$eval("#rows .line.gap", e => e.getBoundingClientRect().right <= 390));
    check("no page errors in the gap checks" + (tag ? " (390 px)" : ""), !p.errs.length, p.errs.join("; "));
    await p.close();
  }
  // LF5-1: no gap line when nothing was sent meanwhile (stopped before the drop), nor after a restart (the lists start clean)
  p = await open(b);
  await p.feed([status({ boot: "A" }), msg("cozy", "s1", "s1", "line before stop")], 150);
  await p.feed([status({ boot: "A", running: false, stopped_at: Date.now() - 5000 })], 150);
  await p.drop(); await p.back();
  await p.feed([status({ boot: "A", running: false, stopped_at: Date.now() - 5000 })], 300);
  const stopRows = await rowsTxt(p);
  await p.feed([status({ boot: "A" }), msg("cozy", "s2", "s2", "new run line")], 150);
  await p.drop(); await p.back();
  await p.feed([status({ boot: "B" })], 300);
  check("LF5-1: no gap line after a drop that came after Stop, or across a restart",
    stopRows.length === 1 && !/GAP/.test(stopRows.join()) && !(await p.$("#rows .line.gap")) && !p.errs.length, JSON.stringify(stopRows) + p.errs.join("; "));
  await p.close();
  // LF5-1: a page the server cut off as too slow (ui_clients_dropped) has a pill, like ui_records_dropped
  p = await open(b);
  await p.feed([status({ boot: "A", stats: { msgs: 1, ui_clients_dropped: 0 } })], 100);
  await p.feed([status({ boot: "A", stats: { msgs: 2, ui_clients_dropped: 1 } })], 300);
  check("LF5-1: a page cut off as too slow shows a server-error pill", /a page was cut off/.test(await p.textContent("#errs")), await p.textContent("#errs"));
  await p.close();

  // ---------------------------------------------------------------- audit 2026-11-01b round 3 (live_frontend_r3.md LF6-1)
  // LF6-1: the gap line, the strip's "not reachable since" and the status tooltip start at the last frame the page
  // received, not at the close. stall: the socket goes silent and the page's watchdog closes it 12-15 s later;
  // sleep: the machine sleeps 90 min with the socket half-open (fake clock for both)
  for (const kind of ["stall", "sleep"]) {
    const T = new Date("2026-11-01T12:00:00Z").getTime();
    p = await open(b, 1500, 950, {}, T);
    const tick = async (ms, real = 20) => { await p.clock.runFor(ms); await sleep(real); };
    p.sock.send(JSON.stringify([status({ boot: "A" }), msg("cozy", "w1", "w1", "before the hole")])); await tick(300, 150);
    const last = await p.evaluate(() => lastFrame);
    p.refuse = true; p.sock = null;                   // nothing reaches the page from here on; every retry is refused
    if (kind === "stall") for (let i = 0; i < 20; i++) await tick(1000);
    else { await p.clock.fastForward("01:30:00"); for (let i = 0; i < 10; i++) await tick(200); }
    const exp = await p.evaluate(t => hhmm(t), last);
    const strip = await p.textContent("#strip"), tip = await p.getAttribute("#status", "title");
    check(`LF6-1 (${kind}): while offline, the strip and the status tooltip give the last frame heard`,
      strip.includes(`server not reachable since ${exp}`) && (tip || "").includes(`has not answered since ${exp}`), JSON.stringify([exp, strip, tip]));
    p.refuse = false;
    for (let i = 0; i < 100 && !p.sock; i++) await tick(200);
    p.sock.send(JSON.stringify([status({ boot: "A" })])); await tick(200, 100);
    p.sock.send(JSON.stringify([msg("cozy", "w2", "w2", "after the hole"), status({ boot: "A" })])); await tick(300, 150);
    const g = await p.evaluate(() => chan("cozy").lines.filter(r => r.type === "gap").map(x => [x.from, x.to]));
    const gt = await p.$$eval("#rows .line.gap", e => e.map(x => x.textContent.trim()));
    check(`LF6-1 (${kind}): the gap line starts at the last frame heard and spans the whole hole`,
      g.length === 1 && g[0][0] === last && g[0][1] - last >= (kind === "stall" ? 12000 : 90 * 60000) && gt.length === 1 && gt[0].startsWith(exp + "–"),
      JSON.stringify([last, g, gt]));
    check(`no page errors in the LF6-1 ${kind} check`, !p.errs.length, p.errs.join("; "));
    await p.close();
  }

  // LF5-2: a replay server that restarts (its boot changed) sends its chat again: the page starts clean, no line twice;
  // the same for a live server restarted into a replay
  {
    const rs = o => status({ mode: "replay", running: true, channels: ["cozy"], session: null, ...o });
    const lines = [msg("cozy", "r1", "r1", "replay one", { flags: ["first"] }), msg("cozy", "r2", "r2", "replay two")];
    p = await open(b);
    await p.feed([rs({ boot: "A" }), ...lines], 200);
    const n0 = await p.$$eval("#rows .line", e => e.length);
    await p.drop(); await p.back();
    await p.feed([rs({ boot: "B" }), ...lines], 300);
    const n1 = await p.$$eval("#rows .line", e => e.length), buf1 = await p.evaluate(() => chan("cozy").lines.length);
    check("LF5-2: a restarted replay server's chat is not shown twice (rows, buffer)", n0 === 2 && n1 === 2 && buf1 === 2,
      JSON.stringify({ n0, n1, buf1 }));
    await p.drop(); await p.back();
    await p.feed([rs({ boot: "B" })], 300);
    check("LF5-2: a replay reconnect to the same boot keeps its lines and marks the gap",
      (await p.$$eval("#rows .line", e => e.length)) === 3 && /not connected: lines from then aren't shown \(your clock\)$/.test(await p.textContent("#rows .line.gap")));
    await p.close();
    p = await open(b);
    await p.feed([status({ boot: "L", channels: ["cozy"] }), ...lines], 200);
    await p.drop(); await p.back();
    await p.feed([rs({ boot: "R" }), ...lines], 300);
    check("LF5-2: a live server restarted into a replay starts the page clean", (await p.$$eval("#rows .line", e => e.length)) === 2
      && (await p.evaluate(() => chan("cozy").lines.length)) === 2 && !p.errs.length, p.errs.join("; "));
    await p.close();
  }

  // LF4-2 (New faces removed 2026-11-10, owner: the First-timers filter shows the same lines): the Watchlist count follows the room
  p = await open(b);
  const lcard = (uid, level) => ({ user_id: uid, display: uid, level, why: ["slur or dehumanising language"], n: 1, severity: 3, toxic: 1, last_seen: T0, since: T0, quote: "q", first: false });
  const lroom = items => ({ type: "room", channel: "cozy", msgs_per_min: 1, chatters: 3, first_timers_this_min: 0, raid: null,
    notices: { items, counts: { act: items.length, watch: 0 }, more: 0 }, notices_earlier: [] });
  await p.feed([status(), lroom([lcard("a", "act")]), msg("cozy", "a", "a", "hi")], 300);
  await p.feed([lroom([lcard("a", "act"), lcard("b", "act"), lcard("c", "act")]), msg("cozy", "z", "z", "hi", { flags: ["first"] })], 400);
  const wl = await p.textContent("#wlCount");
  check("LF4-2: the Watchlist count follows the room; no New faces tab", wl === "3" && !(await p.$("#faces,#fcCount,[data-p]")), JSON.stringify([wl]));
  await p.close();

  // LF4-3: Enter twice, or Start then Enter, while /api/start is in flight sends one POST and leaves no error
  for (const how of ["Enter, Enter", "click, Enter"]) {
    let starts = 0;
    p = await open(b, 1500, 950, { "/api/start": async r => { starts++; await sleep(500);
      return starts === 1 ? json(r, { ok: true, channels: ["alpha"], session: "s1" }) : json(r, { error: "already running; stop first" }, 409); } });
    await p.feed([status({ running: false, channels: [], session: null })]);
    await p.fill("#chanInput", "alpha");
    if (how === "click, Enter") await p.click("#startBtn"); else { await p.focus("#chanInput"); await p.keyboard.press("Enter"); }
    await sleep(60);
    const inFlight = { btn: await p.$eval("#startBtn", e => e.disabled), input: await p.$eval("#chanInput", e => e.disabled) };
    await p.focus("#chanInput").catch(() => {}); await p.keyboard.press("Enter");
    await p.evaluate(() => startStop());                         // an Enter that slips through
    await sleep(900);
    const n = p.posts.filter(x => x[0] === "start").length, err = await p.textContent("#ctlErr");
    check(`LF4-3: ${how} during a Start in flight sends one POST, no error; button and input disabled meanwhile`,
      n === 1 && !err && inFlight.btn && inFlight.input && await p.textContent("#startBtn") === "Stop", JSON.stringify({ n, err, inFlight }));
    await p.close();
  }

  // LF4-4: a double click on the card's LLM tools sends one POST each; the button is disabled while it asks, also across a redraw
  const slow = (body) => async r => { await sleep(500); return json(r, body); };
  p = await open(b, 1500, 950, { "/api/llm_review": slow({ label: "mild", confidence: "low", rationale: "jabs", rules: [] }),
    "/api/draft_warning": slow({ warranted: true, warning: "please keep it civil", why: "insult" }) });
  for (const batch of fx.batches.slice(0, 30)) await p.feed(batch, 5);
  await sleep(200);
  const tu = await p.$eval("#rows .line[data-u]", e => e.dataset.u);
  await p.evaluate(u => openCard(u, channel), tu); await sleep(400);
  await p.dblclick("#llmBtn"); await p.dblclick("#warnBtn", { force: true }).catch(() => {}); await sleep(50);
  const dis = { llm: await p.$eval("#llmBtn", e => e.disabled), warn: await p.$eval("#warnBtn", e => e.disabled), out: await p.textContent("#llmOut") };
  await p.evaluate(u => openCard(u, channel), tu); await sleep(150);   // the card is redrawn while both ask
  const dis2 = { llm: await p.$eval("#llmBtn", e => e.disabled), warn: await p.$eval("#warnBtn", e => e.disabled) };
  await p.$eval("#llmBtn", e => e.scrollIntoView({ block: "center" })); await shot(p, "lf4_tools_asking");
  await p.evaluate(() => { $("llmBtn").disabled = false; $("llmBtn").click(); $("warnBtn").disabled = false; $("warnBtn").click(); });   // clicks that slip through
  await sleep(900);
  const nl = p.posts.filter(x => x[0] === "llm_review").length, nw = p.posts.filter(x => x[0] === "draft_warning").length;
  check("LF4-4: double clicks on the LLM tools send one POST each; disabled with 'Asking…' while they ask, also after a redraw",
    nl === 1 && nw === 1 && dis.llm && dis.warn && /Asking/.test(dis.out) && dis2.llm && dis2.warn, JSON.stringify({ nl, nw, dis, dis2 }));
  check("LF4-4: the answers land in the redrawn card and the buttons work again", /mild/.test(await p.textContent("#llmOut"))
    && /keep it civil/.test(await p.textContent("#warnOut")) && !(await p.$eval("#llmBtn", e => e.disabled)) && !(await p.$eval("#warnBtn", e => e.disabled)));
  await p.close();

  // LF4-5: under the Flagged filter a line that a re-read makes toxic joins the list; scrolled up, it counts in the jump bar
  p = await open(b);
  await p.feed([status(), msg("cozy", "q1", "q1", "first clean line", { id: "cl1" }), msg("cozy", "q2", "q2", "you are trash", { id: "tx1", toxicity: tox("medium") })], 200);
  await p.click('[data-f="flagged"]'); await sleep(200);
  const tb = await p.$$eval("#rows .line", e => e.map(x => x.dataset.id));
  await p.mouse.move(5, 5);
  await p.feed([{ type: "msg_update", channel: "cozy", user_id: "q1", id: "cl1", toxicity: tox("high", ["threat"], { source: "llm", layer: { level: "none" } }) }], 400);
  const ta = await p.$$eval("#rows .line", e => e.map(x => x.dataset.id));
  check("LF4-5: under the Flagged filter a line re-read as toxic joins the list", tb.join() === "tx1" && ta.join() === "cl1,tx1", JSON.stringify([tb, ta]));
  for (let i = 0; i < 80; i++) await p.feed([msg("cozy", "t" + i, "t" + i, "insult " + i, { id: "ti" + i, toxicity: tox("medium") })], 0);
  await p.feed([msg("cozy", "q3", "q3", "another clean line", { id: "cl2" })], 300);
  await p.$eval("#chat", c => { c.scrollTop = 0; c.dispatchEvent(new Event("scroll")); }); await sleep(200);
  await p.feed([{ type: "msg_update", channel: "cozy", user_id: "q3", id: "cl2", toxicity: tox("medium") }], 300);
  const jb = await p.$eval("#jump", j => ({ hidden: j.hidden, text: j.textContent }));
  await p.click("#jump", { timeout: 3000 }).catch(() => {}); await sleep(300);
  check("LF4-5: scrolled up, it shows as a new message and is there after 'Jump to latest'", !jb.hidden && /1 new message/.test(jb.text)
    && !!(await p.$('#rows .line[data-id="cl2"]')) && !p.errs.length, JSON.stringify(jb) + p.errs.join("; "));
  await p.close();

  // LF4-8: the roles list at phone width: a long name gives way, Remove stays on one line
  p = await open(b, 390, 800, { "/api/roles": r => json(r, { roles: [{ user_id: "1", role: "bot", login: "averyveryveryverylongbotloginname",
    display: "AVeryVeryVeryLongBotDisplayNameWithoutSpaces", set: "2026-11-01T01:30:00Z" }], saves: true }) });
  for (const batch of fx.batches.slice(0, 10)) await p.feed(batch, 5);
  await p.feed([status({ owner_roles: 1 })], 200);
  await p.click("#rolesBtn"); await sleep(400);
  const rm = await p.$eval('#drawer [data-unrole="1"]', e => { const r = e.getBoundingClientRect(); return { h: Math.round(r.height), right: Math.round(r.right), lines: e.getClientRects().length }; });
  const dw = await p.$eval("#drawer", d => [d.scrollWidth, d.clientWidth]);
  check("LF4-8: at 390 px a long name in the roles list leaves 'Remove' on one line, inside the card", rm.h <= 32 && rm.right <= 390 && dw[0] <= dw[1], JSON.stringify({ rm, dw }));
  await shot(p, "lf4_phone_roles");
  await p.close();

  // test gap: HTML in every profile field the user card shows and in the LLM tools' answers stays inert
  {
    const X = `<img src=x onerror="window.__xss=(window.__xss||0)+1">`, XA = `x" onmouseover="window.__xss=1" y="`;
    const base = Object.values(fx.profiles)[0];
    const prof = { ...base, display: "D" + X, verdict: "V" + X, sub_months: "3" + X, badges: { ["vip" + X]: 1 }, behaviour_type: "bt" + X,
      archetypes: [{ key: "x", label: "L" + X, desc: "d" + XA, family: "f" + XA, p: 0.5 }],
      reasons: [{ archetype: "A" + X, family: XA, reason: "R" + X, weight: "1" + X }], account_reasons: [{ reason: "AR" + X }],
      register: { style: "S" + X, margin: .5, runner_up: "RU" + X, summary: "SU" + X, words: "9" + X, tags: ["T" + X], confidence: "c" + X },
      vocabulary: [{ family: "F" + X, terms: ["t" + X], t1_messages: "2" + X }], distress_notes: "1" + X, reactions: [[T0, "ban" + X, "det" + X]],
      person_read: { label: "bad_faith" + X, rationale: "RA" + X, n: "3" + X, ts: T0 }, notice: { level: "act" + XA, why: ["W" + X], owner: true },
      toxic: { messages: "1" + X, of: "2" + X, kinds: [["k" + X, "1" + X]] },
      recent_messages: [{ id: "m" + XA, ts: T0, text: "RM" + X, reasons: ["r" + XA], flags: ["first"], toxicity: tox("high", ["hate" + X], { target: "t" + X, source: "llm", layer: { level: "x" + X } }) }],
      evidence: [{ id: "e1", ts: T0, text: "EV" + X, flags: [], counted: false, older: true, toxicity: tox("medium", ["k" + X], { source: "owner", auto: { level: "l" + X, kinds: ["a" + X] } }) }],
      owner_role: null };
    p = await open(b, 1500, 950, { "/api/profile": r => json(r, prof),
      "/api/llm_review": r => json(r, { label: "L" + X, confidence: "C" + X, rationale: "R" + X, rules: ["ru" + X] }),
      "/api/draft_warning": r => json(r, { warranted: true, warning: "W" + X, why: "Y" + X }) });
    for (const batch of fx.batches.slice(0, 10)) await p.feed(batch, 5);
    await p.evaluate(() => openCard("u1", "cozy")); await sleep(400);
    await p.evaluate(() => document.querySelectorAll("#drawer details").forEach(d => d.open = true));
    await p.click("#llmBtn"); await p.click("#warnBtn"); await sleep(400);
    await p.evaluate(() => { for (const el of document.querySelectorAll("*")) el.dispatchEvent(new MouseEvent("mouseover", { bubbles: false })); });
    await sleep(200);
    check("test gap: HTML in the profile fields and the LLM answers stays inert", !(await p.evaluate(() => window.__xss)) && !(await p.$("#drawer img"))
      && !(await p.$("[onmouseover]")) && /W<img/.test(await p.textContent("#warnOut")) && !p.errs.length, p.errs.join("; "));
    await p.close();
  }

  // ---------------------------------------------------------------- the AI: picker and speed before Start, hybrid review
  {
    const J = (r, body, st = 200) => r.fulfill({ status: st, contentType: "application/json", body: JSON.stringify(body) });
    const stopped = o => status({ running: false, connected: false, channels: [], stopped_at: null, session: null, ...o });
    let info = fx.ai.nv12;
    // polish 2026-11-10: a row's speed source and GPU layer cap moved into the speed's tooltip; ROW reads both
    const ROW = e => e.textContent.replace(/\s+/g, " ") + " <" + [...e.querySelectorAll("[data-spd]")].map(x => x.title).join(" | ") + ">";
    p = await open(b, 1500, 950, { "/api/ai/select": r => J(r, { ok: true, task: { kind: "select" } }),
      "/api/ai/speed": r => J(r, { ok: true, task: { kind: "speed" } }), "/api/ai/override": r => J(r, { ok: true }),
      "/api/ai": r => J(r, info) });
    await p.feed([stopped({ ai: { mode: "single", live: "gemma4mtp:31b-n3-text", reviewer: null, task: null, last: {}, scope_default: "b" } })], 600);
    const panel = await p.textContent("#aiPanel");
    const first = await p.$eval("#aiPanel label.r", ROW);
    check("AI panel before Start: specs, the recommended MTP 31B first with its tested rate",
      !(await p.$eval("#aiPanel", e => e.hidden)) && /RTX 4070 Ti/.test(panel) && /Ollama 0\.40\.11/.test(panel)
      && /Gemma 4 31B/.test(first) && /recommended/.test(first) && /Estimated ~40 msgs\/min/.test(first) && /tested on 12 GB Nvidia/.test(first)
      && /\+66% accuracy/.test(first) && !/% of the 20b|117%|over No AI|gates pass|per person read|GPU layers ·/.test(panel)
      && /30 GPU layers/.test(first) && !!(await p.$('#aiPanel label.r i[data-spd="estimate"]')), first.slice(0, 300));
    check("polish: the line under This machine only says to press Check Speed", /^Speeds in italics are estimates: press Check Speed to measure/.test(await p.textContent("#aiPanel [data-tier-note]"))
      && !/tested on this kind|meets or exceeds/.test(panel) && (await p.textContent("#aiPanel [data-ai-speed]")) === "Check Speed");
    // owner 2026-11-10: every row's figures bold; no (i) on the AI rows or the notes (the Review box keeps its own)
    {
      const bold = await p.$$eval("#aiPanel [data-row-figs]", es => es.map(e => +getComputedStyle(e).fontWeight));
      check("single rows' figures (and No AI's) are bold; the AI panel has no (i)", bold.length >= 3 && bold.every(w => w >= 600)
        && !(await p.$("#aiPanel [data-info]")), JSON.stringify(bold));
    }
    check("polish: what accuracy and the hybrid mean are said under the list", /Accuracy: how well a setup marks toxic messages/.test(await p.textContent("#aiPanel [data-acc-note]"))
      && /Hybrid: a fast model reads every message live/.test(await p.textContent("#aiPanel [data-hyb-note]")) && !/tested on 12 GB/.test(await p.textContent("#aiPanel [data-hyb-note]")));
    // vramcap (owner 2026-11-07): the memory available for models, read at start, and the cap estimated from it
    check("no screen on the card (the fixture): no desktop reserve is mentioned", !/kept for the desktop/.test(await p.textContent('#aiPanel [data-avail="vram"]')));
    info = { ...fx.ai.nv12, memory: { ...fx.ai.nv12.memory, reserve_gb: 1.5 } };
    await p.evaluate(() => loadAI(true)); await sleep(300);
    check("a card that drives a screen: the VRAM line says what is kept for the desktop (owner 2026-11-10)",
      /11\.6 GB available, 1\.5 GB kept for the desktop/.test(await p.textContent('#aiPanel [data-avail="vram"]')));
    info = fx.ai.nv12;
    await p.evaluate(() => loadAI(true)); await sleep(300);
    check("the memory available for models is shown beside the GPU and RAM dropdowns",
      /11\.6 GB available/.test(await p.textContent('#aiPanel [data-avail="vram"]')) && /88\.5 GB available/.test(await p.textContent('#aiPanel [data-avail="ram"]')));
    check("the text 31B + MTP's cap on this card: 30, estimated from 11.6 GB available (the measured point)", /30 GPU layers \(estimated from 11\.6 GB available\)/.test(first), first.slice(0, 260));
    const sumTxt = async () => (await p.textContent("#aiPanel .sum")).replace(/\s+/g, " ");
    check("the AI bar: the setup in use and its msgs/min, estimated until checked, in one line", /AI ?Live and Review: Gemma 4 31B \(Estimated ~40 msgs\/min\)/.test(await sumTxt()), await sumTxt());
    // sweep 2026-11-09: the keep-up (~40) counts person reads and flagged lines; the re-reads that carry the 31B's gain
    // thin from ~20 (ASSESSMENT_20261109 sections 3, 6): both on the row and on the bar, the reason in the tooltip
    // owner 2026-11-10: one msgs/min per setup, no "full accuracy up to", no "MTP" anywhere on the page
    check("one msgs/min per setup: no re-read reach on the rows, the bar or in a tooltip; no MTP in the panel",
      !/full accuracy|re-read/.test(panel + first + await sumTxt()) && !(await p.$("#aiPanel [data-reread]")) && !/MTP/.test(panel)
      && /Msgs\/min|msgs\/min figure is the busiest chat/.test(await p.textContent("#aiPanel [data-acc-note]"))
      && !(await p.$("#aiPanel [data-acc-note] b + b")) && (await p.$$eval("#aiPanel [data-acc-note] b", e => e.length)) === 1, first.slice(0, 200));
    const r12 = await p.$$eval("#aiPanel label.r", e => e.map(x => x.textContent.replace(/\s+/g, " ")).find(t => /^\s*Gemma 4 12B(?! live)/.test(t)) || "");
    check("…not on a setup that re-reads every line up to its keep-up (the 12B + MTP alone)", /≥150 msgs\/min/.test(r12) && !/full accuracy up to/.test(r12), r12.slice(0, 200));
    await p.fill("#chanInput", "cozy, other"); await p.click("#startBtn"); await sleep(150);
    check("one channel per live session: two typed are refused on the page, nothing POSTed",
      /one channel at a time: Stop, then Start the other/.test(await p.textContent("#ctlErr")) && !p.posts.some(x => x[0] === "start")
      && (await p.getAttribute("#chanInput", "placeholder")) === "channel");
    await p.fill("#chanInput", "");
    const names = async () => p.$$eval("#aiPanel label.r .nm", e => e.map(x => { const c = x.cloneNode(true); c.querySelectorAll(".note").forEach(n => n.remove());
      return c.textContent.replace(/\s+/g, " ").trim(); }));
    let ids = await names();
    check("one list, no mode tabs: best quality first (31B + MTP, the hybrid, 12B + MTP), No AI last",
      !(await p.$("#aiPanel [data-ai-mode]")) && ids.length === 4 && /^Gemma 4 31B/.test(ids[0])
      && /^12B \+ 31B Hybrid/.test(ids[1]) && /^Gemma 4 12B/.test(ids[2]) && /^No AI/.test(ids[3])
      && /Gemma 4 12B live, Gemma 4 31B reviews/.test(await p.textContent("#aiPanel [data-ai-hyb]  ~ .nm .note").catch(() => "")), JSON.stringify(ids));
    check("one list, one recommended mark: the 31B + MTP (no chat rate known yet)", ids.filter(x => /recommended/.test(x)).length === 1 && /recommended/.test(ids[0]), JSON.stringify(ids));
    // MTP builds only: the names no longer say "MTP" (owner 2026-11-10), so the rows' model ids are checked
    const rowIds = async () => p.$$eval("#aiPanel [data-ai-row],#aiPanel [data-ai-hyb]", e => e.map(x => x.dataset.aiRow || x.dataset.aiHyb));
    const onlyMtp = async () => (await rowIds()).every(id => id.split("|").every(t => /^gemma4mtp:/.test(t)));
    const NOTSHOWN = /gpt-oss|Qwen/;
    check("12 GB: MTP builds only and nothing dominated (no plain Gemma, no 20b, no Qwen, no Qwen hybrid)", await onlyMtp() && !NOTSHOWN.test(ids.join(" | ")), JSON.stringify(await rowIds()));
    const noai = await p.$eval("#aiPanel [data-ai-none]", e => e.closest("label").textContent.replace(/\s+/g, " "));
    check("No AI is a row: +0% accuracy, keeps up with any rate (bold, first after it), the baseline, what it misses, never recommended",
      /\+0% accuracy · Keeps up with any chat rate · the baseline/.test(noai) && /misses some severe messages the AI catches/.test(noai)
      && !/fails the real-chat severe gate/.test(noai) && !/recommended/.test(noai), noai);
    await shot(p, "ai_picker_nv12");
    await p.setViewportSize({ width: 390, height: 800 }); await sleep(200);
    await shot(p, "ai_picker_phone");
    check("stopped, the picker open, at 390 px: no sideways scroll", await p.evaluate(() => document.documentElement.scrollWidth) <= 390);
    await p.setViewportSize({ width: 1500, height: 950 }); await sleep(200);
    // the chat rate decides single vs hybrid (owner 2026-11-07): quality first, the hybrid only above the single's keep-up
    await p.evaluate(() => { aiTypedRate = null; aiPick = null; renderAI(); }); await sleep(100);
    check("no rate known: the page asks to measure the chat to compare", /Measure the chat \(or type its rate\) to compare/.test(await p.textContent("#aiPanel")));
    const typeRate = async v => { await p.fill("#aiPanel [data-ai-typed]", String(v)); await p.dispatchEvent("#aiPanel [data-ai-typed]", "change"); await sleep(150); };
    await typeRate(15);
    let r31 = await p.$eval("#aiPanel label.r", e => e.textContent.replace(/\s+/g, " "));
    check("at 15 msgs/min the single 31B + MTP re-reads every line and is recommended", /at 15 msgs\/min: keeps up/.test(r31) && /recommended/.test(r31), r31.slice(0, 200));
    await typeRate(30);
    r31 = await p.$eval("#aiPanel label.r", e => e.textContent.replace(/\s+/g, " "));
    check("at 30 it keeps up, its re-reads thin, and it stays recommended (it re-reads 80% or more of queued lines to 40)",
      /at 30 msgs\/min: keeps up/.test(r31) && /recommended/.test(r31), r31.slice(0, 200));
    await typeRate(120);
    r31 = await p.$eval("#aiPanel label.r", e => e.textContent.replace(/\s+/g, " "));
    check("at 120 msgs/min it falls behind and is not recommended", /at 120 msgs\/min: falls behind/.test(r31) && !/recommended/.test(r31), r31.slice(0, 200));
    const hy120 = await p.$eval("#aiPanel .rowsAI", e => e.textContent.replace(/\s+/g, " "));
    check("…the hybrid is recommended there, with its likely effect at that rate",
      /recommended/.test(hy120) && /at 120 msgs\/min \(typed\): Review every 10 min takes ~\d+ (s|min) \(live paused \d+%\), likely \+\d+% accuracy/.test(hy120)
      && /once the models are in memory/.test(hy120), hy120.slice(0, 400));   // (no read-ahead yet on this page)
    await shot(p, "ai_hybrid_at_120");
    await p.evaluate(() => { aiTypedRate = null; try { localStorage.removeItem("cw_ai_rate"); } catch(e){} renderAI(); });
    await p.click("#aiPanel [data-ai-hyb]"); await sleep(150);
    const hyb = await p.textContent("#aiPanel .rowsAI");
    check("choosing the hybrid row marks it", !!(await p.$("#aiPanel label.r.sel [data-ai-hyb]")));
    check("hybrid mode: the 12B + 31B pair with both scopes' quality and the review time per chatter",
      /\+62% accuracy · Live:/.test(hyb.replace(/\s+/g, " ")) && !/Watchlist only|\+54%/.test(await p.$eval("#aiPanel label.r.sel", e => e.textContent))
      && /Its accuracy assumes you press Review; between reviews it reads at the live model's level \(\+50%\)\./.test(await p.textContent("#aiPanel [data-hyb-note]"))
      && /Review: Estimated ~\d+ s per chatter/.test(hyb.replace(/\s+/g, " ")), hyb.replace(/\s+/g, " ").slice(0, 300));
    check("the hybrid row's figures are bold, all of them (owner 2026-11-10)",
      +(await p.$eval("#aiPanel [data-hyb-figs]", e => getComputedStyle(e).fontWeight)) >= 600 && /^\+62% accuracy · Live: .+ · Review: .+ per chatter$/.test(
        (await p.textContent("#aiPanel [data-hyb-figs]")).replace(/\s+/g, " ").trim()));
    await shot(p, "ai_picker_hybrid");
    await p.click("#aiPanel [data-ai-use]"); await sleep(200);
    const sel = p.posts.find(x => x[0] === "ai/select");
    check("Use this POSTs the choice with the page's header", !!sel && sel[2] === "1" && JSON.parse(sel[1]).mode === "hybrid"
      && JSON.parse(sel[1]).reviewer === "gemma4mtp:31b-n3-text", JSON.stringify(sel));
    await p.click("#aiPanel [data-ai-none]"); await sleep(150);
    check("No AI can be chosen", !!(await p.$("#aiPanel label.r.sel [data-ai-none]")) && !(await p.$eval("#aiPanel [data-ai-use]", b => b.disabled)));
    await p.click("#aiPanel [data-ai-use]"); await sleep(200);
    check("Use this on No AI POSTs mode none", p.posts.some(x => x[0] === "ai/select" && JSON.parse(x[1]).mode === "none" && x[2] === "1"));
    check("No AI is the baseline and says what it misses", /the baseline/.test(await p.textContent("#aiPanel")) && /misses some severe messages/.test(await p.textContent("#aiPanel")));
    await p.click("#aiPanel [data-ai-speed='gemma4mtp:12b-n2']"); await sleep(200);
    check("Check Speed POSTs the model", p.posts.some(x => x[0] === "ai/speed" && JSON.parse(x[1]).model === "gemma4mtp:12b-n2"));
    await p.feed([stopped({ ai: { mode: "single", live: "gemma4mtp:31b-n3-text", task: { kind: "speed", what: "checking the speed of gemma4mtp:12b-n2", state: "gemma4mtp:12b-n2: measuring (5 calls, 12 s)" }, last: {} } })], 200);
    check("a running task shows its progress and a Cancel", /measuring \(5 calls, 12 s\)/.test(await p.textContent("#aiPanel")) && !!(await p.$("#aiPanel [data-ai-cancel]")));
    // a finished check: one short note on the AI bar, gone after a minute or at the next action (owner 2026-11-08)
    const LS = { model: "gemma4mtp:12b-n2", keepup: 150, keepup_plus: true, person_s: 1.16, aborts: 0, calls: 17, num_gpu: null, notes: ["measured while other work was running"] };
    info = { ...fx.ai.nv12, last: { speed: LS } };
    await p.feed([stopped({ ai: { mode: "single", live: "gemma4mtp:31b-n3-text", task: null, last: { speed: LS } } })], 500);
    const nt = await p.$eval("#aiPanel .sum", e => ({ note: (e.querySelector("[data-ai-note]") || {}).textContent || "", lines: e.closest("#aiPanel").querySelectorAll(".sum").length }));
    check("a finished check leaves one short note on the AI bar (details in its tooltip), not a second line",
      /Checked Gemma 4 12B: ≥150 msgs\/min/.test(nt.note) && nt.lines === 1 && !/other work|aborts/.test(nt.note), JSON.stringify(nt));
    await p.evaluate(() => { aiNoteAt -= 61000; renderAI(); });
    check("…and the note is gone after a minute", !(await p.$("#aiPanel [data-ai-note]")));
    // a 16 GB AMD card: 16 GB, not tested -> no measured-looking number, the cap is found by the check
    info = fx.ai.amd16;
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    const amd = await p.$eval("#aiPanel label.r", ROW);
    check("a bigger card of another vendor (AMD, 16 GB): the 12 GB tier's number as an estimate, its cap estimated from its free VRAM (37, not 30)",
      /Estimated ~40 msgs\/min/.test(amd) && /37 GPU layers \(estimated from 15\.2 GB available\)/.test(amd) && !/untested|not tested|finds it/.test(amd), amd.slice(0, 220));
    check("the AMD card's specs are shown", /Navi 32/.test(await p.textContent("#aiPanel")));
    ids = await names();
    check("16 GB AMD: MTP builds only and nothing dominated", await onlyMtp() && !NOTSHOWN.test(ids.join(" | ")) && ids.length === 4, JSON.stringify(ids));
    // the same card with the desktop holding 2 GB: a LOWER cap than the tested 30 (the safety gain)
    info = fx.ai.nv12busy;
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    let row = await p.$eval("#aiPanel label.r", ROW);
    check("less free VRAM than the test had: a lower cap (20, estimated from 9.6 GB available)", /20 GPU layers \(estimated from 9\.6 GB available\)/.test(row), row.slice(0, 260));
    // 32 GB RAM with 12 GB free: the 31B's CPU part does not fit; flagged, not recommended; the hybrid too; the 12B is
    info = fx.ai.nv12low;
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    const rowsLow = await p.$$eval("#aiPanel label.r", es => es.map(e => e.textContent.replace(/\s+/g, " ")));
    const m31 = rowsLow.find(x => /^ ?Gemma 4 31B/.test(x)) || "", m12 = rowsLow.find(x => /^ ?Gemma 4 12B(?! live)/.test(x)) || "";
    const mHy = rowsLow.find(x => / live, /.test(x)) || "";
    check("free RAM too low for the 31B's CPU part: flagged with what it needs, not recommended",
      /needs ~14 GB free RAM, 12 available/.test(m31) && !/recommended/.test(m31) && /needs ~14 GB free RAM/.test(mHy) && !/recommended/.test(mHy)
      && /recommended/.test(m12) && !/needs ~/.test(m12), JSON.stringify(rowsLow.map(x => x.slice(0, 160))));
    await shot(p, "ai_picker_lowram");
    // a no-GPU laptop with 11 GB free: the larger model flagged, the E4B + MTP recommended and "only just fits"
    info = fx.ai.f32low;
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    const rowsL = await p.$$eval("#aiPanel label.r", es => es.map(e => e.textContent.replace(/\s+/g, " ")));
    const l12 = rowsL.find(x => /Gemma 4 12B/.test(x)) || "", l4 = rowsL.find(x => /Gemma 4 E4B/.test(x)) || "";
    check("no GPU, little free RAM: the 12B + MTP flagged, the E4B + MTP recommended and only just fits",
      /needs ~13 GB free RAM, 11 available/.test(l12) && !/recommended/.test(l12) && /recommended/.test(l4) && /only just fits in memory/.test(l4),
      JSON.stringify(rowsL.map(x => x.slice(0, 160))));
    for (const [fxk, want] of [["f16", /^Gemma 4 E4B/], ["f32", /^Gemma 4 12B/]]) {    // the laptop tiers keep what fits
      info = fx.ai[fxk];
      await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
      ids = await names();
      check(`laptop (${fxk}): its fitting option first, No AI last, no plain Gemma, 20b or Qwen (owner)`, want.test(ids[0]) && /^No AI/.test(ids[ids.length - 1])
        && await onlyMtp() && !NOTSHOWN.test(ids.join(" | ")), JSON.stringify(ids));
    }
    check("a model that is not installed shows its size and command", /not installed · [\d.]+ GB · bin\/ollama\.sh install /
      .test((await p.textContent("#aiPanel")).replace(/\s+/g, " ")));
    await shot(p, "ai_picker_laptop32");
    // ramfit (2026-11-08): a 12 GB card with 16 GB of RAM keeps its GPU: the 12B + MTP (whole on the card, ~1 GB of RAM)
    info = fx.ai.nv12r16;
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    ids = await names();
    const why16 = (await p.textContent("#aiPanel .spec")).replace(/\s+/g, " ");
    check("12 GB GPU with 16 GB RAM: the 12B + MTP recommended (GPU speed, estimated), no 31B or hybrid, No AI last",
      ids.length === 2 && /^Gemma 4 12B recommended/.test(ids[0]) && /^No AI/.test(ids[1]) && /don't fit in the RAM are not listed/.test(why16)
      && /Estimated ≥150 msgs\/min/.test(await p.$eval("#aiPanel label.r", e => e.textContent.replace(/\s+/g, " "))), JSON.stringify([ids, why16]));
    await shot(p, "ai_picker_12gb_16ram");
    // an 8 GB laptop with no free-RAM read (owner 2026-11-09: 16 GB is the lowest RAM option, "<16" is gone): the
    // dropdown shows 16 GB with the RAM read beside it; the E4B + MTP is listed, flagged short, not recommended
    info = fx.ai.f8;
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    ids = await names();
    const ramOpts = await p.$$eval('#aiPanel select[data-ai-ov="ram"] option', es => es.map(e => [e.value, e.selected]));
    const ramRead = await p.$eval("#aiPanel [data-ram-read]", e => e.textContent).catch(() => "");
    const r8 = await p.$eval("#aiPanel label.r", e => e.textContent.replace(/\s+/g, " ")).catch(() => "");
    check("8 GB laptop: RAM options start at 16 GB (no <16), the RAM read shown, E4B + MTP flagged short and not recommended",
      ramOpts.length === 5 && ramOpts[0][0] === "16" && ramOpts[0][1] && !ramOpts.some(o => o[0] === "<16")
      && /7\.6 GB read, below the lowest option/.test(ramRead)
      && ids.length === 2 && /^Gemma 4 E4B/.test(ids[0]) && !/recommended/.test(ids[0]) && /^No AI/.test(ids[1])
      && /needs ~11 GB free RAM, 5\.7 assumed free/.test(r8) && /Estimated/.test(r8)
      && !/tested on this kind/.test(await p.textContent("#aiPanel .spec")), JSON.stringify([ramOpts, ramRead, ids, r8.slice(0, 200)]));
    await shot(p, "ai_picker_8gb_laptop");
    // ramblock (2026-11-08): a saved setup whose RAM part no longer fits starts with the AI off; the AI bar says why and what fits
    info = fx.ai.nv12r16;
    const RB = "Gemma 4 31B + MTP needs ~14 GB free RAM, 12 assumed free with 16 GB of RAM: pick another (recommended: Gemma 4 12B + MTP) or free memory";
    await p.feed([stopped({ ai: { mode: "none", live: "gemma4mtp:31b-n3-text", reviewer: null, task: null, last: {}, blocked: RB } })], 300);
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    const rbTxt = await p.$eval("#aiPanel [data-blocked]", e => e.textContent).catch(() => "");
    check("a saved setup that no longer fits the RAM: AI off for this run, why, and what fits (no MTP in the names)", rbTxt === `AI off for this run: ${RB.replace(/ \+ MTP/g, "")}`, rbTxt);
    await shot(p, "ai_ram_blocked");
    await p.feed([stopped({ ai: { mode: "single", live: "gemma4mtp:31b-n3-text", reviewer: null, task: null, last: {} } })], 200);
    info = fx.ai.amd16;
    await p.evaluate(() => { aiPick = null; return loadAI(); }); await sleep(300);
    await shot(p, "ai_picker_amd16");
    // after a switch the selection follows the setup in use, even when the previous answer (No AI) filled it first
    // (AMD desktop 2026-11-07: "No AI" stayed selected while the hybrid was in use)
    const hybId = (fx.ai.amd16.hybrids[0] || {}).id;
    info = {...fx.ai.amd16, choice: {mode: "none", live: "gemma4mtp:12b-n2", reviewer: null}};
    await p.evaluate(() => { aiPick = null; aiPickByUser = false; return loadAI(); }); await sleep(300);
    info = {...fx.ai.amd16, choice: {mode: "hybrid", live: "gemma4mtp:12b-n2", reviewer: "gemma4mtp:31b-n3-text"}};
    await p.evaluate(() => { aiPick = null; aiPickByUser = false; renderAI(); return loadAI(); }); await sleep(300);
    const selNow = await p.evaluate(() => { const c = document.querySelector('#aiPanel input[name="aiRow"]:checked');
      return c ? (c.dataset.aiHyb || c.dataset.aiRow || (c.dataset.aiNone !== undefined ? "none" : "")) : ""; });
    check("after a switch the selected row is the setup in use, not the previous answer's", !!hybId && selNow === "gemma4mtp:12b-n2|gemma4mtp:31b-n3-text",
      `${selNow} (hybrid ${hybId})`);
    // the text-only 31B (owner 2026-11-09): an 8 GB card shows its own measured figures (re-measured 2026-11-09)
    const row31 = async () => (await p.$$eval("#aiPanel label.r", es => es.map(e => (e.textContent.replace(/\s+/g, " ") + " <"
        + [...e.querySelectorAll("[data-spd]")].map(x => x.title).join(" | ") + ">").trim())))
      .find(t => /^Gemma 4 31B(?! live)/.test(t)) || "";
    info = fx.ai.nv8;
    await p.evaluate(() => { aiPick = null; aiPickByUser = false; return loadAI(); }); await sleep(300);
    const r31e = await row31();
    check("8 GB card: the 31B + MTP at 15 GPU layers, tested on 8 GB with no build note",
      /tested on 8 GB Nvidia/.test(r31e) && !/projector build/.test(r31e) && /15 GPU layers/.test(r31e) && /~20 msgs\/min/.test(r31e), r31e.slice(0, 300));
    await shot(p, "ai_picker_nv8_text31");
    // a 16 GB AMD card's hybrid saved before 2026-11-09 (the projector 31B reviewing, cap 40 found there): the reviewer is
    // the text build, the found cap carried as a floor (above the text build's estimate there)
    info = fx.ai.amd16old;
    await p.feed([stopped({ ai: { mode: "hybrid", live: "gemma4mtp:12b-n2", reviewer: "gemma4mtp:31b-n3-text", task: null, last: {} } })], 200);
    await p.evaluate(() => { aiPick = null; aiPickByUser = false; return loadAI(); }); await sleep(300);
    const r31c = await row31();
    const selOld = await p.evaluate(() => { const c = document.querySelector('#aiPanel input[name="aiRow"]:checked');
      return c ? (c.dataset.aiHyb || c.dataset.aiRow || "") : ""; });
    check("an old saved hybrid: the text 31B reviews (selected, in use) with the carried cap 40",
      fx.ai.amd16old.choice.reviewer === "gemma4mtp:31b-n3-text" && selOld === "gemma4mtp:12b-n2|gemma4mtp:31b-n3-text"
      && /40 GPU layers \(carried from gemma4mtp:31b-n3\)/.test(r31c), `${selOld} | ${r31c.slice(0, 300)}`);
    await shot(p, "ai_picker_amd16_carried");
    // W2 (sweep 2026-11-09): this machine's 11-08 file (the projector 31B at 26) read cold, nothing loaded: the text build
    // at 30 (the measured point is reached by the read itself), not the carried 26, and the page shows what will run
    info = fx.ai.nv12cold;
    await p.feed([stopped({ ai: { mode: "single", live: "gemma4mtp:31b-n3-text", reviewer: null, task: null, last: {} } })], 200);
    await p.evaluate(() => { aiPick = null; aiPickByUser = false; return loadAI(); }); await sleep(300);
    const r31cold = await row31();
    check("W2: a cold read on this card gives the text 31B 30 GPU layers, not the carried 26",
      fx.ai.nv12cold.choice.live === "gemma4mtp:31b-n3-text" && /30 GPU layers \(estimated from 11\.6 GB available\)/.test(r31cold)
      && !/26 GPU layers/.test(r31cold) && /~40 msgs\/min/.test(r31cold), r31cold.slice(0, 300));
    await shot(p, "ai_picker_nv12_cold");
    // the switch by re-read coverage: a machine whose own check gives keep-up 40 but 80% re-read only to 35: at 38 the 31B
    // keeps up, its re-reads thin, and the hybrid is recommended (it catches more there)
    info = fx.ai.nv12meas;
    await p.evaluate(() => { aiPick = null; aiPickByUser = false; return loadAI(); }); await sleep(300);
    await typeRate(38);
    const r31m = await row31();
    const hym = await p.$eval("#aiPanel .rowsAI", e => e.textContent.replace(/\s+/g, " "));
    const recHy = await p.$$eval("#aiPanel label.r", es => es.filter(e => e.querySelector(".chip.first")).map(e => e.textContent.replace(/\s+/g, " ").trim()));
    check("switch by re-read coverage: measured 31B at 38 msgs/min keeps up with re-reads thinning; the hybrid is recommended",
      /measured here 2026-11-09/.test(r31m) && /at 38 msgs\/min: keeps up/.test(r31m) && !/recommended/.test(r31m)
      && !!(await p.$('#aiPanel label.r span[data-spd="measured"]')) && recHy.length === 1 && /^12B \+ 31B Hybrid/.test(recHy[0])
      && !/full accuracy/.test(hym), `${r31m.slice(0, 260)} | ${recHy}`);
    const recTip = await p.getAttribute("#aiPanel label.r .chip.first", "title");
    check("…its tooltip says why (at this rate the hybrid catches more)", /at 38 msgs\/min the hybrid catches more than the best single model/.test(recTip || ""), recTip);
    await shot(p, "ai_picker_switch_by_reread");
    await p.evaluate(() => { aiTypedRate = null; try { localStorage.removeItem("cw_ai_rate"); } catch(e){} renderAI(); });
    info = fx.ai.amd16;
    await p.evaluate(() => { aiPick = null; aiPickByUser = false; return loadAI(); }); await sleep(300);
    // running: the panel folds to one line, changes wait for Stop
    await p.feed([status({ ai: { mode: "single", live: "gemma4mtp:31b-n3-text", task: null, last: {} } })], 300);
    check("while running the AI panel is one line: change it after Stop", /Change after Stop/.test(await p.textContent("#aiPanel"))
      && !(await p.$("#aiPanel label.r")) && !p.errs.length, p.errs.join("; "));
    await p.close();

    // the hybrid's Review button at the top of the watchlist
    info = fx.ai.nv12;
    p = await open(b, 1500, 950, { "/api/review": r => J(r, { ok: true, task: { kind: "review" } }),
      "/api/ai/cancel": r => J(r, { ok: true }), "/api/ai": r => J(r, info) });
    for (const batch of fx.batches) await p.feed(batch, 5);
    const H = fx.ai.status_hybrid;
    await p.feed([status({ ai: H })], 400);
    const box = await p.textContent("#reviewBox");
    const hs = (await p.textContent("#aiPanel .sum")).replace(/\s+/g, " ");
    check("hybrid: the AI bar gives the live model's rate and the reviewer's time per chatter in one line, not the single model's",
      /12B \+ 31B Hybrid · Live: Gemma 4 12B \((Estimated )?≥150 msgs\/min\) · Review: Gemma 4 31B \((Estimated )?~\d+ s per chatter\)/.test(hs) && !/~40|all lines|MTP/.test(hs), hs);
    const nb = H.review.b.chatters;
    check("hybrid: one Review button with the count of chatters with new content and its estimate",
      !(await p.$eval("#reviewBox", e => e.hidden)) && new RegExp(`Review ${nb}`).test(box) && /the live AI pauses meanwhile/.test(box) && /Review Watchlist only/.test(box), box);
    const na0 = H.review.a.chatters, split = (await p.textContent("#reviewBox [data-review-split]")).replace(/\s+/g, " ");
    check("polish: under the button, the checkbox and the count split into the Watchlist and the others with toxic messages (its own block, not in the header)",
      split === `${nb} chatter${nb === 1 ? "" : "s"} with new messages: ${na0} on the Watchlist + ${nb - na0} other${nb - na0 === 1 ? "" : "s"} with toxic messages`
      && await p.$eval("#reviewBox", e => e.parentElement.classList.contains("card") && e.querySelectorAll(".rvrow").length >= 2), split);
    await shot(p, "ai_review_button");
    const aiBar = (await p.textContent("#aiPanel .sum")).replace(/\s+/g, " ");
    check("running: the AI bar carries the session's log name (moved from the top bar) and says changes wait for Stop",
      /Logging: s\b/.test(aiBar) && /Change after Stop/.test(aiBar) && !(await p.$("#aiPanel .sum [data-ai-toggle]")), aiBar);
    await p.feed([{ type: "room", channel: (fx.batches.flat().find(r => r.type === "msg") || {}).channel || "cozy", msgs_per_min: 47, chatters: 37, first_timers_this_min: 0, first_timers_session: 4, notices: { items: [], counts: {} } },
      status({ ai: H, started_at: Date.now() - 12 * 60000 })], 400);
    await shot(p, "header_hybrid_live");
    await p.setViewportSize({ width: 390, height: 800 }); await sleep(200);
    await shot(p, "header_hybrid_live_phone");
    check("live hybrid at 390 px: no sideways scroll", await p.evaluate(() => document.documentElement.scrollWidth) <= 390);
    await p.feed([status({ ai: { ...H, task: null, last: { review: { at: "2026-11-07T23:00:00Z", reviewed: 9, chatters: 12, failed: 1,
      stopped_for_live: { left: 3 }, timings: { outage_s: 361 }, catchup_s: 40 } } }, started_at: Date.now() - 12 * 60000 })], 300);
    const fit = await p.$eval("#reviewBox", box => { const r = box.getBoundingClientRect();
      return [...box.querySelectorAll(".rvrow > *")].map(e => Math.max(Math.round(e.getBoundingClientRect().right - r.right), e.scrollWidth - e.clientWidth)).filter(d => d > 1); });
    check("audit polish F1: at 390 px every Review box line wraps inside the box (a long last-review line too)", !fit.length, JSON.stringify(fit));
    await shot(p, "review_box_phone");
    await p.setViewportSize({ width: 1500, height: 950 }); await sleep(200);
    await p.click("#reviewBox [data-review]"); await sleep(200);
    const rv = p.posts.find(x => x[0] === "review");
    check("Review POSTs the default scope (b: every flagged chatter, owner 2026-11-07)", !!rv && JSON.parse(rv[1]).scope === "b" && rv[2] === "1", JSON.stringify(rv));
    await p.click("#reviewBox [data-scope]"); await sleep(100);
    const na = H.review.a.chatters;
    check("the checkbox switches this press to the watchlist only", na === 0 ? /Nothing new to review/.test(await p.textContent("#reviewBox"))
      : new RegExp(`Review ${na}\\b`).test(await p.textContent("#reviewBox")), await p.textContent("#reviewBox"));
    if (na) { await p.click("#reviewBox [data-review]"); await sleep(200); }
    check("…and POSTs scope a", !na || p.posts.filter(x => x[0] === "review").some(x => JSON.parse(x[1]).scope === "a"));
    await p.feed([status({ ai: { ...H, task: { kind: "review", what: "reviewing", state: "reviewing chatter 3 of 12", done: 2, total: 12, eta_s: 240 } } })], 300);
    const prog = await p.textContent("#reviewBox");
    const bar = await p.textContent("#aiPanel");
    const one = "Reviewing chatter 3 of 12 · ~4 min left · live AI paused";
    check("while it runs: one counter, the same words on the watchlist and the AI bar, and a Cancel", prog.includes(one) && bar.includes(one)
      && !/Reviewing 2 of 12/.test(prog + bar) && !!(await p.$("#reviewBox [data-review-cancel]")), prog + " | " + bar);
    await shot(p, "ai_review_running");
    await p.click("#reviewBox [data-review-cancel]"); await sleep(200);
    check("Cancel POSTs", p.posts.some(x => x[0] === "ai/cancel"));
    await p.feed([status({ ai: { ...H, task: null, last: { review: { at: "2026-11-07T23:00:00Z", reviewed: 12, chatters: 12, failed: 0,
      timings: { outage_s: 361 }, catchup_s: 40 } } } })], 300);
    check("after: what it did and how fast the live AI caught up", /Last review: 12\/12 reviewed in 6 min; live caught up 40 s after/.test(await p.textContent("#reviewBox")),
      await p.textContent("#reviewBox"));
    await p.feed([status({ ai: { ...H, task: null, prefetch: { state: "cold" }, review: { ...H.review, swap: { s: 352, label: "last review here 2026-11-07" } } } })], 300);
    check("hybrid, models not cached: the first review's cost from the share is said on the AI bar", /First review slower: ~6 min to load the reviewer from disk/.test(await p.textContent("#aiPanel .sum")), await p.textContent("#aiPanel .sum"));
    await p.feed([status({ ai: { ...H, task: null, prefetch: { state: "ready", at: "2026-11-07T23:30:00Z" } } })], 300);
    check("hybrid, both models in memory: the AI bar says so, with the swap estimate (once: not again by the Review button)", /Models in memory · Swap ~\d+ s/.test(await p.textContent("#aiPanel .sum")) && !/in memory/.test(await p.textContent("#reviewBox")));
    await shot(p, "ai_review_quick_swap");
    await p.feed([status({ ai: { ...H, mode: "single" } })], 200);
    check("one model: no Review button (the live model already reads everything)", await p.$eval("#reviewBox", e => e.hidden) && !p.errs.length,
      p.errs.join("; ") + " | " + await p.evaluate(() => JSON.stringify({m: (aiStatus || {}).mode, hidden: document.getElementById("reviewBox").hidden})));
    await p.close();
  }

  // ---------------------------------------------------------------- audit 2026-11-10 (frontend_env.md FE-n, queue_llm QL5)
  {
    const J = (r, body, st = 200) => r.fulfill({ status: st, contentType: "application/json", body: JSON.stringify(body) });
    const now = Date.now();
    // FE-1: a second ago() (seconds) had replaced ago(epoch ms): every card age read ~498,000,000 h
    p = await open(b, 1500, 950, { "/api/profile": r => J(r, { ...Object.values(fx.profiles)[0], channel: "cozy", user_id: "u1",
      person_read: { label: "mild", rationale: "jabs", n: 3, ts: Date.now() - 5 * 60000 } }) });
    const card = { user_id: "u1", display: "someone", level: "act", why: ["slur or dehumanising language"], since: now - 2 * 60000,
      last_seen: now - 2 * 60000, quote: "q", first: false, review: null, role: null };
    const qm = o => ({ reads: 1, lines: 2, behind_s: 75, behind_hi_s: 75, catchup_s: null, aged_out: 0, retrying: null, held: false,
      finishing: false, open: true, ...o });
    await p.feed([status({ llm_queue: qm({ catchup_state: "measuring" }) }), msg("cozy", "u1", "someone", "hello", { ts: now - 1000 }),
      { type: "room", channel: "cozy", msgs_per_min: 10, chatters: 3, first_timers_session: 1,
        notices: { items: [card], counts: { act: 1, watch: 0 }, more: 0 },
        notices_earlier: [{ ...card, user_id: "u2", display: "earlier", level: "watch", last_seen: now - 3 * 3600e3 }] }], 400);
    const age = await p.$eval("#heads .hu .t", e => e.textContent);
    const early = await p.$eval("#earlier .erow .t", e => e.textContent).catch(() => "(none)");
    check("FE-1: a watchlist card listed 2 min ago says '2 min ago'", age === "2 min ago", age);
    check("FE-1: an 'Earlier this session' row last seen 3 h ago says '3 h ago'", early === "3 h ago", early);
    // QL5: no call timed yet is "measuring speed", not "falling behind" (the server's catchup_state)
    const bm = { txt: (await p.textContent("#strip")).replace(/\s+/g, " "), title: await p.getAttribute("#strip [data-ai-buffer]", "title") };
    check("QL5: no catch-up time because no speed is known yet reads 'measuring speed', not 'falling behind' (ages in seconds)",
      /3 ?AI buffer · 75 s behind · measuring speed/.test(bm.txt) && !/falling behind/.test(bm.txt + bm.title) && /still being measured/.test(bm.title), JSON.stringify(bm));
    await shot(p, "fe1_ages_ql5_measuring");
    await p.feed([status({ llm_queue: qm({ catchup_state: "behind" }) })], 300);
    check("QL5: catchup_state 'behind' still reads 'falling behind'", /75 s behind · falling behind/.test(await p.textContent("#strip")), await p.textContent("#strip"));
    await p.evaluate(() => openCard("u1", "cozy")); await sleep(400);
    const rd = await p.$eval("#drawer .box.read .note", e => e.textContent).catch(() => "(none)");
    check("FE-1: the user card's LLM read says when: '5 min ago'", /, 5 min ago$/.test(rd.trim()), rd.trim());
    await shot(p, "fe1_card_read_age");
    check("no page errors in the FE-1 / QL5 checks", !p.errs.length, p.errs.join("; "));
    await p.close();

    // FE-2: a status does not redraw the AI panel under typing or a press; it redraws once that ends
    const H = fx.ai.status_hybrid;
    let info = JSON.parse(JSON.stringify(fx.ai.nv12));
    info.hybrids[0].quality = { ...info.hybrids[0].quality, b_over_no_ai: 70, a_over_no_ai: 60 };     // FE-3: not the 12B + 31B's 62 / 54
    info.last = { ...(info.last || {}), speed: { model: "gemma4mtp:12b-n2", keepup: 150, keepup_plus: true, person_s: 1.2, aborts: 0, calls: 8 } };   // the check pressed below
    let reviewReply = [200, { ok: true, task: { kind: "review" } }], cancelReply = [200, { ok: true }];
    p = await open(b, 1500, 950, { "/api/ai/speed": r => J(r, { ok: true, task: { kind: "speed" } }), "/api/ai/cancel": r => J(r, cancelReply[1], cancelReply[0]),
      "/api/review": r => J(r, reviewReply[1], reviewReply[0]), "/api/ai": r => J(r, info) });
    const idle = gb => status({ running: false, connected: false, channels: [], session: null,
      ai: { ...H, task: null, prefetch: { state: "reading", done_gb: gb, total_gb: 30 } } });
    await p.feed([idle(1)], 600);
    await p.click("#aiPanel [data-ai-typed]");
    await p.keyboard.type("30");
    await p.feed([idle(2)], 300);
    const ty = await p.evaluate(() => { const el = document.querySelector("#aiPanel [data-ai-typed]"); return { focused: document.activeElement === el, value: el.value }; });
    check("FE-2: a chat rate being typed survives the next status (value and focus)", ty.focused && ty.value === "30", JSON.stringify(ty));
    await p.evaluate(() => document.activeElement.blur()); await sleep(200);
    check("FE-2: …and the panel catches up once the box is left", /Loading the reviewer into memory: 2 of 30 GB/.test(await p.textContent("#aiPanel .sum")),
      (await p.textContent("#aiPanel .sum")).replace(/\s+/g, " "));
    const press = async (sel, during) => { const bx = await (await p.$(sel)).boundingBox();
      await p.mouse.move(bx.x + 5, bx.y + 5); await p.mouse.down(); await p.feed(during, 150); await p.mouse.up(); await sleep(300); };
    let n0 = p.posts.filter(x => x[0] === "ai/speed").length;
    await press("#aiPanel [data-ai-speed]", [idle(3)]);
    check("FE-2: a press on Check Speed with a status between down and up sends its POST",
      p.posts.filter(x => x[0] === "ai/speed").length - n0 === 1 && /3 of 30 GB/.test(await p.textContent("#aiPanel .sum")),
      `POSTs ${p.posts.filter(x => x[0] === "ai/speed").length - n0}`);
    const sp = await p.$eval("#aiPanel [data-ai-speed]", e => e.dataset.aiSpeed);
    await p.focus(`#aiPanel [data-ai-speed="${sp}"]`);
    await p.feed([idle(4)], 300);
    check("FE-2: keyboard focus on a button stays on it when a status redraws the panel",
      await p.evaluate(s => (document.activeElement.dataset || {}).aiSpeed === s && /4 of 30 GB/.test(document.getElementById("aiPanel").textContent), sp));
    // FE-10 (round 2): type a rate, then click Check speed straight away (no blur first): the box's change event fires on
    // that mousedown, and its redraw must wait for the press
    await p.focus("#aiPanel [data-ai-typed]"); await p.keyboard.press("Control+A"); await p.keyboard.type("45");
    n0 = p.posts.filter(x => x[0] === "ai/speed").length;
    { const bx = await (await p.$("#aiPanel [data-ai-speed]")).boundingBox();
      await p.mouse.move(bx.x + 5, bx.y + 5); await p.mouse.down(); await sleep(80); await p.mouse.up(); await sleep(400); }
    const t10 = { posts: p.posts.filter(x => x[0] === "ai/speed").length - n0, rate: await p.$eval("#aiPanel [data-ai-typed]", e => e.value) };
    check("FE-10: type a chat rate, then click Check speed at once: its POST is sent and the rate kept", t10.posts === 1 && t10.rate === "45", JSON.stringify(t10));
    await p.feed([idle(5)], 300);
    // FE-11 (round 2): focus left on a closed GPU / RAM list after a pick, or on the rate box after Enter, holds nothing
    { const sel = await p.$("#aiPanel select[data-ai-ov]");
      const v = await sel.evaluate(e => [...e.options].map(o => o.value).find(o => o && o !== e.value));
      await sel.focus(); await sel.selectOption(v); await sleep(500); }
    const f11a = { focus: await p.evaluate(() => document.activeElement.matches("#aiPanel select[data-ai-ov]")) };
    await p.feed([idle(6)], 300);
    f11a.sum = (await p.textContent("#aiPanel .sum")).replace(/\s+/g, " ");
    check("FE-11: the panel keeps updating while a closed GPU / RAM list still has focus after a pick", f11a.focus && /6 of 30 GB/.test(f11a.sum), JSON.stringify(f11a));
    await p.focus("#aiPanel [data-ai-typed]"); await p.keyboard.press("Control+A"); await p.keyboard.type("50"); await p.keyboard.press("Enter"); await sleep(200);
    await p.feed([idle(7)], 300);
    const f11b = await p.evaluate(() => { const el = document.querySelector("#aiPanel [data-ai-typed]");
      return { focused: document.activeElement === el, value: el.value, sum: document.querySelector("#aiPanel .sum").textContent.replace(/\s+/g, " ") }; });
    check("FE-11: …and while the rate box keeps focus after Enter (the rate committed and kept)", f11b.focused && f11b.value === "50" && /7 of 30 GB/.test(f11b.sum), JSON.stringify(f11b));
    // FE-13 (round 3): the redraw that put focus back in the rate box selects its text, so the next key replaces it
    await p.keyboard.type("9"); await sleep(100);
    const f13 = await p.$eval("#aiPanel [data-ai-typed]", e => e.value);
    check("FE-13: after a redraw kept focus in the rate box, the next key replaces the rate (not '950')", f13 === "9", f13);
    await p.keyboard.press("Control+A"); await p.keyboard.type("50"); await p.keyboard.press("Enter"); await sleep(200);
    // FE-12 (round 3): a GPU / RAM list opened and closed with no pick holds the panel for at most 8 s
    { const sel = await p.$("#aiPanel select[data-ai-ov]"), bx = await sel.boundingBox();
      await p.mouse.click(bx.x + 5, bx.y + 5); await sleep(150); await p.mouse.click(bx.x + 5, bx.y + 5); await sleep(100); }   // opened and closed by a second click: the page hears no change, no Escape
    await p.feed([idle(8)], 300); await sleep(8500);
    const f12 = { focus: await p.evaluate(() => document.activeElement.matches("#aiPanel select[data-ai-ov]")),
                  sum: (await p.textContent("#aiPanel .sum")).replace(/\s+/g, " ") };
    check("FE-12: a list opened and closed without a pick does not freeze the panel (caught up within 8 s)", /8 of 30 GB/.test(f12.sum), JSON.stringify(f12));
    await p.evaluate(() => document.activeElement.blur()); await sleep(200);
    // FE-14 (round 4): a list opened straight after typing a rate keeps its hold (the rate box's focusout / change
    // must not end it), so neither the rate's own redraw nor the next status rebuilds the open list
    await p.focus("#aiPanel [data-ai-typed]"); await p.keyboard.press("Control+A"); await p.keyboard.type("55");
    await p.$eval("#aiPanel select[data-ai-ov]", e => { e.__fe14 = 1; });
    { const bx = await (await p.$("#aiPanel select[data-ai-ov]")).boundingBox();
      await p.mouse.move(bx.x + 5, bx.y + 5); await p.mouse.down(); await sleep(80); await p.mouse.up(); await sleep(200); }
    await p.feed([idle(9)], 300);
    const f14 = await p.evaluate(() => { const el = document.querySelector("#aiPanel select[data-ai-ov]");
      return { same: el.__fe14 === 1, focus: document.activeElement === el }; });
    check("FE-14: a list opened right after typing a rate is not rebuilt by the rate's redraw or the next status", f14.same && f14.focus, JSON.stringify(f14));
    await p.keyboard.press("Escape"); await sleep(8600);
    await p.evaluate(() => document.activeElement.blur()); await sleep(100);
    // the Review box: a press across a status that changes its count; the pair's own quality in its tooltips (FE-3)
    for (const batch of fx.batches.slice(0, 5)) await p.feed(batch, 5);
    const run = (k, o = {}) => status({ ai: { ...H, task: null, review: { ...H.review, b: { ...H.review.b, chatters: k, seconds: 60 * k } }, ...o } });
    await p.feed([run(3)], 400);
    n0 = p.posts.filter(x => x[0] === "review").length;
    await press("#reviewBox [data-review]", [run(4)]);
    check("FE-2: a press on Review while a status changes its count sends its POST, then the box shows the new count",
      p.posts.filter(x => x[0] === "review").length - n0 === 1 && /Review 4/.test(await p.textContent("#reviewBox")), await p.textContent("#reviewBox"));
    const rt = { btn: await p.getAttribute("#reviewBox [data-review]", "title"), lab: await p.getAttribute("#reviewBox label", "title") };
    // owner 2026-11-10: an (i) beside the checkbox: hover shows the text (its title), a click keeps it open, a second click
    // or Escape folds it; the click never ticks the checkbox, and a status redraw keeps it open
    const ib = { title: await p.getAttribute("#reviewBox [data-info='scope']", "title"), checked0: await p.$eval("#reviewBox [data-scope]", e => e.checked) };
    await p.click("#reviewBox [data-info='scope']"); await sleep(150);
    await p.feed([run(4)], 300);
    ib.open = (await p.textContent("#reviewBox [data-info-pop='scope']").catch(() => "")).trim();
    await shot(p, "review_info_open");
    ib.exp = await p.getAttribute("#reviewBox [data-info='scope']", "aria-expanded");
    ib.checked1 = await p.$eval("#reviewBox [data-scope]", e => e.checked);
    await p.click("#reviewBox [data-info='scope']"); await sleep(150);
    ib.closed = !(await p.$("#reviewBox [data-info-pop]"));
    await p.click("#reviewBox [data-info='scope']"); await sleep(150);
    await p.keyboard.press("Escape"); await sleep(150);
    ib.esc = !(await p.$("#reviewBox [data-info-pop]"));
    await p.click("#reviewBox [data-info='review']"); await sleep(150);
    const rvi = (await p.textContent("#reviewBox [data-info-pop='review']").catch(() => "")).trim();
    await p.click("#reviewBox [data-info='review']"); await sleep(150);
    check("the Review button's (i): what a review does, in plain words", /^The stronger model reads each of these 4 chatters as a whole/.test(rvi)
      && /chat waits in the queue and is caught up after\.$/.test(rvi) && !(await p.$("#reviewBox [data-info-pop]")), rvi);
    check("the (i) beside Review Watchlist only: hover text, click opens and survives a status, click or Escape folds, never ticks the box",
      ib.title === rt.lab && ib.open === rt.lab && ib.exp === "true" && ib.checked1 === ib.checked0 && ib.closed && ib.esc, JSON.stringify(ib));
    check("FE-3: the Review tooltips give the pair's own quality from /api/ai, not a fixed +62% / +54%",
      /\+70% over No AI reviewing all flagged chatters, \+60% the Watchlist only/.test(rt.btn) && /^Fewer chatters and a shorter pause; accuracy \+60% instead of \+70%\.$/.test(rt.lab)
      && !/62%|54%/.test(rt.btn + rt.lab), JSON.stringify(rt));
    // FE-5: a refused Review: the error stays across a redraw, never as the box's tooltip, and goes after its time
    reviewReply = [409, { error: "a speed check is running" }];
    await p.click("#reviewBox [data-review]"); await sleep(300);
    await p.feed([run(5)], 300);
    const e1 = { txt: await p.textContent("#reviewBox"), title: await p.getAttribute("#reviewBox", "title") };
    await shot(p, "fe5_review_error");
    await p.evaluate(() => { reviewErr.at -= 60000; });
    await p.feed([run(5)], 300);
    const e2 = { txt: await p.textContent("#reviewBox"), title: await p.getAttribute("#reviewBox", "title") };
    check("FE-5: a refused Review shows its error through the next status, never as the box's tooltip, then it goes",
      /Review 5/.test(e1.txt) && /a speed check is running/.test(e1.txt) && !e1.title && !/speed check/.test(e2.txt) && !e2.title, JSON.stringify([e1, e2]));
    // FE-5: a refused Cancel, in the Review box and on the AI bar (folded while running: no action row)
    cancelReply = [409, { error: "nothing to cancel" }];
    await p.feed([run(5, { task: { kind: "review", what: "reviewing", state: "reviewing chatter 1 of 5", done: 0, total: 5, eta_s: 200 } })], 300);
    await p.click("#reviewBox [data-review-cancel]"); await sleep(300);
    await p.feed([run(5, { task: { kind: "review", what: "reviewing", state: "reviewing chatter 2 of 5", done: 1, total: 5, eta_s: 150 } })], 300);
    const c1 = await p.textContent("#reviewBox");
    await p.click("#aiPanel [data-ai-cancel]"); await sleep(300);
    await p.feed([run(5, { task: { kind: "review", what: "reviewing", state: "reviewing chatter 3 of 5", done: 2, total: 5, eta_s: 100 } })], 300);
    const c2 = (await p.textContent("#aiPanel .sum")).replace(/\s+/g, " ");
    check("FE-5: a refused Cancel says so, in the Review box and on the AI bar", /nothing to cancel/.test(c1) && /Reviewing chatter 2 of 5/.test(c1)
      && /nothing to cancel/.test(c2) && /Change after Stop/.test(c2), JSON.stringify([c1, c2]));
    await shot(p, "fe5_cancel_errors");
    await p.setViewportSize({ width: 390, height: 800 }); await sleep(200);
    await shot(p, "fe5_cancel_errors_phone");
    check("FE-5: the errors fit at 390 px (no sideways scroll)", await p.evaluate(() => document.documentElement.scrollWidth) <= 390);
    check("no page errors in the FE-2 / FE-3 / FE-5 checks", !p.errs.length, p.errs.join("; "));
    await p.close();

    // FE-4: a failed first read of the AI settings is tried again with the status and recovers without a reload
    let gets = 0;
    p = await open(b, 1500, 950, { "/api/ai": r => (++gets === 1 ? J(r, { error: "busy" }, 500) : J(r, fx.ai.nv12)) });
    const st0 = status({ running: false, connected: false, channels: [], stopped_at: null, session: null,
      ai: { mode: "single", live: "gemma4mtp:31b-n3-text", reviewer: null, task: null, last: {}, scope_default: "b" } });
    await p.feed([st0], 400);
    const f1 = (await p.textContent("#aiPanel")).replace(/\s+/g, " ");
    for (let i = 0; i < 6; i++) await p.feed([st0], 1000);
    const f2 = (await p.textContent("#aiPanel")).replace(/\s+/g, " ");
    check("FE-4: a failed first GET /api/ai says it tries again, then the picker appears without a reload",
      /could not be read: HTTP 500; trying again/.test(f1) && gets === 2 && !/could not be read/.test(f2) && !!(await p.$("#aiPanel label.r")), `GETs ${gets}; ${f1.slice(0, 90)}`);
    await shot(p, "fe4_ai_recovered");
    await p.close();
  }

  await b.close();
  console.log(failed ? `${failed} check(s) FAILED` : "all checks passed");
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
