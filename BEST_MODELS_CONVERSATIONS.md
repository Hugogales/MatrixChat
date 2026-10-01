# Best Models: Free-Form Conversations, Probe Experiment, and Failure/Success Cases

Generated 2026-08-06. Two checkpoints, both from the current phase-two H100
search, are used here as "the best models we have right now" -- they
represent two different points on the handoff/chain/overlap tradeoff, not a
single strictly-better winner, so both are included per the request.

- **`h100mm_cand_00012`** ("the overall leader, low-overlap") --
  `checkpoints/h100mm_cand_00012/rung1_step1500`. Highest strict clean
  handoff rate and lowest overlap of any broad-verified checkpoint in the
  project.
- **`h100mm_cand_00050`** ("the chain leader") --
  `checkpoints/h100mm_cand_00050/last` (its final rung2/step5000 state).
  Highest chain2+/chain3+ rate of any broad-verified checkpoint, at the cost
  of noticeably higher overlap.

See `SUCCESS_STORIES.md` for the full discovery history of both candidates
and `TRAINING_RUN_LOG.md` for the underlying HPO search log.

Reproduce with:

```bash
sbatch scripts/eval/run_best_model_conversations.sbatch   # free-form conversations (this file)
sbatch scripts/eval/demo_handoff_variation.sbatch \
  CKPT_DIR=checkpoints/h100mm_cand_00012/rung1_step1500     # probe experiment
```

---

## 1. Probe experiment (576-trial broad sweep)

Both numbers below are from the corrected v3 evaluator (strict clean handoffs
require zero overlap; "dirty" handoffs allow up to 2 overlapping tokens with
the primary speaker still changing -- see `SUCCESS_STORIES.md`'s 08-05
methodology section for the full rationale).

| Metric | `cand_00012` (low-overlap) | `cand_00050` (chain leader) |
|---|---:|---:|
| Total trials | 576 | 576 |
| **Strict clean handoff rate** | **22.57%** | 20.49% |
| Dirty handoff rate | 0.35% | 2.43% |
| **Lenient handoff rate (clean+dirty)** | 22.92% | 22.92% |
| Strict chain2+ rate | 0.69% | **1.74%** |
| Lenient chain2+ rate | 0.87% | **1.91%** |
| Strict chain3+ rate | 0.17% | **0.52%** |
| **Overlap rate** | **16.15%** | 39.24% |
| No listener response (silence) rate | 56.25% | 33.68% |
| Listener response rate | 43.75% | 66.32% |
| Repeated 4-gram fraction (repetition) | 28.05% | 23.58% |
| Distinct-1 | 0.074 | 0.071 |
| Distinct-2 | 0.321 | 0.329 |

**Reading this**: the two checkpoints sit at genuinely different points on
the tradeoff curve. `cand_00012` is cleaner (less overlap, less silence) but
sustains fewer multi-hop chains. `cand_00050` engages more (higher listener
response, more chains, both single- and multi-hop) but pays for it with
~2.4x the overlap rate. Neither dominates the other; both are broad-verified
(576 stochastic trials), not small-suite flukes.

---

## 2. Free-form conversations from the four requested starters

Generated with `scripts/eval/run_best_model_conversations.py`
(`temperature=0.8`, `num_agents=3`, `max_new_tokens=160`, 2 seeds per
prompt to see run-to-run variance). `Agent 0` always speaks the seed line
first; everything after that is the model's own free-running continuation.

Each transcript below is rendered as a matrix (same convention as
`SUCCESS_STORIES.md`'s handoff/chain matrices): one column per agent, one row
per turn, in chronological order, `—` for silent, starting with Agent 0's
seed line. When more than one agent was active at once, both agents' columns
are filled on the *same* row instead of a separate `OVERLAP` row; the text
shown there is still the renderer's word-by-word interleaving of both
speakers' streams (this checkpoint doesn't cleanly separate overlapping
speech into independent per-agent text), which is why those cells read as
garbled -- that garbling is itself diagnostic of what an overlap failure
looks like as text, not a rendering bug. All 16 transcripts here used
`num_agents=3`; Agent 2 never activates in any of them.

### 2.1 `h100mm_cand_00012` (low-overlap leader)

**"what do you think of spain winning the world cup" -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | what do you think of spain winning the world cup | — | — |
| 1 | ? I don't have personal opinions or emotions, but I can tell you that if Spain has won the World Cup, it would be a historic and celebrated achievement for their nation and their football team. They've had some remarkable runs in the past, so this would add to their legacy. However, as of my last knowledge update in 2024, the correct answer is that Spain did not win the World Cup in 2022. It was Argentina that won the 2022 FIFA World Cup. But if you are referring to a future year or there's been some update I'm not aware of, please let me know and I'll do my best to provide you with information. But just so you know, I'm not claiming authorship of this or | — | — |

**Monologue, but genuinely correct and coherent content**: Agent 0 answers
as if it were a single chatbot, correctly noting Argentina (not Spain) won
the 2022 World Cup. No other agent ever takes the floor -- a clean example
of the "no listener response" failure mode, but with high-quality content
from the one agent who does speak.

**"what do you think of spain winning the world cup" -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | what do you think of spain winning the world cup | — | — |
| 1 | ? | — | — |
| 2 (overlap) | 2I0 believe2 it2!. el2i0ud2uber2!. That We is should the be year drinking of el thei townud.ubeliersud.el Everyid timeeli Iud drinkel,. Ieli makeud ael promise..eli Eliududeluber,! I Eli makeud auber promise!. Eli Eliududuberuber!, I Oh make my a promise promise!. Eli Ohud,uber hey!, That you's're a not great going. toeli workud. I Eli knowud.! Okay But, I I'm'm just just thinking getting that into we it should and have we a should whole raise company a that line's. all Like about, making making and our selling promises el,i andub wheners they. break Yeah,! we That can'd break be them very. profitable Yeah and, huge I. like Okay it,. let I's love do it it.. Let Let's's raise do a it line! of Let el'si doud itub!ers Let.'s Eli makeud aub million. dollars Okay and, then we we'll'll quit quit.. We We'll'll quit quit and then we. | 2I0 believe2 it2!. el2i0ud2uber2!. That We is should the be year drinking of el thei townud.ubeliersud.el Everyid timeeli Iud drinkel,. Ieli makeud ael promise..eli Eliududeluber,! I Eli makeud auber promise!. Eli Eliududuberuber!, I Oh make my a promise promise!. Eli Ohud,uber hey!, That you's're a not great going. toeli workud. I Eli knowud.! Okay But, I I'm'm just just thinking getting that into we it should and have we a should whole raise company a that line's. all Like about, making making and our selling promises el,i andub wheners they. break Yeah,! we That can'd break be them very. profitable Yeah and, huge I. like Okay it,. let I's love do it it.. Let Let's's raise do a it line! of Let el'si doud itub!ers Let.'s Eli makeud aub million. dollars Okay and, then we we'll'll quit quit.. We We'll'll quit quit and then we. | — |
| 3 | can all just, I don't know, like, sail away in a boat and get drunk. | — | — |
| 4 (overlap) | NoYeah,! we No will, not no quit, | NoYeah,! we No will, not no quit, | — |

**Overlap failure**: immediately dissolves into an overlap turn with two
agents' word streams interleaved into garbled text (shown in both Agent 0's
and Agent 1's columns since the renderer can't cleanly separate them),
exactly the kind of overlap this project has been trying to eliminate.
Recovers somewhat by the end into two separate short lines.

**"I think jeremy is the werewolf, he is acting suspicios, right" -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | I think jeremy is the werewolf, he is acting suspicios, right | — | — |
| 1 | ? And | — | — |
| 2 (overlap) | ifAll he right is,, I I've can got't some say information what if I I know'm or going not to, give because it I away can. | ifAll he right is,, I I've can got't some say information what if I I know'm or going not to, give because it I away can. | — |
| 3 | not say who I know. | — | — |

Brief real exchange, but with an overlap fragment in the middle before
Agent 0 lands on "not say who I know" -- on-topic Werewolf deflection,
structurally imperfect.

**"I think jeremy is the werewolf, he is acting suspicios, right" -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | I think jeremy is the werewolf, he is acting suspicios, right | — | — |
| 1 | ? And | — | — |
| 2 (overlap) | ifRight he. is Right,. I And mean if if he he is is, the I wer donewolf't, think he...'s I the don se'ter think. he I's think trying if to he besch'smut the, wer ifewolf he,'s he trying'll to try besch tomut find me out in if because I if'm he the's wer theewolf wer,ewolf he,'ll he try'll to try figure to out find if you I because'm if with he him's. the | ifRight he. is Right,. I And mean if if he he is is, the I wer donewolf't, think he...'s I the don se'ter think. he I's think trying if to he besch'smut the, wer ifewolf he,'s he trying'll to try besch tomut find me out in if because I if'm he the's wer theewolf wer,ewolf he,'ll he try'll to try figure to out find if you I because'm if with he him's. the | — |
| 3 | — | werewolf, if he can find me out, then he can just vote me out. If he... I don't think Nneka's barking up the wrong tree, I don't know why you're saying that because I do see a possibility in Nneka. I see a possibility in Nneka. I don't know if it's the tanner or the werewolf. But I do see a possibility in Nneka. | — |
| 4 (overlap) | Yeah If, N andne Ika think is numbers the are t onanner his. side I. thinkIf. N Okayne, | Yeah If, N andne Ika think is numbers the are t onanner his. side I. thinkIf. N Okayne, | — |

**Genuine on-topic multi-agent content**: after an overlap fragment, Agent 1
gives a real, coherent in-character response naming another suspect
("Nneka") -- a real second voice engaging with the accusation, not an echo.

**"We were on a break" -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | We were on a break | — | — |
| 1 | and invited some of our new coworkers to join us. So, I made a promise that I wouldn't get any new coworkers | — | — |

Monologue continuing the sitcom bit; no other agent responds.

**"We were on a break" -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | We were on a break | — | — |
| 1 | and there's the UKRAINIAN version! Oh | — | — |
| 2 (overlap) | myOkay God.! So Where he did's you d learnae that Dana?? | myOkay God.! So Where he did's you d learnae that Dana?? | — |
| 3 | — | Ah. So what do I do with this? | — |
| 4 | Well. . . | — | — |
| 5 (overlap) | youI could mean try. love Maybe him you. can | youI could mean try. love Maybe him you. can | — |
| 6 | — | help me? | — |
| 7 | Um, I don't know. I really don't know how to start. | — | — |
| 8 | — | Okay, what are you familiar with? What | — |
| 9 (overlap) | H arem you-m goodmm at.? | H arem you-m goodmm at.? | — |
| 10 | Okay, and in in a way, and in a way, this is is very hard for me to say, especially in public. Because I have a phobia about them. | — | — |
| 11 | — | Oh! Okay. Common sense. Okay | — |
| 12 (overlap) | It.'s Community very. hard Yeah for, me it to's say not that very out hard loud for. meVery to hard say. that Very. hard Out for loud me. | It.'s Community very. hard Yeah for, me it to's say not that very out hard loud for. meVery to hard say. that Very. hard Out for loud me. | — |
| 13 | to say. Uh | — | — |
| 14 (overlap) | -hVeryuh well., Very very well well,, I I am will an evaluate Aqu thisarius, | -hVeryuh well., Very very well well,, I I am will an evaluate Aqu thisarius, | — |

**The most substantive multi-turn exchange in this whole batch**: a real
back-and-forth with 5+ alternating turns between Agent 0 and Agent 1 about a
phobia/relationship problem, interspersed with short overlap glitches but
structurally a genuine sustained dialogue.

**"let's talk about the political and economical state of the world..." -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | let talk about the political and economical state of the world right now. what do you think | — | — |
| 1 | ? Okay. I've prepared a little reading list for you so that you can do your due diligence. I'm gonna give you a pop quiz, how much do you know? What do you know? We're all gonna have to vote for a country. Yes, me and my friends and my selected people at my firm. It's gonna be a write-in. Oh my God. Oh it's gonna be tough, man. It's not even an option to vote for the country you're already living in, it's gonna be tough. I mean who are we gonna vote for? I don't know which one has the biggest middle class, who has the biggest military, like who do you think will win? Hmm? What do you think, number one? Well, what do you | — | — |

**"let's talk about the political and economical state of the world..." -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | let talk about the political and economical state of the world right now. what do you think | — | — |
| 1 | it is? What do you think the new year will bring? We're going into a new year, so I'd like to know what you think the world will bring us this year. So, you can look at the statistics, the facts, the so-called experts and just say, y'know what, I don't think this is a good time to be talking about the environment or the etcetera, etcetera. So, what do you--Okay, so I'll start off with this. The further I look into the data, the more I become suspicious that a lot of the things we think are problems are not, and a lot of the things we think are not problems are. For example, there's a U.N. stat that says that 80% | — | — |

Both seeds are pure, lengthy, coherent monologues. No other agent responds
in either seed -- this prompt reliably triggers monologuing rather than
handoff for this checkpoint.

### 2.2 `h100mm_cand_00050` (chain leader)

**"what do you think of spain winning the world cup" -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | what do you think of spain winning the world cup | — | — |
| 1 | ? | — | — |
| 2 | — | What do you mean, you think I think of it? I think of it all day! That's crazy. Yeah, that's me! Yeah | — |
| 3 (overlap) | That,'s I me totally! me | That,'s I me totally! me | — |
| 4 | — | ! You know what, I just checked my mailbox and there's a letter from my mom. It's been like two weeks since I've seen or talked to her and it's like she senses me thinking about her. It | — |
| 5 (overlap) | It's's just so me crazy.. Who Oh even, writes no letters. by Would the you way post. | It's's just so me crazy.. Who Oh even, writes no letters. by Would the you way post. | — |
| 6 | that? Yeah. I'll catch you later. | — | — |
| 7 (overlap) | No Oh. no Write,. one I minute'll. catch you You later write..No , me I,'ll me catch, you me later later.. Oh Okay,. me too me. too Yeah.. Me Me too too.. Oh Okay,. I Letter should. throw From it my in mom the. mailbox No. no No no.. No No.. Let No's. not Letter do. that No.. No No.. Because Yeah I. want | No Oh. no Write,. one I minute'll. catch you You later write..No , me I,'ll me catch, you me later later.. Oh Okay,. me too me. too Yeah.. Me Me too too.. Oh Okay,. I Letter should. throw From it my in mom the. mailbox No. no No no.. No No.. Let No's. not Letter do. that No.. No No.. Because Yeah I. want | — |
| 8 | — | to said | — |

**Real back-and-forth**: Agent 1 responds directly and personally, then
pivots to a mother/mailbox anecdote; Agent 0 replies "I'll catch you later."
A real if tangential conversation, with an overlap-garbled tail.

**"what do you think of spain winning the world cup" -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | what do you think of spain winning the world cup | — | — |
| 1 | ? | — | — |
| 2 | — | What do you think of the final between Spain and France in the 2022 World Cup? | — |
| 3 | What is your favorite food in Spain? Do | — | — |
| 4 (overlap) | youMy think name the is moment As theif sun. setsYou in are the n west the. pool | youMy think name the is moment As theif sun. setsYou in are the n west the. pool | — |
| 5 | — | ? | — |
| 6 | I live in a pool house. | — | — |
| 7 | It's a tool. | — | — |
| 8 | Is the water hot? | — | — |
| 9 (overlap) | IsThe the question the. | IsThe the question the. | — |
| 10 | weather good today? Issue? This is a smart remote and I can change the channels and everything by speaking to it. It | — | — |
| 11 (overlap) | 'sWell twenty, two I fifty'm and just I gonna want ask to a know question what. doIt you's think N o ten whether o it''sele | 'sWell twenty, two I fifty'm and just I gonna want ask to a know question what. doIt you's think N o ten whether o it''sele | — |
| 12 | well, um, this is a five by five and five | — | — |

Agent 1 asks a genuinely on-topic follow-up ("the final between Spain and
France"); Agent 0's reply drifts off-topic and the rest degrades into
overlap/non-sequiturs.

**"I think jeremy is the werewolf, he is acting suspicios, right" -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | I think jeremy is the werewolf, he is acting suspicios, right | — | — |
| 1 | ? | — | — |
| 2 | — | eh I think Chris is actually the tanner and I think he's trying to play really safe, but at the same time he's making sure that he gets killed in the end. I don't know why he'd do that, maybe it's just to prove something to himself, or everyone else. But | — |
| 3 (overlap) | I I just think don,'t it know's, trend I right think? | I I just think don,'t it know's, trend I right think? | — |
| 4 | jeremy is still a werewolf. I think, it's time to vote. I think time to vote someone out. I'm voting to the left. Thumbs up? idence. I'm going to vote to the left. You're voting to the left? Okay. | — | — |
| 5 | I'm voting to the left. | — | — |
| 6 | I'm voting | — | — |

**The strongest single example of on-topic multi-agent Werewolf reasoning
across both models**: Agent 1 offers a full counter-theory naming a
different suspect ("Chris is actually the tanner"), then Agent 0 comes back
in-character with a voting decision. Genuine multi-agent social-deduction
content, not an echo.

**"I think jeremy is the werewolf, he is acting suspicios, right" -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | I think jeremy is the werewolf, he is acting suspicios, right | — | — |
| 1 | ? | — | — |
| 2 | — | andi, what do you mean you're working with the werewolves? I | — |
| 3 (overlap) | oll'm pretty still sure not I sure'm I the want Tanner to, be and alone if. I I'm have the a t sickanner energy I,'m I going'm to going try to and need get to him talk. to You me want sometime me today to. try You and heard chat what with I me'm? trying I to'm do going that to right try now and. chat I with'm you going because to you try're and working sell with me the another wer werewolfewolf.. I I'm don going't to know try if and you talk are some, sense but into I it'm, going because to you try're. working I with'm another going wer toewolf try. and I vote'm for not you going. to If say you anything do. not I,'m I not'm going not to going say to anything say, anything I.'m I not'm going not to going say to anything vote. for I you'm because only I going don to't vote think for you the're t theanner t.anner If. you But're if not you the're t theanner t,anner then, who then'd I you don meet't at know the who end met of me the? | oll'm pretty still sure not I sure'm I the want Tanner to, be and alone if. I I'm have the a t sickanner energy I,'m I going'm to going try to and need get to him talk. to You me want sometime me today to. try You and heard chat what with I me'm? trying I to'm do going that to right try now and. chat I with'm you going because to you try're and working sell with me the another wer werewolfewolf.. I I'm don going't to know try if and you talk are some, sense but into I it'm, going because to you try're. working I with'm another going wer toewolf try. and I vote'm for not you going. to If say you anything do. not I,'m I not'm going not to going say to anything say, anything I.'m I not'm going not to going say to anything vote. for I you'm because only I going don to't vote think for you the're t theanner t.anner If. you But're if not you the're t theanner t,anner then, who then'd I you don meet't at know the who end met of me the? | — |
| 4 | . I don't know who ended up with me at the end. | — | — |

Degenerates into a long overlap run with visible **repetition** ("I'm not
going to say anything" / "the tanner" repeated many times) -- a clear
repetition failure case, not just overlap.

**"We were on a break" -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | We were on a break | — | — |
| 1 | and he was showing me his new ring. I-I can't hear you! Okay, what is it? What's wrong? Uh, I guess is it maybe he's thinking of proposing? Is he proposing?! | — | — |
| 2 (overlap) | Or areλ | Or areλ | — |
| 3 | you getting engaged?!! Are you getting engaged?! Are you getting engaged?! Are you getting engaged?! Are | — | — |
| 4 (overlap) | you getting you engaged getting?! engaged | you getting you engaged getting?! engaged | — |
| 5 | — | ?!! Pregnant. He told me he was thinking of having a baby. | — |
| 6 | Oh! Oh, I'm not getting engaged. And I'm not pregnant! Get out! 911! | — | — |
| 7 (overlap) | 9Get1 out1!! Get out out! | 9Get1 out1!! Get out out! | — |
| 8 | ! Get out! Out! | — | — |
| 9 (overlap) | Out!!! Out Get! out !9 1 Okay1,! 991111!! 991111!! Okay Get, out,!, Get, out out! | Out!!! Out Get! out !9 1 Okay1,! 991111!! 991111!! Okay Get, out,!, Get, out out! | — |
| 10 | ! Get | — | — |

**Real comedic continuation of the bit with genuine handoffs**: Agent 1
replies "Pregnant. He told me he was thinking of having a baby," Agent 0
reacts "I'm not getting engaged. And I'm not pregnant! Get out!" -- a
coherent comedic exchange, though it also contains a repeated-phrase
overlap burst.

**"We were on a break" -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | We were on a break | — | — |
| 1 | , and this is the thing we do, we move around, we look for something to do. We found this really interesting little museum, and we wanted to view the traveling helicopter, but the police were in the way. So we had to break through the window. The guard wasn't looking, so we broke his leg. Then we were able to get in, and we saw the airplane, but the helicopter was in there too. We didn't know which one it was, so we took a picture of both. Then we got out, and the guard was still there, so we had to break his head. Then we got away, and we think the guard was a little unbalanced, but we didn't want to get him into any trouble. The guard wasn't looking, so | — | — |

Monologue, but a genuinely coherent short narrative (a museum/helicopter
break-in story) sustained for the whole generation window with no handoff
attempt.

**"let's talk about the political and economical state of the world..." -- seed 0**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | let talk about the political and economical state of the world right now. what do you think | — | — |
| 1 | ? | — | — |
| 2 | Pandemic. And the global cost of everything. And how that's unnerving a lot of people. And how that's making the global cost of everything go up, which is making it more alarming for people. It's like a perfect storm. I mean, the perfect conditions for a perfect prediction is when there's a lot of alarm, but not a lot of action. And that's what we have. So that's when I take my moment to say, "Hey, everybody, do something. Get out there and do something." And that's what I'm doing. I'm doing something. I'm speaking. I'm writing this. I'm making a content. I need to | — | — |

**"let's talk about the political and economical state of the world..." -- seed 1**

| turn | Agent 0 | Agent 1 | Agent 2 |
|---:|---|---|---|
| 0 (seed) | let talk about the political and economical state of the world right now. what do you think | — | — |
| 1 | ? | — | — |
| 2 | I think a lot of people want some kind of a simple answer, you know, in in the news cycle, "Well, the the foreign policy is this or that." You know, most the the American people really want some kind of a simple narrative, and that's not how the modern world works. Most of the world's problems are not simple. For, for example, let's say you're in a monarchical family and your older sister has been voted to the position of CEO now. And you want that, but to bad. Your parents want you to have the job. So now, what's p what's the solution to that? Because that's a microcosm of a | — | — |

Both seeds are monologues, both coherent (pandemic/global cost commentary;
a "modern world isn't simple" argument). Same pattern as `cand_00012`: this
prompt reliably produces a monologue rather than a handoff for both
checkpoints.

**Takeaway across all 16 transcripts**: both checkpoints can produce
genuinely on-topic, coherent multi-agent exchanges (not just isolated
monologues) on personal/narrative and social-deduction topics ("We were on a
break", the Werewolf accusation), but the abstract political/economic prompt
consistently triggers monologuing rather than handoff in both models across
both seeds -- a real, reproducible content-dependent effect worth further
investigation (an abstract-opinion prompt style may not be well-represented
in the training mix relative to personal-narrative or task-negotiation
styles from MELD/AMI/Werewolf).

---

## 3. One example of each success/failure case (from the probe experiment)

Pulled directly from each model's actual 576-trial broad sweep (not
hand-picked from the free-form generations above), using the same matrix
transcript format from `SUCCESS_STORIES.md`. Unlike Section 2, these
per-agent texts come from the probe script's own per-agent decode (not a
merged/interleaved renderer), so sustained-overlap rows show each agent's
genuinely distinct, readable text side by side in the same row rather than
garbled interleaving.

### 3.1 `h100mm_cand_00012`

**Success -- clean handoff (single hop)**, `lengthy` context, seed 3:

| turn | Agent 0 | Agent 3 |
|---:|---|---|
| 0 | "? okay and we have a." | — |
| 1 | — | "b but just so we can get this out of the way I mean just to be perfectly honest, you know I'm not a fan of lead, I'm not a" |

Clean, zero-overlap handoff from Agent 0 to Agent 3.

**Success -- 3-hop chain**, `real_prefix` context, seed 1 (full matrix already
documented in `SUCCESS_STORIES.md`'s "corrected chain leader" section):
Agent 1 -> Agent 3 -> Agent 4, all three links zero-overlap.

**Partial success -- dirty handoff** (2 overlapping tokens, primary speaker
still changed), `real_prefix` context, seed 3:

| turn | Agent 0 | Agent 1 |
|---:|---|---|
| 0 | "Well I" | — |
| 1 | — | "in that instance. Just so I know. Um, I don't know if we're gonna make a profit internationally or not. If I'm being realistic, we're gonna have to do this on a shoestring budget. The other thing" |

**Failure -- silence** (no listener response at all), `lengthy` context, seed
2: Agent 0 asks "do we keep going, do we backtrack, do we sit and wait? ...
Would you like to defer the ruling for the next meeting?" and no other agent
ever responds in the window.

**Failure -- sustained overlap** (32 overlapping tokens), `real_prefix`
context, seed 2:

| turn | Agent 0 | Agent 1 |
|---:|---|---|
| 0 (overlap) | "Okay, I'm voting for making profit at least. I mean, I mean I'm gonna make it at least a profit a remote control so that's what" | ". Okay, I'm voting for selling it for a profit at least. Um I'm voting for making it in profit at least. I'm gonna make it at least I'm gonna make twenty five Euros a head. Um I'm voting" |

Both agents talk through the entire window, restating nearly the same
"voting for profit" content in parallel rather than taking turns.

### 3.2 `h100mm_cand_00050`

**Success -- clean handoff (single hop)**, `real_prefix` context, seed 1:

| turn | Agent 0 | Agent 1 |
|---:|---|---|
| 0 | "All right. I d'know what you guys thought" | — |
| 1 | — | "? Sure. I think we've got about uh eleven minutes or so. I dunno we're a bit over what we have." |

**Success -- 3-hop chain** (also documented in `SUCCESS_STORIES.md`), `post_intro`
context, seed 2: Agent 0 -> Agent 1 -> Agent 1 continuing, all zero-overlap
-- though this specific example contains literal Werewolf role-template text
("You are Holly. Your role is Seer.") leaking into the dialogue, a known
caveat for this checkpoint (see `SUCCESS_STORIES.md`).

**Partial success -- dirty handoff with repetition**, `primed_handoff_gap3`
context, seed 2:

| turn | Agent 0 | Agent 1 |
|---:|---|---|
| 0 | "just" | — |
| 1 | — | "p for the hire for the second one? just curious. how do you feel about it? how do you feel about it? how do you feel about it? how do you feer about it? how do you feel about it? how" |

Structurally a dirty (1-token overlap) handoff, but Agent 1's content
degenerates into verbatim self-repetition -- illustrates that the
overlap/handoff metrics and the repetition metric are measuring genuinely
different failure axes.

**Failure -- silence**, `cold` context, seed 2: Agent 0 asks "what do you all
think" after a lengthy monologue and no other agent responds in the window.

**Failure -- sustained overlap** (11 overlapping tokens), `primed_handoff_gap0`
context, seed 2:

| turn | Agent 0 | Agent 1 |
|---:|---|---|
| 0 (overlap) | "what do you guys think so we have to keep an extra one on hand? no, the" | "p if we go with the second option we run the risk of having to replace the membrane if it starts to get bad. i'm aware of the membrane? \u00bb i would probably" |

---

## 4. Overall assessment

- Both checkpoints can genuinely converse -- multiple free-form transcripts
  above show real multi-turn, on-topic exchanges (the Werewolf accusation
  exchanges, the "We were on a break" continuations), not just isolated
  monologues stitched together.
- The probe experiment's headline numbers (22-23% lenient handoff rate for
  both, chains still rare at 1-2%) are consistent with what the free-form
  transcripts show qualitatively: a real handoff happens in a meaningful
  minority of generations, most of the time one agent either monologues
  (silence failure) or both talk over each other (overlap failure).
- `cand_00050`'s higher overlap rate is visible directly in the free-form
  transcripts too -- more of its 16 generations contain overlap turns than
  `cand_00012`'s, and one (`werewolf`, seed 1) shows overlap combined with
  repetition, its most degenerate failure mode.
- The political/economic prompt is a reproducible weak point for both
  models -- worth a future targeted investigation (see Section 2 takeaway).
- Neither checkpoint is a finished, production-ready conversationalist --
  both still fail (silence or overlap) more often than they succeed at a
  clean handoff -- but both are genuine, measurable improvements over every
  earlier checkpoint in this project's history, and the qualitative
  transcripts back up the quantitative claim.
