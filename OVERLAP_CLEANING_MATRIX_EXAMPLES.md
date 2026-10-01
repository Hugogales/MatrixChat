# Overlap Cleaning: Matrix-Format Examples

Showing conversations in matrix table format (like SUCCESS_STORIES.md) to see
exactly where overlaps occur and how cleaning resolves them.

> **Finding: this prototype is not coherence-safe.** It removes most overlap, but it moves only the overlapping token fragments to the end of the example. The decoded AFTER conversations below contain split sentences and out-of-order fragments. Do not use this transformation for training; the next design must operate on complete utterances and snap moves to nearby turn boundaries.

**Configuration:**
- **AMI**: Keep ≤2 column overlaps, defer >2 column overlaps
- **Werewolf**: Keep ≤3 column overlaps, defer >3 column overlaps

- **Both**: Remove isolated speaker runs whose decoded text contains no letters or numbers (for example `.`, `...`, or `—`). Runs such as `yeah` remain because they contain letters.

---

## AMI Dataset

### Example 1 (dataset index 645)

**Statistics:**
- Original: 384 cols, 103 overlap cols (26.8%)
- Cleaned: 504 cols, 12 overlap cols (2.4%)
- Reduction: 91 fewer overlap columns

- Removed punctuation-only runs: 5

#### BEFORE (overlaps present)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 |
| --- | --- | --- | --- | --- |
| 0 | — | — | — | " so" |
| 1 | — | — | "Thanks" | "." |
| 2 | — | — | — | — |
| 3 | — | — | — | "Looking" |
| 4 | — | — | — | " at" |
| 5 | — | — | — | " what" |
| 6 | — | — | — | " we" |
| 7 | — | — | — | "'ve" |
| 8 | — | — | — | " got" |
| 9 | — | — | — | "," |
| 10 | — | — | — | " we" |
| 11 | — | — | — | " we" |
| 12 | — | — | — | " want" |
| 13 | — | — | — | " an" |
| 14 | — | — | — | " L" |
| 15 | — | — | — | "_C" |
| 16 | — | — | — | "_D" |
| 17 | — | — | — | "_" |
| 18 | — | — | — | " display" |
| 19 | — | — | — | " with" |
| 20 | — | — | — | " a" |
| 21 | "Yeah" | — | — | " spinning" |
| 22 | — | — | — | " wheel" |
| 23 | — | — | — | "." |
| 24 | "." | — | — | — |
| 25 | "Let" | — | — | — |
| 26 | "'s" | — | — | — |
| 27 | " let" | — | — | — |
| 28 | "'s" | — | — | — |
| 29 | " try" | — | — | — |
| 30 | " to" | — | — | — |
| 31 | " r" | — | — | — |
| 32 | " rub" | — | — | — |
| 33 | " off" | — | — | — |
| 34 | " things" | — | — | "Yeah" |
| 35 | " and" | — | — | "," |
| 36 | "yeah" | — | — | "rub" |
| 37 | — | — | — | " off" |
| 38 | — | — | — | " some" |
| 39 | — | — | — | " of" |

*(showing first 40 of 384 columns)*


**BEFORE as readable dialogue:**

```text
[000-000] Agent 3: so
[001-001] Agent 2: Thanks
[001-020] Agent 3: .Looking at what we've got, we we want an L_C_D_ display with a
[021-021] Agent 0: Yeah
[021-023] Agent 3: spinning wheel.
[024-034] Agent 0: .Let's let's try to r rub off things
[034-034] Agent 3: Yeah
[035-035] Agent 0: and
[035-035] Agent 3: ,
[036-036] Agent 0: yeah
[036-041] Agent 3: rub off some of those.
[042-042] Agent 0: ,
[042-042] Agent 2: [c
[043-043] Agent 0: so
[043-044] Agent 2: ough]
[045-058] Agent 0: umhand dynamos are definitely out, right?You you got
[058-058] Agent 2: Yeah
[058-058] Agent 3: Yeah
[059-059] Agent 0: a
[059-059] Agent 2: uh
[059-059] Agent 3: ,
[060-060] Agent 0: wind
[060-060] Agent 2: -h
[060-060] Agent 3: it
[061-061] Agent 0: dynam
[061-061] Agent 2: um
[061-061] Agent 3: 's
[062-062] Agent 0: o
[062-063] Agent 2: yeah.
[064-064] Agent 0: ,
[064-064] Agent 3: not
[065-065] Agent 0: yeah
[065-065] Agent 3: that
[066-066] Agent 0: .
[066-071] Agent 3: 's not streamlined and sexy,
[072-072] Agent 0: Okay
[072-072] Agent 3: having
[073-073] Agent 0: .
[073-078] Agent 3: a having a wind up.
[079-091] Agent 0: Umkinetic energy does seem to have some kind of uh uh
[091-093] Agent 3: I think tha
[094-097] Agent 0: appeal,but uh
[099-100] Agent 1: It's
[101-101] Agent 0: it
[101-101] Agent 1: about
[102-102] Agent 0: 's
[102-108] Agent 1: the practicality of it really,
[109-109] Agent 0: Yeah
[109-109] Agent 1: isn
[110-110] Agent 0: .
[110-110] Agent 1: 't
[111-111] Agent 0: As
[111-111] Agent 1: it
[112-112] Agent 0: against
[112-112] Agent 1: ?
[113-113] Agent 0: a
[113-118] Agent 1: You know?I mean if
[119-146] Agent 0: watch, which constantly keeps moving,this this thing will have to be tapped every time,which which might be very frustrating for the user.
[147-152] Agent 3: Depends how much how much
[153-153] Agent 0: Kin
[153-153] Agent 3: movement
[154-154] Agent 0: etic
[154-154] Agent 3: it
[155-155] Agent 0: energy
[155-155] Agent 3: really
[156-156] Agent 0: it
[156-156] Agent 3: needs
[157-157] Agent 0: needs
[157-157] Agent 3: .
[158-158] Agent 0: I
[158-158] Agent 3: Pr
[159-159] Agent 0: don
[159-159] Agent 3: presumably
[160-160] Agent 0: 't
[160-160] Agent 3: if
[161-161] Agent 0: have
[161-161] Agent 3: they
[162-162] Agent 0: too
[162-162] Agent 3: 're
[163-163] Agent 0: much
[163-163] Agent 3: suggesting
[164-164] Agent 0: technical
[164-164] Agent 3: it
[165-165] Agent 0: information
[165-165] Agent 3: ,
[166-166] Agent 0: on
[166-166] Agent 3: then
[167-167] Agent 0: that
[167-167] Agent 3: we
[168-168] Agent 0: ,
[168-168] Agent 3: could
[169-169] Agent 0: yeah
[169-169] Agent 3: use
[170-170] Agent 0: ,
[170-170] Agent 3: it
[171-171] Agent 0: right
[171-171] Agent 3: .
[172-172] Agent 0: .
[172-172] Agent 3: I
[173-173] Agent 0: Okay
[173-173] Agent 3: 'd
[174-174] Agent 0: ,
[174-174] Agent 3: I
[175-175] Agent 0: let
[175-175] Agent 3: 'd
[176-176] Agent 0: 's
[176-176] Agent 3: keep
[177-177] Agent 0: keep
[177-177] Agent 3: it
[178-178] Agent 0: it
[178-178] Agent 3: on
[179-179] Agent 0: option
[179-179] Agent 3: .
[180-197] Agent 0: uh keep an option,yeah.Um the flat co completely flat case is definitely out,
[197-197] Agent 3: We
[198-198] Agent 0: right
[198-198] Agent 3: don
[199-199] Agent 0: ?
[199-200] Agent 3: 't want
[201-201] Agent 0: It
[201-201] Agent 2: Yeah
[201-201] Agent 3: that
[202-202] Agent 0: has
[202-202] Agent 2: it
[202-202] Agent 3: it
[203-203] Agent 0: to
[203-203] Agent 2: 's
[203-203] Agent 3: 's
[204-204] Agent 0: be
[204-204] Agent 2: yeah
[204-204] Agent 3: no
[205-205] Agent 0: at
[205-205] Agent 2: .
[205-205] Agent 3: it
[206-206] Agent 0: least
[206-206] Agent 3: 's
[207-207] Agent 0: curved
[207-207] Agent 3: not
[208-208] Agent 0: from
[208-208] Agent 3: not
[209-213] Agent 0: one side, yeah.
[214-215] Agent 3: vegetable.
[216-233] Agent 0: Um okay,we still have all all the options.Wood, do you think wood will
[233-233] Agent 2: N
[234-234] Agent 0: be
[234-234] Agent 2: wood
[235-235] Agent 0: a
[235-235] Agent 2: is
[236-238] Agent 0: good idea?
[239-255] Agent 2: I can't nhow do youuh I mean you can't keep it really small
[256-258] Agent 0: Mm.
[259-267] Agent 2: uhyou can't make it like thin and
[268-269] Agent 0: Right.
[270-271] Agent 2: The wood
[272-272] Agent 1: I
[272-274] Agent 3: Mm.
[275-275] Agent 1: can
[275-275] Agent 2: thing
[276-276] Agent 1: 't
[276-276] Agent 2: .
[277-277] Agent 1: imagine
[277-277] Agent 2: Because
[278-278] Agent 1: a
[278-278] Agent 2: you
[279-279] Agent 1: m
[279-279] Agent 2: need
[280-280] Agent 1: wooden
[280-280] Agent 2: to
[281-281] Agent 1: remote
[281-281] Agent 2: you
[282-282] Agent 1: control
[282-282] Agent 2: n
[283-283] Agent 1: .
[283-301] Agent 2: you need to put all the technology in,so I mean if the case you add the case
[302-302] Agent 0: Yeah
[302-302] Agent 2: and
[303-303] Agent 0: if
[303-303] Agent 2: it
[304-304] Agent 0: if
[304-304] Agent 2: it
[305-305] Agent 0: it
[305-305] Agent 2: becomes
[306-306] Agent 0: is
[306-306] Agent 2: a
[307-307] Agent 0: really
[307-307] Agent 2: bit
[308-308] Agent 0: thin
[308-308] Agent 2: bulky
[309-309] Agent 0: if
[309-309] Agent 2: wi
[310-310] Agent 0: it
[310-313] Agent 2: mm-mm yeah.
[314-314] Agent 0: is
[314-316] Agent 3: Mm.
[317-328] Agent 0: really thin it it's likely to break,it's it
[328-328] Agent 2: Yeah
[329-329] Agent 0: 's
[329-329] Agent 2: ,
[330-330] Agent 0: much
[330-331] Agent 2: yeah.
[332-333] Agent 0: more uh
[333-333] Agent 2: Yeah
[333-333] Agent 3: Yeah
[334-334] Agent 2: .
[334-354] Agent 3: ,and given that we're we're looking at more spongy material preferences, I ha would think
[355-355] Agent 2: U
[355-355] Agent 3: maybe
[356-356] Agent 2: yeah
[356-356] Agent 3: rubber
[357-357] Agent 2: wood
[357-357] Agent 3: or
[358-358] Agent 2: is
[358-358] Agent 3: plastic
[359-359] Agent 2: not
[359-359] Agent 3: is
[360-360] Agent 2: really
[361-361] Agent 0: Right
[361-361] Agent 2: yeah
[361-361] Agent 3: more
[362-362] Agent 0: .
[363-363] Agent 1: Well
[363-363] Agent 2: .
[364-370] Agent 1: it's not very cleanable either
[370-371] Agent 3: Yeah.
[372-372] Agent 0: That
[372-372] Agent 1: ,
[373-373] Agent 0: 's
[373-373] Agent 1: do
[374-374] Agent 0: true
[374-374] Agent 1: you
[374-374] Agent 2: Yeah
[375-375] Agent 0: .
[375-375] Agent 2: .
[376-383] Agent 1: know.It's it's not a
```

**Overlap analysis:** Look for rows where multiple agents have text (not —) at the same time step.

#### AFTER (overlaps cleaned)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 |
| --- | --- | --- | --- | --- |
| 0 | — | — | — | " so" |
| 1 | — | — | "Thanks" | "." |
| 2 | — | — | — | — |
| 3 | — | — | — | "Looking" |
| 4 | — | — | — | " at" |
| 5 | — | — | — | " what" |
| 6 | — | — | — | " we" |
| 7 | — | — | — | "'ve" |
| 8 | — | — | — | " got" |
| 9 | — | — | — | "," |
| 10 | — | — | — | " we" |
| 11 | — | — | — | " we" |
| 12 | — | — | — | " want" |
| 13 | — | — | — | " an" |
| 14 | — | — | — | " L" |
| 15 | — | — | — | "_C" |
| 16 | — | — | — | "_D" |
| 17 | — | — | — | "_" |
| 18 | — | — | — | " display" |
| 19 | — | — | — | " with" |
| 20 | — | — | — | " a" |
| 21 | "Yeah" | — | — | " spinning" |
| 22 | — | — | — | " wheel" |
| 23 | — | — | — | "." |
| 24 | "." | — | — | — |
| 25 | "Let" | — | — | — |
| 26 | "'s" | — | — | — |
| 27 | " let" | — | — | — |
| 28 | "'s" | — | — | — |
| 29 | " try" | — | — | — |
| 30 | " to" | — | — | — |
| 31 | " r" | — | — | — |
| 32 | " rub" | — | — | — |
| 33 | " off" | — | — | — |
| 34 | — | — | — | "Yeah" |
| 35 | — | — | — | "," |
| 36 | — | — | — | "rub" |
| 37 | — | — | — | " off" |
| 38 | — | — | — | " some" |
| 39 | — | — | — | " of" |

*(showing first 40 of 504 columns)*


**AFTER as readable dialogue:**

```text
[000-000] Agent 3: so
[001-001] Agent 2: Thanks
[001-020] Agent 3: .Looking at what we've got, we we want an L_C_D_ display with a
[021-021] Agent 0: Yeah
[021-023] Agent 3: spinning wheel.
[024-033] Agent 0: .Let's let's try to r rub off
[034-041] Agent 3: Yeah,rub off some of those.
[042-042] Agent 2: [c
[043-043] Agent 0: so
[043-044] Agent 2: ough]
[045-057] Agent 0: umhand dynamos are definitely out, right?You you
[058-063] Agent 2: Yeah uh-hum yeah.
[064-071] Agent 3: not that's not streamlined and sexy,
[072-072] Agent 0: Okay
[072-078] Agent 3: having a having a wind up.
[079-091] Agent 0: Umkinetic energy does seem to have some kind of uh uh
[091-093] Agent 3: I think tha
[094-097] Agent 0: appeal,but uh
[099-100] Agent 1: It's
[101-101] Agent 0: it
[101-101] Agent 1: about
[102-102] Agent 0: 's
[102-118] Agent 1: the practicality of it really, isn't it? You know?I mean if
[119-146] Agent 0: watch, which constantly keeps moving,this this thing will have to be tapped every time,which which might be very frustrating for the user.
[147-179] Agent 3: Depends how much how much movement it really needs.Pr presumably if they're suggesting it, then we could use it.I'd I'd keep it on.
[180-196] Agent 0: uh keep an option,yeah.Um the flat co completely flat case is definitely out
[197-208] Agent 3: We don't want thatit's no it's not not
[209-213] Agent 0: one side, yeah.
[214-215] Agent 3: vegetable.
[216-232] Agent 0: Um okay,we still have all all the options.Wood, do you think wood
[233-235] Agent 2: N wood is
[236-238] Agent 0: good idea?
[239-255] Agent 2: I can't nhow do youuh I mean you can't keep it really small
[256-258] Agent 0: Mm.
[259-267] Agent 2: uhyou can't make it like thin and
[268-269] Agent 0: Right.
[270-271] Agent 2: The wood
[272-272] Agent 1: I
[272-274] Agent 3: Mm.
[275-313] Agent 2: thing.Because you need to you n you need to put all the technology in,so I mean if the case you add the case and it it becomes a bit bulkywi mm-mm yeah.
[314-314] Agent 0: is
[314-316] Agent 3: Mm.
[317-327] Agent 0: really thin it it's likely to break,it's
[328-331] Agent 2: Yeah, yeah.
[332-333] Agent 0: more uh
[333-333] Agent 2: Yeah
[333-359] Agent 3: Yeah,and given that we're we're looking at more spongy material preferences, I ha would think maybe rubber or plastic is
[360-360] Agent 2: really
[361-361] Agent 0: Right
[361-361] Agent 2: yeah
[361-361] Agent 3: more
[363-370] Agent 1: Well it's not very cleanable either
[370-371] Agent 3: Yeah.
[372-375] Agent 0: That's true.
[376-383] Agent 1: know.It's it's not a
[385-393] Agent 0: things andyeah got a wind dynamo
[395-398] Agent 3: Yeah,it's
[401-450] Agent 0: , yeah.Yeah.As against aKinetic energy it needsI don't have too much technical information on that,yeah, right.Okay, let's keep it option, right?It has to be at least curved from
[452-456] Agent 2: Yeahit'syeah.
[461-463] Agent 0: will be a
[465-473] Agent 1: can't imagine a m wooden remote control.
[475-487] Agent 0: Yeahif if it is really thin if it it's much
[489-493] Agent 2: U yeah wood is not
[495-497] Agent 1: ,do you
[502-503] Agent 2: Yeah.
```

**Result:** 504 columns, 12 overlap columns (2.4%). The 91 overlapping segments were deferred to create clean handoffs.

**Coherence warning:** this prototype defers only the tokens inside an overlap span and appends them near the end. Inspect the decoded AFTER dialogue for split words, detached acknowledgements, or out-of-order replies; lower overlap alone does not prove the conversation remains valid.

---

### Example 2 (dataset index 2962)

**Statistics:**
- Original: 384 cols, 93 overlap cols (24.2%)
- Cleaned: 503 cols, 19 overlap cols (3.8%)
- Reduction: 74 fewer overlap columns

- Removed punctuation-only runs: 12

#### BEFORE (overlaps present)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 |
| --- | --- | --- | --- | --- |
| 0 | — | — | "'s" | — |
| 1 | — | — | " hard" | — |
| 2 | — | — | " but" | — |
| 3 | "Yeah" | — | " I" | — |
| 4 | " well" | — | " think" | — |
| 5 | "yeah" | — | " it" | — |
| 6 | — | — | "'s" | — |
| 7 | — | — | " possible" | — |
| 8 | " well" | — | "but" | "Well" |
| 9 | " that" | — | " it" | " we" |
| 10 | " has" | — | " uh" | " already" |
| 11 | " been" | — | "yeah" | " eliminated" |
| 12 | " e" | — | " yeah" | — |
| 13 | " that" | — | — | — |
| 14 | " has" | — | — | — |
| 15 | " been" | — | — | — |
| 16 | " eliminated" | "Elim" | — | " that" |
| 17 | "," | "inated" | — | "." |
| 18 | — | "." | — | — |
| 19 | " so" | — | " yeah" | — |
| 20 | " that" | — | "so" | — |
| 21 | "'s" | — | " it" | — |
| 22 | " that" | — | "'s" | — |
| 23 | "'s" | — | " it" | — |
| 24 | " unfortunately" | — | "'s" | — |
| 25 | " a" | — | " okay" | — |
| 26 | " moot" | — | "," | — |
| 27 | " point" | — | " yeah" | — |
| 28 | " now" | — | "," | — |
| 29 | "." | — | " yeah" | — |
| 30 | " M" | — | — | — |
| 31 | "m" | — | — | — |
| 32 | "-h" | — | — | — |
| 33 | "mm" | — | — | — |
| 34 | "." | — | — | — |
| 35 | — | — | "." | — |
| 36 | — | — | "And" | — |
| 37 | — | — | " uh" | — |
| 38 | — | — | " different" | — |
| 39 | — | — | " shapes" | — |

*(showing first 40 of 384 columns)*


**BEFORE as readable dialogue:**

```text
[000-002] Agent 2: 's hard but
[003-003] Agent 0: Yeah
[003-003] Agent 2: I
[004-004] Agent 0: well
[004-004] Agent 2: think
[005-005] Agent 0: yeah
[005-007] Agent 2: it's possible
[008-008] Agent 0: well
[008-008] Agent 2: but
[008-008] Agent 3: Well
[009-009] Agent 0: that
[009-009] Agent 2: it
[009-009] Agent 3: we
[010-010] Agent 0: has
[010-010] Agent 2: uh
[010-010] Agent 3: already
[011-011] Agent 0: been
[011-011] Agent 2: yeah
[011-011] Agent 3: eliminated
[012-012] Agent 0: e
[012-012] Agent 2: yeah
[013-016] Agent 0: that has been eliminated
[016-016] Agent 1: Elim
[016-016] Agent 3: that
[017-017] Agent 0: ,
[017-017] Agent 1: inated
[017-017] Agent 3: .
[018-018] Agent 1: .
[019-019] Agent 0: so
[019-019] Agent 2: yeah
[020-020] Agent 0: that
[020-020] Agent 2: so
[021-021] Agent 0: 's
[021-021] Agent 2: it
[022-022] Agent 0: that
[022-022] Agent 2: 's
[023-023] Agent 0: 's
[023-023] Agent 2: it
[024-024] Agent 0: unfortunately
[024-024] Agent 2: 's
[025-025] Agent 0: a
[025-025] Agent 2: okay
[026-026] Agent 0: moot
[026-026] Agent 2: ,
[027-027] Agent 0: point
[027-027] Agent 2: yeah
[028-028] Agent 0: now
[028-028] Agent 2: ,
[029-029] Agent 0: .
[029-029] Agent 2: yeah
[030-034] Agent 0: Mm-hmm.
[035-070] Agent 2: .And uh different shapes that we can dolike uh we can have you know a all animals shapes or you know comfortable uh whi which can fit into your handsand um so
[070-070] Agent 3: Now
[071-071] Agent 2: that
[071-071] Agent 3: that
[072-072] Agent 2: uh
[072-080] Agent 3: 's good from a marketing point of view,
[081-081] Agent 2: Yeah
[081-081] Agent 3: the
[082-082] Agent 2: ,
[082-082] Agent 3: fun
[083-083] Agent 2: yeah
[083-083] Agent 3: the
[084-084] Agent 2: ,
[084-084] Agent 3: fun
[085-085] Agent 2: yeah
[085-085] Agent 3: shape
[086-086] Agent 2: and
[086-086] Agent 3: .
[087-089] Agent 2: colours also,
[090-090] Agent 0: Yeah
[090-090] Agent 2: different
[090-091] Agent 3: And that
[092-092] Agent 0: I
[093-093] Agent 1: M
[093-093] Agent 2: colours
[093-093] Agent 3: you
[094-094] Agent 1: m
[094-094] Agent 2: ,
[094-094] Agent 3: you
[095-098] Agent 1: -hmm colours.
[099-099] Agent 2: and
[100-110] Agent 3: say that won't add too much to the budget?
[111-111] Agent 2: No
[111-111] Agent 3: To
[112-112] Agent 2: no
[112-112] Agent 3: d
[113-113] Agent 2: no
[113-113] Agent 3: the
[114-114] Agent 2: ,
[114-114] Agent 3: shape
[115-115] Agent 2: it
[115-115] Agent 3: is
[116-116] Agent 2: won
[116-116] Agent 3: uh
[117-136] Agent 2: 't uh I don't think it will be like,you can have you know for uh if you
[137-137] Agent 0: It
[137-137] Agent 2: want
[138-138] Agent 0: just
[138-138] Agent 2: ther
[139-139] Agent 0: build
[139-139] Agent 2: there
[140-140] Agent 0: a
[140-140] Agent 2: to
[141-141] Agent 0: mould
[141-142] Agent 2: be more
[143-145] Agent 0: basically and uh
[145-145] Agent 2: Yeah
[146-146] Agent 0: you
[146-149] Agent 2: yeah.It's
[150-150] Agent 0: know
[150-150] Agent 1: Yes
[150-150] Agent 2: it
[151-151] Agent 0: .
[151-151] Agent 2: 's
[152-152] Agent 1: exactly
[152-152] Agent 2: just
[153-153] Agent 1: .
[153-161] Agent 2: a s shape so it doesn't matter.
[162-178] Agent 0: As the budget we're looking at if you build one mould I don't think that
[178-178] Agent 2: Yeah
[179-179] Agent 0: 's
[179-179] Agent 2: .
[180-192] Agent 0: going to make a big difference whether it's gonna be square or
[193-219] Agent 3: Do you think there's any chance of um having ser in having basically the same machine with the same buttons but maybe several different shapes?
[220-222] Agent 2: Yeah that is
[223-223] Agent 0: Oh
[223-223] Agent 2: also
[224-224] Agent 0: yes
[224-224] Agent 2: possible
[225-225] Agent 0: .
[226-226] Agent 2: I
[226-226] Agent 3: Is
[227-227] Agent 2: uh
[227-227] Agent 3: that
[228-228] Agent 2: yeah
[228-229] Agent 3: gonna be
[230-230] Agent 1: Yes
[230-230] Agent 2: I
[230-230] Agent 3: a
[231-231] Agent 1: .
[231-232] Agent 3: possible?
[233-233] Agent 2: I
[233-235] Agent 3: 'Cause that
[236-236] Agent 0: I
[236-236] Agent 2: yeah
[236-236] Agent 3: might
[237-237] Agent 2: .
[237-237] Agent 3: help
[238-238] Agent 0: think
[238-238] Agent 3: with
[239-239] Agent 0: I
[239-239] Agent 3: the
[240-240] Agent 0: think
[240-240] Agent 3: marketing
[241-241] Agent 0: we
[241-241] Agent 3: .
[242-245] Agent 0: will have to look
[245-245] Agent 2: Yeah
[246-246] Agent 0: at
[246-246] Agent 2: that
[247-247] Agent 0: the
[247-247] Agent 2: will
[248-248] Agent 0: budget
[248-248] Agent 2: be
[249-266] Agent 0: on thatbut I think in principle that that would be that would be kind of fun,
[266-266] Agent 2: Yeah
[266-266] Agent 3: Because
[267-267] Agent 0: you
[267-267] Agent 3: we
[268-269] Agent 0: know.
[270-270] Agent 2: yeah
[270-270] Agent 3: had
[271-271] Agent 2: .
[272-274] Agent 3: something sort of
[275-275] Agent 2: M
[275-275] Agent 3: sexy
[276-279] Agent 2: m-hmm.
[280-281] Agent 3: for adults
[282-282] Agent 0: Yeah
[282-282] Agent 3: and
[283-283] Agent 0: .
[283-287] Agent 3: we could have something sort
[288-288] Agent 0: S
[288-288] Agent 3: of
[289-289] Agent 0: illy
[289-289] Agent 3: s
[290-290] Agent 0: for
[290-290] Agent 3: illy
[291-291] Agent 0: children
[291-291] Agent 2: Yeah
[292-292] Agent 0: .
[293-293] Agent 2: ,
[293-293] Agent 3: for
[294-294] Agent 2: for
[294-294] Agent 3: children
[295-295] Agent 2: children
[296-296] Agent 0: Like
[296-296] Agent 2: ,
[296-296] Agent 3: or
[297-297] Agent 0: an
[297-297] Agent 2: yeah
[297-297] Agent 3: a
[298-298] Agent 0: animal
[298-299] Agent 3: little animal
[300-300] Agent 0: or
[300-300] Agent 1: Like
[300-300] Agent 2: exactly
[300-300] Agent 3: shape
[301-302] Agent 1: a doll
[302-302] Agent 2: .
[302-302] Agent 3: or
[303-303] Agent 1: ,
[303-303] Agent 2: Yeah
[303-303] Agent 3: in
[304-304] Agent 1: or
[304-304] Agent 2: ,
[304-304] Agent 3: a
[305-306] Agent 2: that's
[307-307] Agent 0: Yeah
[307-307] Agent 2: what
[307-307] Agent 3: or
[308-308] Agent 0: .
[308-308] Agent 2: ,
[308-308] Agent 3: a
[309-309] Agent 2: yeah
[309-309] Agent 3: little
[310-310] Agent 2: .
[311-317] Agent 3: elephant so they can remember where it
[318-319] Agent 2: Yeah,
[320-320] Agent 1: Yes
[320-320] Agent 2: exactly
[320-321] Agent 3: is.
[322-322] Agent 1: .
[323-383] Agent 2: . Yeah.And and the butto buttons also I think if you want to have more features in your remote controller then there should be more buttons.If there are more buttons then it will be more complicated.If you have less features then your remote controller won't be attractive,so I think uh we need
```

**Overlap analysis:** Look for rows where multiple agents have text (not —) at the same time step.

#### AFTER (overlaps cleaned)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 |
| --- | --- | --- | --- | --- |
| 0 | — | — | "'s" | — |
| 1 | — | — | " hard" | — |
| 2 | — | — | " but" | — |
| 3 | — | — | " I" | — |
| 4 | — | — | " think" | — |
| 5 | — | — | " it" | — |
| 6 | — | — | "'s" | — |
| 7 | — | — | " possible" | — |
| 8 | — | — | "but" | — |
| 9 | — | — | " it" | — |
| 10 | — | — | " uh" | — |
| 11 | — | — | "yeah" | — |
| 12 | — | — | " yeah" | — |
| 13 | " that" | — | — | — |
| 14 | " has" | — | — | — |
| 15 | " been" | — | — | — |
| 16 | " eliminated" | "Elim" | — | " that" |
| 17 | — | "inated" | — | — |
| 18 | — | — | — | — |
| 19 | — | — | " yeah" | — |
| 20 | — | — | "so" | — |
| 21 | — | — | " it" | — |
| 22 | — | — | "'s" | — |
| 23 | — | — | " it" | — |
| 24 | — | — | "'s" | — |
| 25 | — | — | " okay" | — |
| 26 | — | — | "," | — |
| 27 | — | — | " yeah" | — |
| 28 | — | — | "," | — |
| 29 | — | — | " yeah" | — |
| 30 | " M" | — | — | — |
| 31 | "m" | — | — | — |
| 32 | "-h" | — | — | — |
| 33 | "mm" | — | — | — |
| 34 | "." | — | — | — |
| 35 | — | — | "." | — |
| 36 | — | — | "And" | — |
| 37 | — | — | " uh" | — |
| 38 | — | — | " different" | — |
| 39 | — | — | " shapes" | — |

*(showing first 40 of 503 columns)*


**AFTER as readable dialogue:**

```text
[000-012] Agent 2: 's hard but I think it's possiblebut it uhyeah yeah
[013-016] Agent 0: that has been eliminated
[016-016] Agent 1: Elim
[016-016] Agent 3: that
[017-017] Agent 1: inated
[019-029] Agent 2: yeahso it's it's okay, yeah, yeah
[030-034] Agent 0: Mm-hmm.
[035-069] Agent 2: .And uh different shapes that we can dolike uh we can have you know a all animals shapes or you know comfortable uh whi which can fit into your handsand um
[070-086] Agent 3: Now that's good from a marketing point of view, the fun the fun shape.
[087-089] Agent 2: colours also,
[090-090] Agent 0: Yeah
[090-090] Agent 2: different
[090-091] Agent 3: And that
[092-092] Agent 0: I
[093-093] Agent 1: M
[093-093] Agent 2: colours
[093-093] Agent 3: you
[094-094] Agent 1: m
[094-094] Agent 3: you
[095-098] Agent 1: -hmm colours.
[099-099] Agent 2: and
[100-116] Agent 3: say that won't add too much to the budget?To d the shape is uh
[117-142] Agent 2: 't uh I don't think it will be like,you can have you know for uh if you want ther there to be more
[143-145] Agent 0: basically and uh
[145-145] Agent 2: Yeah
[146-146] Agent 0: you
[146-161] Agent 2: yeah.It's it's just a s shape so it doesn't matter.
[162-178] Agent 0: As the budget we're looking at if you build one mould I don't think that
[178-178] Agent 2: Yeah
[179-192] Agent 0: 's going to make a big difference whether it's gonna be square or
[193-219] Agent 3: Do you think there's any chance of um having ser in having basically the same machine with the same buttons but maybe several different shapes?
[220-222] Agent 2: Yeah that is
[223-223] Agent 0: Oh
[223-223] Agent 2: also
[224-224] Agent 0: yes
[224-224] Agent 2: possible
[226-229] Agent 3: Is that gonna be
[230-230] Agent 1: Yes
[230-230] Agent 2: I
[230-232] Agent 3: a possible?
[233-233] Agent 2: I
[233-241] Agent 3: 'Cause that might help with the marketing.
[242-244] Agent 0: will have to
[245-248] Agent 2: Yeah that will be
[249-266] Agent 0: on thatbut I think in principle that that would be that would be kind of fun,
[266-266] Agent 2: Yeah
[266-266] Agent 3: Because
[267-267] Agent 0: you
[267-267] Agent 3: we
[268-269] Agent 0: know.
[270-270] Agent 2: yeah
[270-274] Agent 3: had something sort of
[275-275] Agent 2: M
[275-275] Agent 3: sexy
[276-279] Agent 2: m-hmm.
[280-281] Agent 3: for adults
[282-282] Agent 0: Yeah
[282-287] Agent 3: and we could have something sort
[288-292] Agent 0: Silly for children.
[293-293] Agent 3: for
[294-294] Agent 2: for
[294-294] Agent 3: children
[295-295] Agent 2: children
[296-299] Agent 3: or a little animal
[300-300] Agent 0: or
[300-300] Agent 1: Like
[300-300] Agent 2: exactly
[300-300] Agent 3: shape
[301-301] Agent 1: a
[302-304] Agent 3: or in a
[305-306] Agent 2: that's
[307-317] Agent 3: or a little elephant so they can remember where it
[318-319] Agent 2: Yeah,
[320-320] Agent 1: Yes
[320-320] Agent 2: exactly
[320-321] Agent 3: is.
[323-383] Agent 2: . Yeah.And and the butto buttons also I think if you want to have more features in your remote controller then there should be more buttons.If there are more buttons then it will be more complicated.If you have less features then your remote controller won't be attractive,so I think uh we need
[385-393] Agent 0: Yeah wellyeah well that has been e
[395-398] Agent 3: Well we already eliminated
[401-411] Agent 0: so that's that's unfortunately a moot point now.
[413-429] Agent 2: so that uhYeah, yeah, yeahandNo no no, it won
[431-435] Agent 0: It just build a mould
[437-440] Agent 1: Yes exactly.
[442-443] Agent 0: know.
[447-449] Agent 2: I uh yeah
[451-456] Agent 0: I thinkI think we
[458-459] Agent 2: yeah.
[465-468] Agent 0: look at the budget
[470-472] Agent 3: ofsilly
[478-478] Agent 2: Yeah
[480-482] Agent 0: Like an animal
[484-490] Agent 2: , yeah.Yeah,
[492-494] Agent 1: doll, or
[496-498] Agent 2: what, yeah
[500-501] Agent 0: Yeah.
```

**Result:** 503 columns, 19 overlap columns (3.8%). The 74 overlapping segments were deferred to create clean handoffs.

**Coherence warning:** this prototype defers only the tokens inside an overlap span and appends them near the end. Inspect the decoded AFTER dialogue for split words, detached acknowledgements, or out-of-order replies; lower overlap alone does not prove the conversation remains valid.

---

## WEREWOLF Dataset

### Example 1 (dataset index 2232)

**Statistics:**
- Original: 307 cols, 69 overlap cols (22.5%)
- Cleaned: 424 cols, 2 overlap cols (0.5%)
- Reduction: 67 fewer overlap columns

- Removed punctuation-only runs: 4

#### BEFORE (overlaps present)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
| --- | --- | --- | --- | --- | --- |
| 0 | — | " had" | — | — | — |
| 1 | — | " one" | — | — | — |
| 2 | — | " vote" | — | — | — |
| 3 | — | "." | — | — | — |
| 4 | — | "Now" | — | "I" | — |
| 5 | — | "," | — | " voted" | — |
| 6 | — | " who" | — | " for" | — |
| 7 | — | " did" | — | " Mak" | — |
| 8 | — | " you" | — | "ay" | — |
| 9 | — | " vote" | — | "la" | — |
| 10 | — | " for" | — | "." | — |
| 11 | — | "?" | — | — | — |
| 12 | — | — | — | "Who" | — |
| 13 | — | — | — | " did" | — |
| 14 | — | — | — | " you" | — |
| 15 | — | — | — | " swap" | — |
| 16 | — | — | — | " with" | — |
| 17 | — | — | — | "?" | — |
| 18 | — | "Yeah" | — | "And" | "So" |
| 19 | — | "." | — | " you" | "," |
| 20 | — | — | — | " also" | " you" |
| 21 | — | — | — | " voted" | " all" |
| 22 | — | — | — | — | " tie" |
| 23 | — | — | — | — | "-" |
| 24 | "L" | — | — | "L" | "..." |
| 25 | "uis" | — | — | "uis" | " because" |
| 26 | "." | — | — | " was" | " you" |
| 27 | — | — | — | " the" | " both" |
| 28 | — | — | — | " narrator" | " voted" |
| 29 | — | — | — | "." | " for" |
| 30 | — | — | — | — | " Alan" |
| 31 | — | — | — | — | "-" |
| 32 | "Yeah" | — | — | — | — |
| 33 | "." | — | — | — | — |
| 34 | " So" | — | — | — | — |
| 35 | " we" | — | — | — | — |
| 36 | " tied" | — | — | — | — |
| 37 | "." | — | — | — | — |
| 38 | — | — | — | — | — |
| 39 | — | — | — | "and" | "..." |

*(showing first 40 of 307 columns)*


**BEFORE as readable dialogue:**

```text
[000-004] Agent 1: had one vote.Now
[004-004] Agent 3: I
[005-005] Agent 1: ,
[005-005] Agent 3: voted
[006-006] Agent 1: who
[006-006] Agent 3: for
[007-007] Agent 1: did
[007-007] Agent 3: Mak
[008-008] Agent 1: you
[008-008] Agent 3: ay
[009-009] Agent 1: vote
[009-009] Agent 3: la
[010-010] Agent 1: for
[010-010] Agent 3: .
[011-011] Agent 1: ?
[012-017] Agent 3: Who did you swap with?
[018-018] Agent 1: Yeah
[018-018] Agent 3: And
[018-018] Agent 4: So
[019-019] Agent 1: .
[019-019] Agent 3: you
[019-019] Agent 4: ,
[020-020] Agent 3: also
[020-020] Agent 4: you
[021-021] Agent 3: voted
[021-023] Agent 4: all tie-
[024-024] Agent 0: L
[024-024] Agent 3: L
[024-024] Agent 4: ...
[025-025] Agent 0: uis
[025-025] Agent 3: uis
[025-025] Agent 4: because
[026-026] Agent 0: .
[026-026] Agent 3: was
[026-026] Agent 4: you
[027-027] Agent 3: the
[027-027] Agent 4: both
[028-028] Agent 3: narrator
[028-028] Agent 4: voted
[029-029] Agent 3: .
[029-031] Agent 4: for Alan-
[032-037] Agent 0: Yeah. So we tied.
[039-039] Agent 3: and
[039-039] Agent 4: ...
[040-040] Agent 3: nobody
[040-040] Agent 4: and
[041-041] Agent 3: died
[041-041] Agent 4: then
[042-042] Agent 3: .
[042-053] Agent 4: they voted for Makayla.So, nobody died.
[054-082] Agent 0: Seth, sorry. I pointed. I've probably had too much wine. No, I swapped with Seth.It was a Mexican standoff.
[084-089] Agent 3: So, then I win.
[093-093] Agent 1: Yeah
[093-093] Agent 2: That
[093-093] Agent 3: That
[094-094] Agent 1: .
[094-094] Agent 2: was
[094-094] Agent 3: was
[095-095] Agent 2: nice
[095-095] Agent 3: -
[096-096] Agent 2: .
[097-105] Agent 3: ... not the way to win the game.
[107-130] Agent 1: Is it the wine? Or is there another pathogen in your blood, that is circling working its way forward?
[133-133] Agent 0: The
[133-133] Agent 1: It
[133-133] Agent 2: The
[133-133] Agent 4: No
[134-134] Agent 0: win
[134-134] Agent 1: 's
[134-134] Agent 2: win
[134-134] Agent 4: .
[135-135] Agent 0: .
[135-135] Agent 1: through
[135-135] Agent 2: .
[135-135] Agent 4: The
[136-136] Agent 1: the
[136-136] Agent 2: The
[136-136] Agent 4: win
[137-137] Agent 1: win
[137-137] Agent 2: win
[137-137] Agent 4: ?
[138-138] Agent 1: .
[138-138] Agent 2: .
[139-144] Agent 1: The win.The win.
[148-161] Agent 0: The win, the win, the win. What was my card?
[162-162] Agent 1: Oh
[162-162] Agent 2: I
[162-162] Agent 4: Did
[163-163] Agent 1: ,
[163-163] Agent 2: was
[163-163] Agent 4: you
[164-164] Agent 1: no
[164-164] Agent 2: the
[164-164] Agent 4: steal
[165-165] Agent 1: ,
[165-165] Agent 2: Trou
[165-165] Agent 4: the
[166-166] Agent 1: no
[166-166] Agent 2: blem
[166-166] Agent 4: tech
[167-167] Agent 1: .
[167-167] Agent 2: aker
[167-167] Agent 4: ?
[168-168] Agent 1: Yeah
[168-168] Agent 2: ,
[168-168] Agent 4: It
[169-169] Agent 1: .
[169-169] Agent 2: and
[169-169] Agent 4: was
[170-170] Agent 1: What
[170-170] Agent 2: I
[170-170] Agent 4: tech
[171-171] Agent 1: are
[171-171] Agent 2: swapped
[171-171] Agent 4: .
[172-172] Agent 1: you
[172-172] Agent 2: yours
[172-172] Agent 4: I
[173-173] Agent 1: talking
[173-173] Agent 2: and
[173-173] Agent 4: 'm
[174-174] Agent 1: about
[174-174] Agent 2: Kevin
[174-174] Agent 4: just
[175-175] Agent 1: ?
[175-175] Agent 2: 's
[175-175] Agent 4: kidding
[176-176] Agent 2: cards
[176-176] Agent 4: .
[177-199] Agent 2: .Yeah. All right. You creeped to me like nobody  He's not afraid or anything like that.
[202-211] Agent 0: Okay, so now we're really shuffling.
[213-213] Agent 1: You
[213-213] Agent 4: It
[214-214] Agent 1: Sw
[214-214] Agent 4: was
[215-215] Agent 1: apped
[215-215] Agent 4: tech
[216-216] Agent 1: our
[216-216] Agent 4: .
[217-233] Agent 1: cards? Okay, I know you're the Werewolf.We built on it.
[237-241] Agent 2: Well, I mean-
[242-246] Agent 0: How do you know?
[247-255] Agent 1: We're built on SpongeBob memes.Because
[255-255] Agent 2: ...
[256-256] Agent 1: I
[256-256] Agent 2: everyone
[257-257] Agent 1: was
[257-257] Agent 2: watching
[258-258] Agent 1: the
[258-258] Agent 2: this
[259-259] Agent 1: Wer
[259-259] Agent 2: is
[260-260] Agent 1: ewolf
[260-260] Agent 2: going
[261-261] Agent 1: ,
[261-261] Agent 2: to
[262-262] Agent 1: but
[262-262] Agent 2: be
[263-263] Agent 1: I
[263-263] Agent 2: our
[264-264] Agent 1: 'm
[264-264] Agent 2: age
[265-265] Agent 1: not
[265-265] Agent 2: .
[266-266] Agent 1: the
[266-266] Agent 2: I
[267-267] Agent 1: Wer
[267-267] Agent 2: don
[268-268] Agent 1: ewolf
[268-268] Agent 2: 't
[269-269] Agent 1: anymore
[269-269] Agent 2: know
[270-270] Agent 1: .
[270-274] Agent 2: what's that like.
[275-275] Agent 0: Oh
[275-275] Agent 4: That
[276-276] Agent 0: ,
[276-276] Agent 4: 's
[277-277] Agent 0: yeah
[277-277] Agent 4: ...
[278-278] Agent 0: .
[278-282] Agent 4: Yeah, that's...
[283-306] Agent 0: Oh, yeah. We're going to have a bunch of old professors watching this video.But you're also trying to
```

**Overlap analysis:** Look for rows where multiple agents have text (not —) at the same time step.

#### AFTER (overlaps cleaned)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
| --- | --- | --- | --- | --- | --- |
| 0 | — | " had" | — | — | — |
| 1 | — | " one" | — | — | — |
| 2 | — | " vote" | — | — | — |
| 3 | — | "." | — | — | — |
| 4 | — | — | — | "I" | — |
| 5 | — | — | — | " voted" | — |
| 6 | — | — | — | " for" | — |
| 7 | — | — | — | " Mak" | — |
| 8 | — | — | — | "ay" | — |
| 9 | — | — | — | "la" | — |
| 10 | — | — | — | "." | — |
| 11 | — | — | — | — | — |
| 12 | — | — | — | "Who" | — |
| 13 | — | — | — | " did" | — |
| 14 | — | — | — | " you" | — |
| 15 | — | — | — | " swap" | — |
| 16 | — | — | — | " with" | — |
| 17 | — | — | — | "?" | — |
| 18 | — | — | — | — | "So" |
| 19 | — | — | — | — | "," |
| 20 | — | — | — | — | " you" |
| 21 | — | — | — | — | " all" |
| 22 | — | — | — | — | " tie" |
| 23 | — | — | — | — | "-" |
| 24 | — | — | — | — | "..." |
| 25 | — | — | — | — | " because" |
| 26 | — | — | — | — | " you" |
| 27 | — | — | — | — | " both" |
| 28 | — | — | — | — | " voted" |
| 29 | — | — | — | — | " for" |
| 30 | — | — | — | — | " Alan" |
| 31 | — | — | — | — | "-" |
| 32 | "Yeah" | — | — | — | — |
| 33 | "." | — | — | — | — |
| 34 | " So" | — | — | — | — |
| 35 | " we" | — | — | — | — |
| 36 | " tied" | — | — | — | — |
| 37 | "." | — | — | — | — |
| 38 | — | — | — | — | — |
| 39 | — | — | — | — | "..." |

*(showing first 40 of 424 columns)*


**AFTER as readable dialogue:**

```text
[000-003] Agent 1: had one vote.
[004-017] Agent 3: I voted for Makayla.Who did you swap with?
[018-031] Agent 4: So, you all tie-... because you both voted for Alan-
[032-037] Agent 0: Yeah. So we tied.
[039-053] Agent 4: ... and then they voted for Makayla.So, nobody died.
[054-082] Agent 0: Seth, sorry. I pointed. I've probably had too much wine. No, I swapped with Seth.It was a Mexican standoff.
[084-089] Agent 3: So, then I win.
[093-093] Agent 1: Yeah
[093-093] Agent 2: That
[093-093] Agent 3: That
[094-094] Agent 2: was
[094-094] Agent 3: was
[095-095] Agent 2: nice
[097-105] Agent 3: ... not the way to win the game.
[107-130] Agent 1: Is it the wine? Or is there another pathogen in your blood, that is circling working its way forward?
[133-138] Agent 2: The win.The win.
[139-144] Agent 1: The win.The win.
[148-161] Agent 0: The win, the win, the win. What was my card?
[162-176] Agent 4: Did you steal the tech? It was tech. I'm just kidding.
[177-199] Agent 2: .Yeah. All right. You creeped to me like nobody  He's not afraid or anything like that.
[202-211] Agent 0: Okay, so now we're really shuffling.
[213-216] Agent 4: It was tech.
[217-233] Agent 1: cards? Okay, I know you're the Werewolf.We built on it.
[237-241] Agent 2: Well, I mean-
[242-246] Agent 0: How do you know?
[247-254] Agent 1: We're built on SpongeBob memes.
[255-274] Agent 2: ... everyone watching this is going to be our age. I don't know what's that like.
[275-282] Agent 4: That's...Yeah, that's...
[283-306] Agent 0: Oh, yeah. We're going to have a bunch of old professors watching this video.But you're also trying to
[308-314] Agent 1: Now, who did you vote for
[316-319] Agent 3: And you also voted
[321-322] Agent 1: Yeah.
[326-331] Agent 3: Luis was the narrator.
[333-335] Agent 0: Luis.
[340-343] Agent 3: and nobody died.
[345-350] Agent 1: It's through the win.
[352-356] Agent 4: No.The win?
[359-361] Agent 0: The win.
[366-380] Agent 2: I was the Troublemaker, and I swapped yours and Kevin's cards
[382-418] Agent 1: Oh, no, no.Yeah. What are you talking about?You Swapped ourBecause I was the Werewolf, but I'm not the Werewolf anymore.
[420-423] Agent 0: Oh, yeah.
```

**Result:** 424 columns, 2 overlap columns (0.5%). The 67 overlapping segments were deferred to create clean handoffs.

**Coherence warning:** this prototype defers only the tokens inside an overlap span and appends them near the end. Inspect the decoded AFTER dialogue for split words, detached acknowledgements, or out-of-order replies; lower overlap alone does not prove the conversation remains valid.

---

### Example 2 (dataset index 2299)

**Statistics:**
- Original: 255 cols, 65 overlap cols (25.5%)
- Cleaned: 355 cols, 3 overlap cols (0.8%)
- Reduction: 62 fewer overlap columns

- Removed punctuation-only runs: 2

#### BEFORE (overlaps present)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Agent 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | — | — | " on" | — | — | — |
| 1 | — | — | "." | — | — | — |
| 2 | — | — | — | — | — | — |
| 3 | — | — | — | — | "There" | — |
| 4 | — | — | — | — | "'s" | — |
| 5 | — | — | — | — | " also" | — |
| 6 | — | — | — | — | " two" | — |
| 7 | — | — | — | — | " cards" | — |
| 8 | — | — | — | — | " in" | — |
| 9 | — | — | — | — | " the" | — |
| 10 | — | — | — | — | " middle" | — |
| 11 | — | — | — | — | " that" | — |
| 12 | — | — | — | — | " we" | — |
| 13 | — | — | — | — | "..." | — |
| 14 | — | — | — | — | — | — |
| 15 | — | — | — | — | — | — |
| 16 | — | — | — | — | — | "Don" |
| 17 | — | — | — | — | — | "'t" |
| 18 | — | — | — | — | — | " know" |
| 19 | — | — | — | — | — | "." |
| 20 | — | — | — | — | — | — |
| 21 | — | — | — | — | — | — |
| 22 | — | — | — | — | "Don" | — |
| 23 | — | — | — | — | "'t" | — |
| 24 | — | — | — | — | " know" | — |
| 25 | — | — | — | — | "." | — |
| 26 | — | — | — | — | — | — |
| 27 | — | — | "But" | — | "There" | — |
| 28 | — | — | " one" | — | "'s" | — |
| 29 | — | — | " of" | — | " one" | — |
| 30 | — | — | " them" | — | " card" | — |
| 31 | — | — | " is" | — | " here" | — |
| 32 | — | — | " out" | — | " that" | — |
| 33 | — | — | " because" | — | " we" | — |
| 34 | — | — | " of" | — | " don" | — |
| 35 | — | — | " the" | — | "'t" | — |
| 36 | — | — | " drunk" | — | " know" | — |
| 37 | — | — | "." | — | "." | — |
| 38 | — | — | "Because" | — | "Yeah" | — |
| 39 | — | — | " the" | — | "," | — |

*(showing first 40 of 255 columns)*


**BEFORE as readable dialogue:**

```text
[000-001] Agent 2: on.
[003-013] Agent 4: There's also two cards in the middle that we...
[016-019] Agent 5: Don't know.
[022-025] Agent 4: Don't know.
[027-027] Agent 2: But
[027-027] Agent 4: There
[028-028] Agent 2: one
[028-028] Agent 4: 's
[029-029] Agent 2: of
[029-029] Agent 4: one
[030-030] Agent 2: them
[030-030] Agent 4: card
[031-031] Agent 2: is
[031-031] Agent 4: here
[032-032] Agent 2: out
[032-032] Agent 4: that
[033-033] Agent 2: because
[033-033] Agent 4: we
[034-034] Agent 2: of
[034-034] Agent 4: don
[035-035] Agent 2: the
[035-035] Agent 4: 't
[036-036] Agent 2: drunk
[036-036] Agent 4: know
[037-037] Agent 2: .
[037-037] Agent 4: .
[038-038] Agent 2: Because
[038-038] Agent 4: Yeah
[039-039] Agent 2: the
[039-039] Agent 4: ,
[040-040] Agent 2: dr
[040-040] Agent 4: but
[041-041] Agent 2: unks
[041-041] Agent 4: we
[042-042] Agent 2: swap
[042-042] Agent 4: don
[043-043] Agent 2: from
[043-043] Agent 4: 't
[044-044] Agent 2: the
[044-044] Agent 4: know
[045-045] Agent 2: middle
[045-045] Agent 4: what
[046-046] Agent 2: .
[046-048] Agent 4: she is.
[049-049] Agent 2: Right
[049-049] Agent 4: So
[050-050] Agent 2: .
[050-060] Agent 4: she could be a werewolf or the tanner.
[064-064] Agent 0: I
[064-064] Agent 3: I
[064-064] Agent 4: But
[065-065] Agent 0: have
[065-065] Agent 3: wouldn
[065-065] Agent 4: you
[066-066] Agent 0: no
[066-066] Agent 3: 't
[066-066] Agent 4: wouldn
[067-067] Agent 0: idea
[067-067] Agent 3: put
[067-067] Agent 4: 't
[068-068] Agent 0: who
[068-068] Agent 3: the
[068-068] Agent 4: know
[069-069] Agent 0: 's
[069-069] Agent 3: Tanner
[069-069] Agent 4: .
[070-070] Agent 0: the
[070-070] Agent 3: on
[071-071] Agent 0: drunk
[071-071] Agent 3: her
[072-072] Agent 0: .
[072-072] Agent 3: .
[073-088] Agent 2: I realize that we should be able to make noise. Shuffle in our chairs.
[089-097] Agent 4: No, I mean, I wouldn't...
[100-111] Agent 2: So she's safe to kill. Just, you know..
[112-115] Agent 4: I would put-
[116-116] Agent 0: I
[116-116] Agent 2: I
[116-116] Agent 3: There
[117-117] Agent 0: don
[117-117] Agent 2: don
[117-117] Agent 3: could
[118-118] Agent 0: 't
[118-118] Agent 2: 't
[118-118] Agent 3: have
[119-119] Agent 0: know
[119-119] Agent 2: remember
[119-119] Agent 3: been
[120-120] Agent 0: guys
[120-120] Agent 2: what
[120-120] Agent 3: a
[121-121] Agent 0: .
[121-121] Agent 2: I
[121-121] Agent 3: wer
[122-122] Agent 2: did
[122-122] Agent 3: ewolf
[123-123] Agent 2: .
[123-123] Agent 3: in
[124-124] Agent 2: Exactly
[124-124] Agent 3: the
[125-125] Agent 2: .
[125-125] Agent 3: middle
[126-126] Agent 2: Wh
[126-126] Agent 3: .
[127-128] Agent 2: ack.
[129-132] Agent 1: Our poor chairs.
[133-137] Agent 0: Just side to side.
[138-143] Agent 2: Just bounce up and down.
[145-149] Agent 0: Not up and down.
[150-150] Agent 3: Maybe
[150-150] Agent 4: I
[151-151] Agent 3: I
[151-151] Agent 4: would
[152-152] Agent 3: 'll
[152-152] Agent 4: put
[153-153] Agent 3: just
[153-153] Agent 4: the
[154-154] Agent 3: whistle
[154-160] Agent 4: Tanner over here just because-That
[160-160] Agent 5: That
[161-161] Agent 4: way
[161-161] Agent 5: 's
[162-162] Agent 4: for
[162-162] Agent 5: true
[163-163] Agent 4: sure
[163-163] Agent 5: .
[164-194] Agent 4: nobody wins, because we didn't have the Tanner. Oh wait, then that's where Villagers were. N Nevermind.Me
[194-194] Agent 5: I
[195-195] Agent 4: neither
[195-195] Agent 5: can
[196-196] Agent 4: .
[196-202] Agent 5: 't risk some of these cards.
[205-209] Agent 2: You can stop now.
[213-213] Agent 1: Z
[213-213] Agent 5: There
[214-214] Agent 1: oe
[214-214] Agent 5: could
[215-215] Agent 1: ,
[215-215] Agent 5: be
[216-216] Agent 1: that
[216-216] Agent 5: no
[217-217] Agent 1: 's
[217-217] Agent 5: minion
[218-218] Agent 1: enough
[218-218] Agent 5: too
[219-219] Agent 1: .
[219-219] Agent 5: .
[220-220] Agent 1: Uh
[220-220] Agent 2: Right
[221-221] Agent 1: -
[221-221] Agent 2: .
[222-222] Agent 1: oh
[222-222] Agent 2: Let
[223-223] Agent 1: .
[223-230] Agent 2: 's be petty. Who heard shit?
[237-251] Agent 1: I already told you, I felt Hamza reach for the drunk title.
```

**Overlap analysis:** Look for rows where multiple agents have text (not —) at the same time step.

#### AFTER (overlaps cleaned)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Agent 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | — | — | " on" | — | — | — |
| 1 | — | — | "." | — | — | — |
| 2 | — | — | — | — | — | — |
| 3 | — | — | — | — | "There" | — |
| 4 | — | — | — | — | "'s" | — |
| 5 | — | — | — | — | " also" | — |
| 6 | — | — | — | — | " two" | — |
| 7 | — | — | — | — | " cards" | — |
| 8 | — | — | — | — | " in" | — |
| 9 | — | — | — | — | " the" | — |
| 10 | — | — | — | — | " middle" | — |
| 11 | — | — | — | — | " that" | — |
| 12 | — | — | — | — | " we" | — |
| 13 | — | — | — | — | "..." | — |
| 14 | — | — | — | — | — | — |
| 15 | — | — | — | — | — | — |
| 16 | — | — | — | — | — | "Don" |
| 17 | — | — | — | — | — | "'t" |
| 18 | — | — | — | — | — | " know" |
| 19 | — | — | — | — | — | "." |
| 20 | — | — | — | — | — | — |
| 21 | — | — | — | — | — | — |
| 22 | — | — | — | — | "Don" | — |
| 23 | — | — | — | — | "'t" | — |
| 24 | — | — | — | — | " know" | — |
| 25 | — | — | — | — | "." | — |
| 26 | — | — | — | — | — | — |
| 27 | — | — | — | — | "There" | — |
| 28 | — | — | — | — | "'s" | — |
| 29 | — | — | — | — | " one" | — |
| 30 | — | — | — | — | " card" | — |
| 31 | — | — | — | — | " here" | — |
| 32 | — | — | — | — | " that" | — |
| 33 | — | — | — | — | " we" | — |
| 34 | — | — | — | — | " don" | — |
| 35 | — | — | — | — | "'t" | — |
| 36 | — | — | — | — | " know" | — |
| 37 | — | — | — | — | "." | — |
| 38 | — | — | — | — | "Yeah" | — |
| 39 | — | — | — | — | "," | — |

*(showing first 40 of 355 columns)*


**AFTER as readable dialogue:**

```text
[000-001] Agent 2: on.
[003-013] Agent 4: There's also two cards in the middle that we...
[016-019] Agent 5: Don't know.
[022-048] Agent 4: Don't know.There's one card here that we don't know.Yeah, but we don't know what she is.
[049-049] Agent 2: Right
[049-060] Agent 4: So she could be a werewolf or the tanner.
[064-072] Agent 3: I wouldn't put the Tanner on her.
[073-088] Agent 2: I realize that we should be able to make noise. Shuffle in our chairs.
[089-097] Agent 4: No, I mean, I wouldn't...
[100-111] Agent 2: So she's safe to kill. Just, you know..
[112-115] Agent 4: I would put-
[116-126] Agent 3: There could have been a werewolf in the middle.
[127-128] Agent 2: ack.
[129-132] Agent 1: Our poor chairs.
[133-137] Agent 0: Just side to side.
[138-143] Agent 2: Just bounce up and down.
[145-149] Agent 0: Not up and down.
[150-159] Agent 4: I would put the Tanner over here just because-
[160-163] Agent 5: That's true.
[164-194] Agent 4: nobody wins, because we didn't have the Tanner. Oh wait, then that's where Villagers were. N Nevermind.Me
[194-194] Agent 5: I
[195-195] Agent 4: neither
[195-202] Agent 5: can't risk some of these cards.
[205-209] Agent 2: You can stop now.
[213-223] Agent 1: Zoe, that's enough.Uh-oh.
[224-230] Agent 2: be petty. Who heard shit?
[237-251] Agent 1: I already told you, I felt Hamza reach for the drunk title.
[256-275] Agent 2: But one of them is out because of the drunk.Because the drunks swap from the middle.
[277-285] Agent 0: I have no idea who's the drunk.
[287-292] Agent 4: But you wouldn't know.
[297-307] Agent 2: I don't remember what I did. Exactly. Wh
[309-314] Agent 0: I don't know guys.
[321-325] Agent 3: Maybe I'll just whistle
[327-330] Agent 4: That way for sure
[332-338] Agent 5: There could be no minion too.
[351-354] Agent 2: Right. Let's
```

**Result:** 355 columns, 3 overlap columns (0.8%). The 62 overlapping segments were deferred to create clean handoffs.

**Coherence warning:** this prototype defers only the tokens inside an overlap span and appends them near the end. Inspect the decoded AFTER dialogue for split words, detached acknowledgements, or out-of-order replies; lower overlap alone does not prove the conversation remains valid.

---
