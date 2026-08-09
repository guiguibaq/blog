---
title: Nestor - the AI in the phone
summary: Wiring an AI into the telephone network turns out to be easy. Getting people to enjoy talking to it is the hard part.
date: 2026-08-09

# Featured image
image:
  caption: 'Nestor takes a call. Image credit: [**Tintinomania**](https://tintinomania.com)'

authors:
  - me
---

Can you plug an AI into the telephone network and have it hold a real
conversation? I wanted to find out — and, more interesting to me, to see how
people react when the phone rings and it isn't a person.

The tech part turned out to be easy — a weekend's work with Claude's help, and
about $30 in OpenAI and Twilio credits.

The social part is where I had a few surprises.

## The tech

The code is on GitHub: [**nestor**](https://github.com/guiguibaq/nestor)
(MIT, ~1,900 lines). You can try it yourself: add a contact and a prompt, then the following lines will call for you:

```bash
uv run call --contact alex --prompt catchup-dinner --dry-run   # rings you
uv run call --contact alex --prompt catchup-dinner --run       # rings Alex
```

Three moving parts:

1. A script asks **Twilio** to place the call.
2. When the person answers, Twilio fetches instructions from my laptop (via a
   tunnel), which speak an AI disclosure and then open a **WebSocket** carrying
   the call audio.
3. That socket is bridged to the **OpenAI Realtime API**.

Two things surprised me. First, **no audio conversion anywhere** — Twilio speaks
G.711 μ-law at 8 kHz and the Realtime API does too, if you ask. Every design I
read beforehand described resampling to 24 kHz and back; that component doesn't
need to exist. Second, the genuinely fiddly part is **interruptions**: audio
streams to Twilio much faster than realtime, so when someone talks over the AI
there's a second or more already sent but not yet heard. Get it wrong and the AI
refers back to sentences nobody received.

## The first calls

So far I tested on a handful of friends and family. Technically it holds up — around 630 ms to respond, handles being interrupted, switched to French unprompted, and arranges a dinner without me.

It does not, in my view, pass the Turing test. The voice is excellent and the
reactions are convincing, but you can tell. Something is slightly off — a small
lag before it answers, a delivery that is *too* clean, no stumbles or
half-finished words. It doesn't quite flow the way people do. Very close, but
you can tell it's not a real human.

Socially, my hypothesis was wrong. I gave Nestor one personal detail about each
person — how a recent trip went, how the new job was going, that sort of thing —
on the theory that a caller who knows a little about you feels warm rather than
robotic, and that people would open up.

The calls went the same way every time. Nestor introduces itself as an AI:
people are amused, and happy to keep talking. Then it asks the personal
question — and they hang up.

I can't cleanly separate the two explanations. It may be that a machine knowing
something about you is creepy, whatever the something is; note that *I* supplied
those details, but the person on the line has no way of knowing that. Or it may
simply be that nobody wants to make small talk with an AI. Probably both.

Either way the lesson is the same: **the AI shouldn't try to be
your mate.** Small talk is what breaks the interaction. It works far better as a
polite assistant with an obvious purpose — say why you're calling, do the thing,
get off the phone.

My dad, bless him, found the whole thing delightful and had a lovely bit of
small talk with the AI.

## Next

Three directions.

**Give it a real task.** Booking a restaurant is the obvious test: the other end
is expecting a transaction rather than a chat, which is exactly the mode that
seems to work.

**Stop people hanging up.** Now that I know small talk is what breaks it, there's
a lot to try — how it opens, how much it explains, how quickly it gets to the
point. Granted I have enough friends still willing to pick up!

**Think about misuse.** This is the part that nags. If a weekend project can hold
a convincing phone conversation, what happens to everyone whose work runs on a
phone line — your local vet, a car-rental call centre? Not one nuisance call, but
the possibility of generating them at a scale a human queue was never built to
absorb. I don't have a good answer, but it seems worth asking before rather than
after.
