## Writing the paper (paper/)

When writing or editing anything in paper/, you are writing an ICLR paper
for a tired reviewer, not documentation.

Structure:
- Full paragraphs only. No bullet lists, no bold-led lists. The only list
  in the paper is the contributions list at the end of the intro.
- One point per paragraph. First sentence states the claim; the rest is
  evidence. A reader skimming only first sentences should get the whole argument.
- Each section opens with 1-2 sentences orienting the reader: what this
  section does and why it comes now.

Sentences:
- Short. If a sentence needs a semicolon or has two subordinate clauses, split it.
- Active voice, "we". Present tense for the paper's claims, past tense for
  what experiments did.
- No throat-clearing ("It is worth noting", "In recent years") and no hype
  ("novel", "remarkably"). State the thing.

Precision:
- One name per concept, used identically everywhere. Define notation before
  first use.
- Concrete before abstract: give intuition or an example, then the general form.
- Numbers over adjectives ("+4.2 accuracy", not "substantially better").
  Every number must trace to a file in paper/results/ — if you can't point
  to the source, write \todo{verify} instead of a number.
- After every reported result, one sentence of interpretation: what should
  the reader conclude.

Process:
- Before drafting a section, read the relevant source and results files.
- Draft one section per request, then stop. Never touch sections I didn't ask about.
