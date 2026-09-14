# E17 into the paper (item 45), proposed 2026-09-14. Nothing applied.

Registry block `e17_boundary`. All numbers below are from it.

## 1. Section 8, the Yee paragraph: replace the hedged closing clause

C: "We read their results as characterizing in-flight values and ours as characterizing committed ones, and we treat the boundary as the hypothesis for the gap rather than an established cause."

P: "We read their results as characterizing in-flight values and ours as characterizing committed ones. A direct test on their own items supports the direction but not the size: completing the statement after the corrupted numeral lowers GPT-4o's recovery by about 9 points in each of two numberings (pooled paired $p=0.08$ over 90 items; Appendix~\ref{app:renumber}), so the boundary accounts for part of the gap and the rest lies in the task and the grader."

[`e17_boundary._prose`: orig 55.6 to 46.7, renum 40.0 to 31.1, pooled diff +8.9, pooled McNemar p = 0.0768, n = 90.]

## 2. Appendix A.5: new paragraph after the renumbering paragraph

P:
> \paragraph{Statement-boundary test.} To test whether resuming mid-statement is what lets their models recover, we took the 45 original-number items from the renumbering replication whose original chain continues past the corrupted numeral without another number before the end of the sentence, and their 45 renumbered twins. For each we built two prefixes identical up to the corrupted numeral: one ending on the numeral, as in their protocol, and one ending after the sentence's own closing words taken verbatim from the model's original chain (median one word). Same model, prompt, and decoding as the replication; uncorrupted prefixes at both boundaries as controls; 360 calls. GPT-4o recovers 55.6 percent of original items at the numeral boundary and 46.7 percent at the statement boundary; on renumbered items, 40.0 and 31.1 percent. The paired difference is 8.9 points in both numberings (pooled over 90 items: 31 recovered at both boundaries, 12 only at the numeral, 4 only after the statement, 43 at neither; exact McNemar $p=0.08$). Propagation rises by the same amount. Uncorrupted prefixes are unaffected (93.3 and 97.8 percent correct). Completing the statement therefore moves recovery toward propagation, in the registered direction, but by far less than the gap between their recovery rates and ours; the boundary is part of the explanation, not the whole of it. Reading the continuations shows the in-flight mechanism directly: at the numeral boundary the model sometimes finishes the number itself, writing the correct value's remaining digits after the planted one, which no statement-boundary continuation can do. The test was registered before the run (specification and amendment in the experiment archive).

## 3. What changes elsewhere
- Nothing in the abstract or intro depends on the boundary claim.
- The "More fundamentally" sentence in Sec 8 (item 44/A4) stays; only its closing clause changes as above.
- The registration and amendment are committed (201a0dd, ec7717d) and the artifacts are archived under `e17_boundary/`.

## 4. Honest notes
- 45 paired items per numbering is small; the pooled p of 0.08 is suggestive, not significant at 0.05. The paper text says "supports the direction but not the size" and prints the flip table.
- The digit-append observation ("= 2" continued as "5 years old", i.e. 25 written but 5 intended) is from reading four samples; it is stated as an observation, not counted. If you want it counted, the scorer can flag continuations that begin with a digit (cheap, no new calls).
