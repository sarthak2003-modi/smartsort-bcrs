# Ethics, limitations & responsible use

Drawn from the SmartSort NTU AI Literacy final deck (Honest Scope + Responsible AI).

## Limitations & assumptions

**What this prototype does not do yet**

- Trained on a small, self-collected image set — real deployment needs far more
  data across lighting, angles and container wear.
- No live load-cell or NIR sensor in the trial; the app takes a **manual weight
  entry** instead.
- Confidence and weight-band thresholds are **illustrative, not calibrated**
  against real machine data.
- Runs as a browser / Python simulation, not integrated into physical RVM hardware.
- **This GitHub repo ships no trained weights and no measured accuracy numbers.**

**Assumptions**

- Deposit marks stay printed and legible even on worn or crumpled containers.
- Camera framing and chute lighting stay reasonably consistent in the field.
- Operators review held items rather than discarding them outright.
- Observed fraud mostly resembles the contaminant categories we trained on.

## Responsible AI

### Bias & fairness

Training data must span diverse brands, lighting and wear, or the system may
systematically reject valid returns from people whose containers look different —
wrongly denying genuine recyclers their refund.

### Privacy

The camera should frame **only the chute**, never bystanders. Images are used
solely for the eligibility check and should not be retained longer than needed,
with **no facial recognition**.

### Responsible use

A held item should trigger a **human-reviewable appeal path**, not an automatic,
unexplained denial. The system assists operators; it does not replace accountability.

### Transparency

Clear signage should explain how verification works, so users understand why an
item was accepted or held, and who to approach if they disagree.

## Next steps (from the deck)

1. Expand the dataset across real-world lighting, wear and brand variety.
2. Pilot with an actual load cell and NIR material sensor on a test rig.
3. Build out the fraud-flagging dashboard for TOMRA, RVM Systems and SG Recycle.
