# Speaker 2 script · Slides 4–5 (System design & computer vision)

**Target: 2–2.5 minutes** · Sarthak "Conti" Modi

---

### Slide 4 — System design: four checks before a payout (~60–70 s)

“Moving to system design — SmartSort runs **four checks before any payout**.

**One — Capture.** A chute camera photographs whatever was just inserted.

**Two — Classify.** A CNN vision model names the category — PET, aluminium,
rubbish, food, crushed or tampered — and gives a confidence score.

**Three — Weigh and sense.** A load cell checks mass against empty-container
bands — about **8 to 70 grams** for PET, **10 to 25 grams** for cans — and an
NIR or capacitive sensor checks the material really is plastic or aluminium.

**Four — Fuse and decide.** Those signals combine into **one confidence score**.
If the item is eligible, empty, and in-band, we **PAY ten cents**. Otherwise we
**HOLD** it for an operator — we don’t silently pay fraud, and we don’t
auto-discard a genuine recycler’s bottle without review.”

---

### Slide 5 — Computer vision methods (~60–80 s)

“On the vision side we use three methods.

**First, object detection and classification.** A YOLO-family detector —
we default to **YOLOv8n** for edge size, with **YOLO11n** as a drop-in
alternative — recognises container type, shape and condition straight from
the chute image.

**Second, transfer learning.** We take pretrained **MobileNet or ResNet**
backbones, freeze them, train a small head on BCRS items, then fine-tune.
That keeps inference realistic for low-power hardware inside the RVM.

**Third, OCR and deposit-mark scanning.** We read the barcode and BCRS
deposit mark so we can confirm it’s a genuine participating container, not
just something bottle-shaped.

Vision alone can still be fooled — so the next slide’s sensors catch what the
camera can’t — but these three CV pieces are how we *see* the return before we
ever weigh it.”

---

### Timing tips

- Speak the PAY vs HOLD contrast clearly; it’s the slide’s visual punchline.
- If short on time, keep YOLO + transfer learning and trim OCR to one sentence.
- If long, add one concrete fraud example: “food waste crushed into a bottle shape.”
