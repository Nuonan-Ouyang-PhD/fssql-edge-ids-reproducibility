# Pi4B8 Q preregistration erratum V1

The frozen `pi4b8_q_training_prereg_v1.json` inherited two unused metadata
items: the old 4101→3101… seed map and Pi3B+ near-cap references. The trainer
does not read the seed-map field, and it uses the near-cap paths only for
generic SHA-256 integrity checks. They are excluded from the Pi4B8 numeric
training dependency graph and did not affect the accepted tables.

The formal mapping is therefore fixed as identity:

`5101→5101`, `5102→5102`, `5103→5103`, `5104→5104`, `5105→5105`.

No preregistration JSON was modified; this is an additive clarification.
