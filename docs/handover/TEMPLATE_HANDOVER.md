# Agent Session Handover Template

Copy this block to the **top** of your module handover file (`docs/handover/<teammate>_<module>.md`) under **Session History** before ending every coding session, and also add a 1-row summary to `HANDOVER.md`.

```markdown
### Session: YYYY-MM-DD HH:MM (UTC+4) — [Teammate Name] / [Agent Name]
- **Git Branch & Base Commit:** `feat/<teammate>-<topic>` (branched from `<commit-sha>`)
- **Objective:** What task or bugfix was targeted.
- **Files Modified / Added:**
  - `path/to/file.py`: What changed and why.
- **Shared Contract Check:**
  - [ ] 21-joint MANO order (`0..20`) & `PARENT_INDICES` untouched / verified
  - [ ] `scale_mm = 100.0` & 4D node features `(B, T, 21, 4)` preserved
  - [ ] `data/splits/split_indices.json` untouched
- **Commands & Tests Executed (Exact Observed Results):**
  - `python tests/test_kinematics.py` -> PASS / FAIL
  - Other scripts run (`evaluate_and_plot.py`, `benchmark_edge.py`, etc.) -> exact key metrics
- **Blockers / Known Issues:** Any hardware, OS, or dependency issues encountered.
- **Handover to Next Agent / Teammate:** Exact next command to run or interface change teammates must pull before editing.
```
