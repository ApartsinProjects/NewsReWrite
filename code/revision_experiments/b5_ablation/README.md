# B5 Guide-ablation

The four-cell grid produced by `b2_held_out_rewrite_eval/run_rewrites.py` IS
the ablation. There is no separate runner:

| cell | alpha | beta | reads as |
| --- | --- | --- | --- |
| off / off | 0.0 | 0.0 | prompt-only baseline inside FUDGE loop |
| pos only  | 0.7 | 0.0 | tactics guide on, clickbait brake off |
| neg only  | 0.0 | 0.7 | tactics guide off, clickbait brake on |
| both      | 0.7 | 0.7 | full method |

`report.py` reformats the b2 summary into an ablation table that is easier
to paste into the paper.
