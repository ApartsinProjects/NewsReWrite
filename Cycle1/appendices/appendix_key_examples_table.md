# Appendix: Controlled Headline Generation — Key Examples

A curated set of examples demonstrating the core findings: engagement-tactic activation (alpha), clickbait avoidance (beta), and the fidelity trade-off. All numbers are from the authoritative run.

**Authoritative run (Cycle 2).** Meta-Llama-3-8B-Instruct, log-domain FUDGE objective, top-k = 50, cloud A10G/L40S via Modal. Guidance weights swept over alpha, beta in {0, 1, 2, 4}. Clickbait probability (cb) is from an independent DistilBERT detector trained only on human-authored Chakraborty data; STS is Sentence-Transformer cosine to the source. These appendices supersede the earlier exploratory top-k{50,500}, {0,1,2} sweep.

---

## A.1 Tactic activation (alpha) raises engagement and induced clickbait

| Source | Tactic | alpha | beta | Generated | cb (ext.) | STS |
|---|---|---|---|---|---|---|
| 'Numerous' arrests made during Trump i | Curiosity + Question | 0 | 0 | Behind the Scenes: What Led to the Wave of Arrests During Trump's Inauguration Protests?". | 0.035 | 0.76 |
| 'Numerous' arrests made during Trump i | Curiosity + Question | 4 | 0 | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 0.164 | 0.59 |
| No. 2 Republican in U.S. House sees co | Curiosity + Question | 0 | 0 | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? | 0.001 | 0.51 |
| No. 2 Republican in U.S. House sees co | Curiosity + Question | 4 | 0 | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? A Conference is Imminent. | 0.001 | 0.60 |
| On the trail with Mike Pence: Putting  | Curiosity + Question | 0 | 0 | What's driving the vice president's high-stakes damage control tour? | 0.007 | 0.52 |
| On the trail with Mike Pence: Putting  | Curiosity + Question | 4 | 0 | What's driving Mike Pence's high-stakes tour of the country? A closer look at the vice president's efforts to contain the fallout from his boss's latest controversies. | 0.004 | 0.69 |
| U.S. lawmakers ask Trump to turn over  | Curiosity + Question | 0 | 0 | What's Behind the President's Silence on the Comey Conversations? | 0.007 | 0.50 |
| U.S. lawmakers ask Trump to turn over  | Curiosity + Question | 4 | 0 | What's Behind the President's Silence on the Comey Conversations? | 0.007 | 0.50 |
| Obama, Saudi king discuss U.S.-Saudi t | Curiosity + Question | 0 | 0 | What's behind the latest meeting between Obama and the Saudi king? | 0.006 | 0.75 |
| Obama, Saudi king discuss U.S.-Saudi t | Curiosity + Question | 4 | 0 | What's behind the latest meeting between Obama and the Saudi king? | 0.006 | 0.75 |

## A.2 Clickbait brake (beta) suppresses the independent detector

| Source | Tactic | alpha | beta | Generated | cb (ext.) |
|---|---|---|---|---|---|
| 'Numerous' arrests made during Trump i | Curiosity + Question | 4 | 0 | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 0.164 |
| 'Numerous' arrests made during Trump i | Curiosity + Question | 4 | 2 | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 0.164 |
| No. 2 Republican in U.S. House sees co | Curiosity + Question | 4 | 0 | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? A Conference is Imminent. | 0.001 |
| No. 2 Republican in U.S. House sees co | Curiosity + Question | 4 | 2 | When Will House Republican Leaders Unify Behind Tax Reform Plan Next Week's Conference Holds the Key? | 0.000 |
| On the trail with Mike Pence: Putting  | Curiosity + Question | 4 | 0 | What's driving Mike Pence's high-stakes tour of the country? A closer look at the vice president's efforts to contain the fallout from his boss's latest controversies. | 0.004 |
| On the trail with Mike Pence: Putting  | Curiosity + Question | 4 | 2 | As Mike Pence Hits the Campaign Trail, Is the Trump Blaze Becoming a Wildfire to Contain? | 0.004 |
| U.S. lawmakers ask Trump to turn over  | Curiosity + Question | 4 | 0 | What's Behind the President's Silence on the Comey Conversations? | 0.007 |
| U.S. lawmakers ask Trump to turn over  | Curiosity + Question | 4 | 2 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.013 |
| Obama, Saudi king discuss U.S.-Saudi t | Curiosity + Question | 4 | 0 | What's behind the latest meeting between Obama and the Saudi king? | 0.006 |
| Obama, Saudi king discuss U.S.-Saudi t | Curiosity + Question | 4 | 2 | As Obama meets with the Saudi king behind closed doors: Will their talks yield a new path forward for a complex relationship? | 0.003 |

## A.3 Marginal effect of each weight (mean over all sources/tactics)

*These means are computed over the 40-item tuning sweep used to select the operating point; the statistically-powered 300-item ablation with confidence intervals and paired significance is Table 6 in the main text (numbers differ slightly due to the smaller sample).*

| alpha | beta | tactic (judge) | cb (ext.) | STS | perplexity |
|---|---|---|---|---|---|
| 0 | 0 | 0.404 | 0.041 | 0.639 | 213 |
| 4 | 0 | 0.487 | 0.078 | 0.636 | 172 |
| 0 | 2 | 0.367 | 0.014 | 0.653 | 256 |
| 4 | 2 | 0.479 | 0.061 | 0.628 | 227 |

*Generated with Meta-Llama-3-8B-Instruct; clickbait/tactic scored by independent models (see Section 4).*