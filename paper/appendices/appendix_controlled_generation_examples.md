# Appendix A: Controlled Headline Generation — Examples and Analysis

Detailed four-cell examples showing the interaction between tactic activation (alpha) and clickbait avoidance (beta), with the independent metrics for each cell.

**Authoritative run (Cycle 2).** Meta-Llama-3-8B-Instruct, log-domain FUDGE objective, top-k = 50, cloud A10G/L40S via Modal. Guidance weights swept over alpha, beta in {0, 1, 2, 4}. Clickbait probability (cb) is from an independent DistilBERT detector trained only on human-authored Chakraborty data; STS is Sentence-Transformer cosine to the source. These appendices supersede the earlier exploratory top-k{50,500}, {0,1,2} sweep.

---

## A.1 'Numerous' arrests made during Trump inauguration protests: 

**Source (neutral):** "'Numerous' arrests made during Trump inauguration protests: police"  
**Target tactic:** Curiosity + Question

| cell | Generated headline | tactic (judge) | cb (ext.) | STS |
|---|---|---|---|---|
| (0,0) baseline | Behind the Scenes: What Led to the Wave of Arrests During Trump's Inauguration Protests?". | 0.00 | 0.035 | 0.76 |
| (4,0) engage only | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 1.00 | 0.164 | 0.59 |
| (0,2) brake only | Behind the Scenes: What Happened During Trump Inauguration Protests to Lead to So Many Arrests?". | 0.00 | 0.020 | 0.73 |
| (4,2) dual | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 1.00 | 0.164 | 0.59 |

## A.2 No. 2 Republican in U.S. House sees conference on tax bill s

**Source (neutral):** "No. 2 Republican in U.S. House sees conference on tax bill soon"  
**Target tactic:** Curiosity + Question

| cell | Generated headline | tactic (judge) | cb (ext.) | STS |
|---|---|---|---|---|
| (0,0) baseline | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? | 0.00 | 0.001 | 0.51 |
| (4,0) engage only | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? A Conference is Imminent. | 0.00 | 0.001 | 0.60 |
| (0,2) brake only | Who Will Be at the Table: A High-Stakes Tax Negotiation Looms. | 0.00 | 0.001 | 0.46 |
| (4,2) dual | When Will House Republican Leaders Unify Behind Tax Reform Plan Next Week's Conference Holds the Key? | 0.00 | 0.000 | 0.61 |

## A.3 On the trail with Mike Pence: Putting out fires lit by Trump

**Source (neutral):** "On the trail with Mike Pence: Putting out fires lit by Trump"  
**Target tactic:** Curiosity + Question

| cell | Generated headline | tactic (judge) | cb (ext.) | STS |
|---|---|---|---|---|
| (0,0) baseline | What's driving the vice president's high-stakes damage control tour? | 0.00 | 0.007 | 0.52 |
| (4,0) engage only | What's driving Mike Pence's high-stakes tour of the country? A closer look at the vice president's efforts to contain the fallout from his boss's latest controversies. | 0.00 | 0.004 | 0.69 |
| (0,2) brake only | As Mike Pence Hits the Campaign Trail, Whose Embers Will He Fan? | 0.00 | 0.002 | 0.63 |
| (4,2) dual | As Mike Pence Hits the Campaign Trail, Is the Trump Blaze Becoming a Wildfire to Contain? | 0.00 | 0.004 | 0.72 |

## A.4 U.S. lawmakers ask Trump to turn over any Comey tapes

**Source (neutral):** "U.S. lawmakers ask Trump to turn over any Comey tapes"  
**Target tactic:** Curiosity + Question

| cell | Generated headline | tactic (judge) | cb (ext.) | STS |
|---|---|---|---|---|
| (0,0) baseline | What's Behind the President's Silence on the Comey Conversations? | 0.00 | 0.007 | 0.50 |
| (4,0) engage only | What's Behind the President's Silence on the Comey Conversations? | 0.00 | 0.007 | 0.50 |
| (0,2) brake only | Did Trump Record Conversations with Comey? Lawmakers Demand Answers". | 0.00 | 0.001 | 0.69 |
| (4,2) dual | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.00 | 0.013 | 0.64 |

## A.5 Obama, Saudi king discuss U.S.-Saudi ties, conflicts: White 

**Source (neutral):** "Obama, Saudi king discuss U.S.-Saudi ties, conflicts: White House"  
**Target tactic:** Curiosity + Question

| cell | Generated headline | tactic (judge) | cb (ext.) | STS |
|---|---|---|---|---|
| (0,0) baseline | What's behind the latest meeting between Obama and the Saudi king? | 0.83 | 0.006 | 0.75 |
| (4,0) engage only | What's behind the latest meeting between Obama and the Saudi king? | 0.00 | 0.006 | 0.75 |
| (0,2) brake only | As Obama meets with Saudi king, a fragile balance of power hangs in the balance. | 0.00 | 0.001 | 0.67 |
| (4,2) dual | As Obama meets with the Saudi king behind closed doors: Will their talks yield a new path forward for a complex relationship? | 0.83 | 0.003 | 0.78 |

## A.6 House intel panel advances NSA spying bill despite privacy o

**Source (neutral):** "House intel panel advances NSA spying bill despite privacy objections"  
**Target tactic:** Curiosity + Question

| cell | Generated headline | tactic (judge) | cb (ext.) | STS |
|---|---|---|---|---|
| (0,0) baseline | What's driving the House intel panel to push forward with a surveillance bill despite growing concerns about individual privacy? | 0.00 | 0.018 | 0.74 |
| (4,0) engage only | What's Behind the House's Surprising Move to Push Forward with Controversial Surveillance Legislation Despite Widespread Concerns Over Privacy? | 0.00 | 0.088 | 0.57 |
| (0,2) brake only | As lawmakers move to pass new surveillance powers, who will be watching the watchers when privacy concerns go unheard? | 0.00 | 0.001 | 0.46 |
| (4,2) dual | As lawmakers move to greenlight the surveillance measure, will privacy advocates' cries for caution fall on deaf ears? | 0.00 | 0.001 | 0.54 |
