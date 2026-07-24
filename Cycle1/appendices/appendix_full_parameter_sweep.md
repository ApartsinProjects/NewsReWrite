# Appendix: Complete Parameter Sweep Results

This appendix contains the full guided-generation sweep for a set of source headlines, showing every (alpha, beta) combination per engagement tactic.

**Authoritative run (Cycle 2).** Meta-Llama-3-8B-Instruct, log-domain FUDGE objective, top-k = 50, cloud A10G/L40S via Modal. Guidance weights swept over alpha, beta in {0, 1, 2, 4}. Clickbait probability (cb) is from an independent DistilBERT detector trained only on human-authored Chakraborty data; STS is Sentence-Transformer cosine to the source. These appendices supersede the earlier exploratory top-k{50,500}, {0,1,2} sweep.

---

## Headline 1

**Source:** "'Numerous' arrests made during Trump inauguration protests: police"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Behind the Scenes: What Led to the Wave of Arrests During Trump's Inauguration Protests?". | 0.035 | 0.76 |
| 0 | 1 | Behind the peaceful protests, a different story unfolded: how many were taken away in handcuffs?". | 0.036 | 0.60 |
| 0 | 2 | Behind the Scenes: What Happened During Trump Inauguration Protests to Lead to So Many Arrests?". | 0.020 | 0.73 |
| 0 | 4 | Behind the peaceful protests, a different story: How many Trump inauguration demonstrators ended up behind bars?". | 0.006 | 0.71 |
| 1 | 0 | Behind the Scenes of a Turbulent Inauguration Day: How Many Protesters Were Taken into Custody?". | 0.002 | 0.72 |
| 1 | 1 | Behind the peaceful faces of Trump's inauguration protests, how many secrets were locked away?". | 0.022 | 0.55 |
| 1 | 2 | Behind the peaceful faces, how many secrets were locked up during Trump inauguration protests?". | 0.021 | 0.61 |
| 1 | 4 | Behind the peaceful faces, how many secrets were locked up tight on Inauguration Day?". | 0.004 | 0.39 |
| 2 | 0 | Behind the Scenes of the Inauguration Day Unrest: What Led to the 'Numerous' Arrests?". | 0.005 | 0.76 |
| 2 | 1 | Behind the peaceful faces of Trump's inauguration protests, how many secrets were locked away?". | 0.022 | 0.55 |
| 2 | 2 | Behind the Scenes of a Turbulent Inauguration: How Many Protesters Were Taken into Custody?". | 0.001 | 0.75 |
| 2 | 4 | Behind the peaceful faces of Trump inauguration protests, how many secrets were locked away?". | 0.007 | 0.57 |
| 4 | 0 | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 0.164 | 0.59 |
| 4 | 1 | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 0.164 | 0.59 |
| 4 | 2 | Was the force used to quell dissent or protect the crowd during the Trump inauguration protests, as 'numerous' arrests were made by police?". | 0.164 | 0.59 |
| 4 | 4 | Was the force of law used to quell dissent or simply maintain order during the Trump inauguration protests, as police report 'numerous' arrests were made?". | 0.201 | 0.58 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Chaos Erupts as Thousands Detained During Trump's Sworn-In Ceremony". | 0.001 | 0.61 |
| 0 | 1 | Chaos Erupts on Inauguration Day as Protesters Clash with Authorities". | 0.000 | 0.50 |
| 0 | 2 | Chaos Erupts on Inauguration Day as Protesters Clash with Authorities, Leaving Dozens in Custody". | 0.001 | 0.61 |
| 0 | 4 | Chaos Erupts on Inauguration Day, Hundreds Hauled Away by Authorities". | 0.000 | 0.54 |
| 1 | 0 | Chaos Erupts as Thousands Face the Brunt of Law Enforcement's Firm Hand During Trump Inauguration Protests". | 0.001 | 0.63 |
| 1 | 1 | Chaos Erupts on Inauguration Day as Protesters Clash with Authorities". | 0.000 | 0.50 |
| 1 | 2 | Chaos Erupts on Inauguration Day as Protesters Clash with Authorities". | 0.000 | 0.50 |
| 1 | 4 | Chaos Erupts on Inauguration Day, Hundreds Hauled Away by Authorities". | 0.000 | 0.54 |
| 2 | 0 | Chaos Erupts as Thousands Take to the Streets, Leaving a Trail of Detained Protesters in the Wake of Trump's Swearing-In". | 0.003 | 0.57 |
| 2 | 1 | Chaos Erupts as Peaceful Protesters Face Unexpected Confrontation with Authorities on Inauguration Day". | 0.001 | 0.55 |
| 2 | 2 | Chaos Erupts on Inauguration Day as Hundreds Taken into Custody". | 0.000 | 0.59 |
| 2 | 4 | Chaos Erupts on D.C. Streets as Protest Turned Ugly, Leaving Dozens in Handcuffs". | 0.001 | 0.54 |
| 4 | 0 | Chaos unfolds as authorities crack down on defiant crowds during Trump's historic inauguration celebration". | 0.005 | 0.60 |
| 4 | 1 | Chaos erupts as unprecedented wave of arrests sweeps through DC streets during Trump's inauguration". | 0.004 | 0.64 |
| 4 | 2 | Chaos erupts as peaceful protests turn violent, widespread arrests made on Inauguration Day". | 0.000 | 0.63 |
| 4 | 4 | Chaos erupts on D.C. streets as widespread peaceful expression is met with brute force". | 0.001 | 0.44 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Chaos Erupts on Inauguration Day: 'Numerous' Detentions Spark Questions About Free Speech". | 0.000 | 0.61 |
| 0 | 1 | Chaos Erupts on Inauguration Day: 'Numerous' Detentions Made as Protesters Clash with Authorities". | 0.000 | 0.70 |
| 0 | 2 | Chaos Erupts on Inauguration Day: 'Numerous' Detentions Spark Questions About Free Speech". | 0.000 | 0.61 |
| 0 | 4 | Chaos Erupts on Inauguration Day: 'Numerous' Detentions Made as Protesters Clash with Authorities". | 0.000 | 0.70 |
| 1 | 0 | Chaos Erupts as Thousands Take to the Streets: How Many Will Face the Consequences?". | 0.001 | 0.38 |
| 1 | 1 | Chaos Erupts on Inauguration Day: How Many Protesters Will Face the Music?". | 0.001 | 0.51 |
| 1 | 2 | Chaos Erupts on Inauguration Day: 'Numerous' Detentions Spark Questions About Free Speech". | 0.000 | 0.61 |
| 1 | 4 | Chaos Erupts on Inauguration Day: 'Numerous' Detentions Spark Questions of Free Speech". | 0.000 | 0.61 |
| 2 | 0 | Chaos Erupts as Thousands Take to the Streets: 'Numerous' Detentions Made Amid Trump Inauguration Protests". | 0.001 | 0.70 |
| 2 | 1 | Chaos Erupts: How Many Protesters Got Caught Up in the Chaos of Trump's Big Day?". | 0.024 | 0.58 |
| 2 | 2 | Chaos Erupts on Inauguration Day as 'Hundreds' Taken into Custody". | 0.000 | 0.64 |
| 2 | 4 | Chaos Erupts on Inauguration Day: 'Numerous' Detentions Spark Questions of Free Speech". | 0.000 | 0.61 |
| 4 | 0 | Chaos Erupted: Was the Force of Resistance Too Much for Law Enforcement to Handle During Trump's Inaugural Protest Showdowns, Leaving 'Numerous' Detainees in Its Wake?". | 0.176 | 0.67 |
| 4 | 1 | Chaos Erupts Amid 'Peaceful' Protests: Dozens of Demonstrators Taken into Custody During Trump's Sworn-In Ceremony?". | 0.000 | 0.69 |
| 4 | 2 | Chaos Erupts Amid 'Peaceful' Protests: Dozens of Demonstrators Taken into Custody Amid Scenes of Tension and Turmoil". | 0.000 | 0.61 |
| 4 | 4 | Chaos Erupts on D.C. Streets as 'Hundreds' Detained Amid Trump Inauguration Unrest". | 0.000 | 0.64 |

---

## Headline 2

**Source:** "No. 2 Republican in U.S. House sees conference on tax bill soon"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? | 0.001 | 0.51 |
| 0 | 1 | When Will House Republicans Unite Behind a Tax Plan? | 0.001 | 0.57 |
| 0 | 2 | Who Will Be at the Table: A High-Stakes Tax Negotiation Looms. | 0.001 | 0.46 |
| 0 | 4 | When Tax Reform Talks Heat Up, Who Will Be the Next to Weigh In? | 0.000 | 0.52 |
| 1 | 0 | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? | 0.001 | 0.51 |
| 1 | 1 | When Will House Republicans Unite Behind a Tax Plan? | 0.001 | 0.57 |
| 1 | 2 | When Will House Republican Leaders Unify Behind Tax Reform? | 0.000 | 0.57 |
| 1 | 4 | When Tax Reform Takes Center Stage: Will House Republicans Rally Behind New Proposal Soon? | 0.000 | 0.58 |
| 2 | 0 | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? | 0.001 | 0.51 |
| 2 | 1 | When Will House Republicans Unite Behind a Tax Plan? | 0.001 | 0.57 |
| 2 | 2 | Who Will Be at the Table: A High-Stakes Tax Negotiation Looms for the No. 2 Republican in the U.S. House. | 0.003 | 0.67 |
| 2 | 4 | When Tax Reform Takes Center Stage in Congress: Will Party Unity Hold Strong? | 0.000 | 0.53 |
| 4 | 0 | Will the Republican Party's Second-in-Command Deliver a Breakthrough on Tax Reform? A Conference is Imminent. | 0.001 | 0.60 |
| 4 | 1 | When Will House Republicans Unite Behind a Tax Plan? The No. 2 Republican Says a Key Meeting is Just Around the Corner. | 0.002 | 0.68 |
| 4 | 2 | When Will House Republican Leaders Unify Behind Tax Reform Plan Next Week's Conference Holds the Key? | 0.000 | 0.61 |
| 4 | 4 | When and Where Will House Republican Leaders Huddle on Tax Reform Next? | 0.001 | 0.59 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Tax Reform on the Brink: Top Republican Leader Hints at Urgent Meeting". | 0.000 | 0.59 |
| 0 | 1 | Tax Reform on the Brink: Top Republican Leader Hints at Urgent Meeting". | 0.000 | 0.59 |
| 0 | 2 | Tax Reform on the Brink: Top Republican Leader Hints at Urgent Meeting". | 0.000 | 0.59 |
| 0 | 4 | Tax Reform on the Brink of Breakthrough: Top Republican Leader Hints at Urgency". | 0.000 | 0.52 |
| 1 | 0 | Tax Reform on the Brink: Top Republican Leader Hints at Imminent Congressional Showdown". | 0.001 | 0.55 |
| 1 | 1 | Tax Reform on the Brink: Top Republican Leader Hints at Urgent Congressional Meeting". | 0.000 | 0.59 |
| 1 | 2 | Tax Reform on the Brink: GOP's Second-in-Command Hints at Imminent Conference". | 0.001 | 0.60 |
| 1 | 4 | Tax Reform on the Brink of Breakthrough: Top Republican Leader Hints at Imminent Show-Down". | 0.001 | 0.51 |
| 2 | 0 | Tax Reform on the Brink: Top Republican Leader Hints at Imminent Conference". | 0.000 | 0.60 |
| 2 | 1 | Tax Reform on the Brink: Top Republican Leader Hints at Urgent Congressional Meeting". | 0.000 | 0.59 |
| 2 | 2 | Tax Reform on the Brink: Top Republican Leader Hints at Imminent Conference". | 0.000 | 0.60 |
| 2 | 4 | Tax Reform on the Brink: GOP's Second-in-Command Hints at Imminent Congressional Showdown". | 0.001 | 0.56 |
| 4 | 0 | Tax Reform on the Horizon: Top GOP Leader Signals Major Move Imminent in the House". | 0.000 | 0.54 |
| 4 | 1 | Tax Reform on the Horizon: Senior GOP Lawmaker Eyes Critical Conference This Week". | 0.002 | 0.64 |
| 4 | 2 | Tax Reform on the Horizon: Senior GOP Lawmaker Eyes Critical Conference This Week". | 0.002 | 0.64 |
| 4 | 4 | Tax Reform on the Horizon: Senior GOP Lawmaker Eyes Critical Conference This Week". | 0.002 | 0.64 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 0 | 1 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 0 | 2 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 0 | 4 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver?". | 0.001 | 0.54 |
| 1 | 0 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 1 | 1 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 1 | 2 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 1 | 4 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver on Promises?". | 0.001 | 0.51 |
| 2 | 0 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 2 | 1 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 2 | 2 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver a Deal?". | 0.001 | 0.57 |
| 2 | 4 | Tax Bill Showdown Looms: Will the GOP's Second-in-Command Deliver on Promises?". | 0.001 | 0.51 |
| 4 | 0 | Is a Tax Bill Breakthrough Looming in the Republican Ranks?". | 0.001 | 0.51 |
| 4 | 1 | Will the GOP's Tax Plan Get a Boost from a Surprise House Conference?". | 0.006 | 0.53 |
| 4 | 2 | Will the GOP's Tax Mastermind Deliver Breakthrough or Breakdown in Congress Soon?". | 0.002 | 0.46 |
| 4 | 4 | Will the Republican Party Fast-Track Tax Reforms, or Will Internal Disagreements Derail Progress?". | 0.001 | 0.52 |

---

## Headline 3

**Source:** "On the trail with Mike Pence: Putting out fires lit by Trump"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | What's driving the vice president's high-stakes damage control tour? | 0.007 | 0.52 |
| 0 | 1 | As Mike Pence Hits the Campaign Trail, Whose Embers Will He Fan? | 0.002 | 0.63 |
| 0 | 2 | As Mike Pence Hits the Campaign Trail, Whose Embers Will He Fan? | 0.002 | 0.63 |
| 0 | 4 | As Mike Pence Hits the Campaign Trail, Whispers of a Blazing Controversary Lurk. | 0.001 | 0.57 |
| 1 | 0 | What's driving the vice president's high-stakes tour of the country? | 0.006 | 0.51 |
| 1 | 1 | As Mike Pence Hits the Campaign Trail, What's the Real Agenda Behind the Blazes of Controversy? | 0.014 | 0.60 |
| 1 | 2 | As Mike Pence Hits the Campaign Trail, Whose Embers Will He Fan Next? | 0.003 | 0.65 |
| 1 | 4 | As Mike Pence Hits the Campaign Trail, Whispers of a Blazing Controversary Lurk. | 0.001 | 0.57 |
| 2 | 0 | What's driving the frantic pace of Mike Pence's campaign trail? | 0.011 | 0.63 |
| 2 | 1 | As Mike Pence Hits the Campaign Trail, What Flames Will He Try to Douse Next? | 0.002 | 0.69 |
| 2 | 2 | As Mike Pence Hits the Campaign Trail, Did Trump's Embers Leave Lasting Scorch Marks? | 0.002 | 0.63 |
| 2 | 4 | As Mike Pence Hits the Campaign Trail, Did Trump-Era Flames Die Out or Simmer Under the Surface? | 0.001 | 0.69 |
| 4 | 0 | What's driving Mike Pence's high-stakes tour of the country? A closer look at the vice president's efforts to contain the fallout from his boss's latest controversies. | 0.004 | 0.69 |
| 4 | 1 | As Mike Pence's path forward ignites scrutiny: Can he contain the flames of controversy sparked by his predecessor? | 0.006 | 0.66 |
| 4 | 2 | As Mike Pence Hits the Campaign Trail, Is the Trump Blaze Becoming a Wildfire to Contain? | 0.004 | 0.72 |
| 4 | 4 | As Mike Pence Hits the Campaign Trail, Did Trump Set Fires Ahead of His Arrival? | 0.001 | 0.72 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Blazing a Path of Damage Control: The Vice President's High-Stakes Mission". | 0.002 | 0.55 |
| 0 | 1 | Blazing a Path of Damage Control: The Vice President's High-Stakes Road Trip". | 0.003 | 0.60 |
| 0 | 2 | Blazing a Path of Damage Control: The Vice President's High-Stakes Roadshow". | 0.003 | 0.57 |
| 0 | 4 | Blazing a Path of Damage Control: The Vice President's High-Stress Road Trip". | 0.007 | 0.56 |
| 1 | 0 | Blazing a Path of Damage Control: The Vice President's High-Stakes Journey". | 0.006 | 0.56 |
| 1 | 1 | Blazing a Path of Damage Control: The Vice President's High-Stakes Tour". | 0.002 | 0.58 |
| 1 | 2 | Blazing a Path of Damage Control: The Vice President's High-Stakes Roadshow". | 0.003 | 0.57 |
| 1 | 4 | Blazing a Path of Damage Control: The Vice President's High-Stress Road Trip". | 0.007 | 0.56 |
| 2 | 0 | Blazing a Path of Damage Control: The Vice President's High-Stakes Journey to Douse the Flames of Controversy". | 0.011 | 0.56 |
| 2 | 1 | Blazing a Path of Damage Control: The Vice President's High-Stakes Tour". | 0.002 | 0.58 |
| 2 | 2 | Blazing a Path of Damage Control: The Vice President's High-Stakes Tour". | 0.002 | 0.58 |
| 2 | 4 | Blazing a Path of Damage Control: The Vice President's High-Stress Tour". | 0.011 | 0.52 |
| 4 | 0 | Blazing a Critical Path: Pence's High-Stakes Mission to Douse the Flames of Controversy". | 0.007 | 0.70 |
| 4 | 1 | Blazing a Critical Path: Pence Tackles the Flames of Controversy". | 0.005 | 0.65 |
| 4 | 2 | Blazing a Critical Path with Pence: Dousing Flames of Controversy Left by the President". | 0.002 | 0.69 |
| 4 | 4 | Blazing a Path of Recovery: The Real Work Begins with Pence at the Helm". | 0.004 | 0.64 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Blazing a Trail of Controversy: Can Pence Contain the Flames of Trump's Legacy?". | 0.004 | 0.71 |
| 0 | 1 | Blazing a Trail of Controversy: Can Pence Contain the Flames of Trump's Legacy?". | 0.004 | 0.71 |
| 0 | 2 | Blazing a Trail of Controversy: Can Pence Contain the Flames of Trump's Legacy?". | 0.004 | 0.71 |
| 0 | 4 | Blazing a Trail of Controversy: Pence's High-Stakes Damage Control". | 0.001 | 0.66 |
| 1 | 0 | Blazing a Trail of Controversy: Can Mike Pence Keep the Flames of Political Discord Under Control?". | 0.005 | 0.70 |
| 1 | 1 | Blazing a Trail of Controversy: Can Pence Douse the Flames of Presidential Fallout?". | 0.001 | 0.69 |
| 1 | 2 | Blazing a Trail of Controversy: Can Pence Douse the Flames of Presidential Fallout?". | 0.001 | 0.69 |
| 1 | 4 | Blazing a Trail of Controversy: Can Pence Douse the Flames of Trump's Legacy?". | 0.004 | 0.71 |
| 2 | 0 | Blazing a Trail of Controversy: Can Vice President Pence Really Douse the Flames of Presidential Discord?". | 0.002 | 0.63 |
| 2 | 1 | Blazing a Trail of Controversy: Can Vice President Pence Keep Up with the Flames Ignited by His Boss?". | 0.002 | 0.74 |
| 2 | 2 | Blazing a Trail of Controversy: Can Pence Really Put Out the Flames Ignited by Trump?". | 0.004 | 0.76 |
| 2 | 4 | Blazing a Trail of Controversy: Can Pence Really Put Out the Flames Ignited by Trump?". | 0.004 | 0.76 |
| 4 | 0 | Is the Vice President's Damage Control Efforts Enough to Douse the Flames of Controversy Ignited by the President?". | 0.015 | 0.49 |
| 4 | 1 | Is the Vice President's Damage Control Efforts Enough to Douse the Flames of Controversy Ignited by the President?". | 0.015 | 0.49 |
| 4 | 2 | Is the Vice President's Damage Control Efforts Enough to Douse the Flames of Controversy Ignited by the President?". | 0.015 | 0.49 |
| 4 | 4 | Blazing a Trail of Damage Control: Is the Vice President Really Tame to Trump's Flames or Just Fanning the Flames Himself?". | 0.025 | 0.64 |

---

## Headline 4

**Source:** "U.S. lawmakers ask Trump to turn over any Comey tapes"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | What's Behind the President's Silence on the Comey Conversations? | 0.007 | 0.50 |
| 0 | 1 | Did Trump Record Conversations with Comey? Lawmakers Demand Answers". | 0.001 | 0.69 |
| 0 | 2 | Did Trump Record Conversations with Comey? Lawmakers Demand Answers". | 0.001 | 0.69 |
| 0 | 4 | Did Trump Record Conversations with Comey? Lawmakers Demand Answers". | 0.001 | 0.69 |
| 1 | 0 | What's Behind the President's Silence on the Comey Conversations? | 0.007 | 0.50 |
| 1 | 1 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress?". | 0.009 | 0.65 |
| 1 | 2 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress?". | 0.009 | 0.65 |
| 1 | 4 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.013 | 0.64 |
| 2 | 0 | What's Behind the President's Silence on the Comey Conversations? | 0.007 | 0.50 |
| 2 | 1 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress?". | 0.009 | 0.65 |
| 2 | 2 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress?". | 0.009 | 0.65 |
| 2 | 4 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.013 | 0.64 |
| 4 | 0 | What's Behind the President's Silence on the Comey Conversations? | 0.007 | 0.50 |
| 4 | 1 | Did Trump's Oval Office conversations with Comey leave a lasting record?". | 0.006 | 0.48 |
| 4 | 2 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.013 | 0.64 |
| 4 | 4 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.013 | 0.64 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Trump's Secrets Under Siege: Lawmakers Demand Truth About Comey's Fate". | 0.003 | 0.58 |
| 0 | 1 | Trump's Secrets Under Siege: Lawmakers Demand Truth About Comey's Fate". | 0.003 | 0.58 |
| 0 | 2 | Trump's Secrets Under Siege: Lawmakers Demand Truth About Comey's Fate". | 0.003 | 0.58 |
| 0 | 4 | Trump's Secrets Under Siege: Lawmakers Demand Truth About Comey's Fate". | 0.003 | 0.58 |
| 1 | 0 | Trump's Silence Sparks Fear of a Hidden Truth: Lawmakers Demand Answers on Secret Recordings". | 0.003 | 0.53 |
| 1 | 1 | Trump's Secret Recordings of Comey Meeting Raise Questions of Transparency and Accountability". | 0.002 | 0.54 |
| 1 | 2 | Trump's Secret Recordings of Comey Meeting Leave Lawmakers Demanding Transparency". | 0.001 | 0.62 |
| 1 | 4 | Trump's Secret Recordings Put Under Microscope as Lawmakers Demand Truth". | 0.001 | 0.57 |
| 2 | 0 | Trump's Silence Raises Concerns as Lawmakers Demand Truth Behind Secret Conversations". | 0.002 | 0.50 |
| 2 | 1 | Trump's Silence on Comey Secrets Sparks Fear of Hidden Truths". | 0.005 | 0.44 |
| 2 | 2 | Trump's Silence on Comey Secrets is Deafening: Lawmakers Demand Truth". | 0.011 | 0.54 |
| 2 | 4 | Trump's Secret Recordings Cast Shadow Over White House". | 0.001 | 0.42 |
| 4 | 0 | Trump's Silence Raises Concerns as Lawmakers Demand Truth Behind Secret Conversations". | 0.002 | 0.50 |
| 4 | 1 | Trump's Silence Raises Concerns: Lawmakers Demand Truth Behind Secret Recordings". | 0.001 | 0.57 |
| 4 | 2 | Trump's Silence is the Real Smoking Gun: Will He Finally Come Clean on Comey's Fate?". | 0.091 | 0.49 |
| 4 | 4 | Trump's Silence on Comey Secrets is Cause for Anxiety on Capitol Hill". | 0.007 | 0.46 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Did Trump's Oval Office Secrets Just Get Exposed?". | 0.006 | 0.39 |
| 0 | 1 | Did Trump Record His Conversations with Comey? Lawmakers Demand to Know". | 0.001 | 0.70 |
| 0 | 2 | Did Trump Record His Conversations with Comey? Lawmakers Demand to Know". | 0.001 | 0.70 |
| 0 | 4 | Did Trump Record His Conversations with Comey? Lawmakers Demand to Know". | 0.001 | 0.70 |
| 1 | 0 | Did Trump's Silence on Comey's Fate Hint at a Hidden Agenda?". | 0.005 | 0.44 |
| 1 | 1 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress?". | 0.009 | 0.65 |
| 1 | 2 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress?". | 0.009 | 0.65 |
| 1 | 4 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.013 | 0.64 |
| 2 | 0 | Did Trump's Silence on Comey's Fate Hint at a Hidden Agenda?". | 0.005 | 0.44 |
| 2 | 1 | Did Trump's Oval Office Secrets Just Get Exposed?". | 0.006 | 0.39 |
| 2 | 2 | Did Trump Silence Comey with Secret Recordings?". | 0.001 | 0.57 |
| 2 | 4 | Did Trump Record His Conversations with Comey - and Will He Share Them with Congress Now?". | 0.013 | 0.64 |
| 4 | 0 | Did the President's Silence Speak Louder Than Words on the FBI Chief's Fate?". | 0.021 | 0.38 |
| 4 | 1 | Did Trump's Oval Office Secrets Just Get Exposed?". | 0.006 | 0.39 |
| 4 | 2 | Did Trump Silence the Comey Truth: Lawmakers Demand Access to Secret Recordings". | 0.018 | 0.65 |
| 4 | 4 | Did Trump Silence Comey's Secrets Forever: Lawmakers Demand Access to Muted Truth". | 0.122 | 0.58 |

---

## Headline 5

**Source:** "Obama, Saudi king discuss U.S.-Saudi ties, conflicts: White House"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | What's behind the latest meeting between Obama and the Saudi king? | 0.006 | 0.75 |
| 0 | 1 | As Obama meets with Saudi king, what's on the agenda for a delicate dance of diplomacy? | 0.002 | 0.71 |
| 0 | 2 | As Obama meets with Saudi king, a fragile balance of power hangs in the balance. | 0.001 | 0.67 |
| 0 | 4 | Obama and Saudi King Explore the Future of a Complex Partnership. | 0.000 | 0.76 |
| 1 | 0 | What's behind the latest meeting between Obama and the Saudi king? | 0.006 | 0.75 |
| 1 | 1 | As Obama meets with Saudi king, what secrets will be shared behind closed doors to strengthen U.S.-Saudi bonds? | 0.001 | 0.69 |
| 1 | 2 | As Obama meets with Saudi king, are U.S.-Saudi ties strong enough to weather Middle East conflicts? | 0.001 | 0.77 |
| 1 | 4 | Obama and Saudi King Explore the Future of a Complex Partnership. | 0.000 | 0.76 |
| 2 | 0 | What's behind the latest meeting between Obama and the Saudi king? | 0.006 | 0.75 |
| 2 | 1 | As Obama meets with Saudi king, are U.S.-Saudi ties stronger than conflicts in the balance? | 0.001 | 0.79 |
| 2 | 2 | As Obama meets with Saudi king, are U.S.-Saudi ties stronger than conflicts abroad? | 0.001 | 0.78 |
| 2 | 4 | Obama and Saudi King Explore the Future of a Complex Partnership. | 0.000 | 0.76 |
| 4 | 0 | What's behind the latest meeting between Obama and the Saudi king? | 0.006 | 0.75 |
| 4 | 1 | As Obama meets with the Saudi king behind closed doors: Can U.S. and Saudi Arabia find common ground on thorny conflicts and bilateral relations? | 0.001 | 0.81 |
| 4 | 2 | As Obama meets with the Saudi king behind closed doors: Will their talks yield a new path forward for a complex relationship? | 0.003 | 0.78 |
| 4 | 4 | Obama and Saudi King's Conversation Leaves Question: Can Friction in Relations with Riyadh be Smoothed Out Amid Overt Conflicts? | 0.000 | 0.79 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Tensions Simmer as Obama and Saudi King Confront Thorny Issues in High-Stakes Meeting. | 0.000 | 0.78 |
| 0 | 1 | Tensions Ease as Obama and Saudi King Unite to Strengthen Bonds Amid Global Unrest". | 0.000 | 0.70 |
| 0 | 2 | Tensions Ease as Obama and Saudi King Unite to Strengthen Bonds Amid Global Unrest". | 0.000 | 0.70 |
| 0 | 4 | Tensions Ease as Obama and Saudi King Unite to Strengthen Bonds Amid Global Unrest". | 0.000 | 0.70 |
| 1 | 0 | Tensions Simmer as Obama and Saudi King Confront the Future of a Fractious Alliance. | 0.000 | 0.77 |
| 1 | 1 | Tensions and Ties on the Table: Obama and Saudi King Hold High-Stakes Talks". | 0.000 | 0.81 |
| 1 | 2 | Tensions and Ties on the Table: Obama and Saudi King Hold High-Stakes Talks". | 0.000 | 0.81 |
| 1 | 4 | Tensions and Ties on the Table: Obama and Saudi King Hold High-Stakes Talks". | 0.000 | 0.81 |
| 2 | 0 | Tensions Simmer as Obama and Saudi King Confront the Future of a Fractious Alliance. | 0.000 | 0.77 |
| 2 | 1 | Tensions and Ties: Obama and Saudi King Hold High-Stakes Talks". | 0.000 | 0.83 |
| 2 | 2 | Tensions and Ties: Obama and Saudi King Hold High-Stakes Talks". | 0.000 | 0.83 |
| 2 | 4 | Tensions and Ties on the Table: Obama and Saudi King Hold High-Stakes Talks". | 0.000 | 0.81 |
| 4 | 0 | Tensions and Trust: The Weight of a Lasting Alliance is Put to the Test as Obama and the Saudi King Confront the Complexities of U.S.-Saudi Relations. | 0.001 | 0.67 |
| 4 | 1 | Tensions Soar as Obama and Saudi King Confront Dire Consequences of Their Relationship's Dark Underbelly: A Candid Discussion Unfolds at the White House". | 0.250 | 0.81 |
| 4 | 2 | Tensions Soar as Obama and Saudi King Confront Dire Confrontational Crossroads". | 0.001 | 0.76 |
| 4 | 4 | Tensions Soar as Obama and Saudi King Confront Dire Confrontational Crosscurrent in Vital U.S.-Saudi Alliance". | 0.000 | 0.79 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Behind Closed Doors: Obama and Saudi King Confront the Fractured Future of a Tainted Alliance". | 0.002 | 0.72 |
| 0 | 1 | Behind Closed Doors: Obama and Saudi King Confront Tensions in U.S.-Saudi Relationship". | 0.001 | 0.81 |
| 0 | 2 | Behind Closed Doors: Obama and Saudi King Confront Tensions in U.S.-Saudi Relationship". | 0.001 | 0.81 |
| 0 | 4 | Behind Closed Doors: Obama and Saudi King Confront Tensions in U.S.-Saudi Relationship". | 0.001 | 0.81 |
| 1 | 0 | Behind Closed Doors: What's Really at Stake in the U.S.-Saudi Relationship?". | 0.005 | 0.68 |
| 1 | 1 | Behind Closed Doors: Obama and Saudi King Confront the Fractured State of U.S.-Saudi Relations". | 0.001 | 0.77 |
| 1 | 2 | Behind Closed Doors: Obama and Saudi King Confront Tensions in U.S.-Saudi Relationship". | 0.001 | 0.81 |
| 1 | 4 | Behind Closed Doors: Obama and Saudi King Confront Tensions in U.S.-Saudi Relationship". | 0.001 | 0.81 |
| 2 | 0 | Behind Closed Doors: What Really Unites and Divides the U.S. and Saudi Arabia?". | 0.009 | 0.61 |
| 2 | 1 | Behind Closed Doors: Is the Future of U.S.-Saudi Relations at Risk Amidst Turbulent Conflicts?". | 0.001 | 0.68 |
| 2 | 2 | Behind Closed Doors: Is the Future of US-Saudi Relations at Risk of Collision Course with Global Conflicts?". | 0.001 | 0.68 |
| 2 | 4 | Behind Closed Doors: Obama and Saudi King Confront Tensions in U.S.-Saudi Relationship". | 0.001 | 0.81 |
| 4 | 0 | Is the Special Bond Between Obama and the Saudi King Enough to Bridge the Growing Divide Between the U.S. and Saudi Arabia?". | 0.004 | 0.69 |
| 4 | 1 | Is the Special Bond Between Obama and the Saudi King Enough to Bridge the Growing Divide Between the U.S. and Saudi Arabia?". | 0.004 | 0.69 |
| 4 | 2 | Behind Closed Doors: Is the Future of a Fractious Friendship at Risk or Will Diplomacy Save the Day?". | 0.003 | 0.37 |
| 4 | 4 | Behind Closed Doors: Is the Future of a Fractious Relationship Between Old Allies Up for Grapple or Gracious Give-and-Take with the King of Saudi Arabia?". | 0.930 | 0.61 |

---

## Headline 6

**Source:** "House intel panel advances NSA spying bill despite privacy objections"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | What's driving the House intel panel to push forward with a surveillance bill despite growing concerns about individual privacy? | 0.018 | 0.74 |
| 0 | 1 | As lawmakers move to pass new surveillance powers, who will be watching the watchers? | 0.000 | 0.45 |
| 0 | 2 | As lawmakers move to pass new surveillance powers, who will be watching the watchers when privacy concerns go unheard? | 0.001 | 0.46 |
| 0 | 4 | Despite Privacy Concerns, House Panel Pushes Forward with Controversial Surveillance Measure. | 0.000 | 0.69 |
| 1 | 0 | What's driving the House intel panel to push forward with a surveillance bill despite growing concerns about individual privacy? | 0.018 | 0.74 |
| 1 | 1 | As lawmakers move to pass the surveillance measure, what concessions will be made to address growing concerns about individual privacy? | 0.001 | 0.51 |
| 1 | 2 | As lawmakers move to pass new surveillance powers, will privacy protections be sacrificed for national security gains? | 0.000 | 0.49 |
| 1 | 4 | Despite Privacy Concerns, Will Controversial Surveillance Measure Clear Congressional Hurdles? | 0.000 | 0.59 |
| 2 | 0 | What's driving the House intel panel to push forward with a surveillance bill despite growing concerns about individual privacy? | 0.018 | 0.74 |
| 2 | 1 | As lawmakers move to pass the surveillance measure, are privacy concerns being silenced in the shadows of national security? | 0.000 | 0.51 |
| 2 | 2 | As lawmakers move to pass new surveillance powers, will privacy be sacrificed for national security? | 0.000 | 0.48 |
| 2 | 4 | Despite Privacy Concerns, Will Controversial Surveillance Measure Clear Next Hurdle in Congress? | 0.000 | 0.58 |
| 4 | 0 | What's Behind the House's Surprising Move to Push Forward with Controversial Surveillance Legislation Despite Widespread Concerns Over Privacy? | 0.088 | 0.57 |
| 4 | 1 | As lawmakers move forward with a controversial surveillance measure, will privacy concerns be silenced by the pursuit of national security? | 0.001 | 0.52 |
| 4 | 2 | As lawmakers move to greenlight the surveillance measure, will privacy advocates' cries for caution fall on deaf ears? | 0.001 | 0.54 |
| 4 | 4 | Despite Privacy Concerns, Can Surveillance Powers Overreach Be Checked in Secret Intelligence Legislation's Latest Step Closer to Passage? | 0.001 | 0.63 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 0 | 1 | Surveillance Bill Clears Hurdle Despite Fears of Invasion of Personal Space". | 0.000 | 0.52 |
| 0 | 2 | Surveillance Bill Clears Hurdle Despite Fears of Invasion of Personal Space". | 0.000 | 0.52 |
| 0 | 4 | Surveillance Bill Clears Hurdle Despite Fears of Invasion of Personal Space". | 0.000 | 0.52 |
| 1 | 0 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 1 | 1 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 1 | 2 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 1 | 4 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 2 | 0 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 2 | 1 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 2 | 2 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 2 | 4 | Surveillance Bill Clears Hurdle Despite Fears of Unchecked Government Snooping". | 0.000 | 0.52 |
| 4 | 0 | Surveillance Bill Clears Major Hurdle Despite Concerns Over Americans' Private Lives". | 0.000 | 0.50 |
| 4 | 1 | Surveillance Bill Clears Major Hurdle Despite Concerns Over Americans' Private Lives". | 0.000 | 0.50 |
| 4 | 2 | Surveillance Bill Clears Hurdle Despite Concerns Over Americans' Private Lives". | 0.000 | 0.52 |
| 4 | 4 | Surveillance Bill Clears Hurdle Despite Concerns Over Americans' Private Lives". | 0.000 | 0.52 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 0 | 1 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 0 | 2 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 0 | 4 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 1 | 0 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 1 | 1 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 1 | 2 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 1 | 4 | Spying Bill Clears Hurdle Despite Growing Fears of Privacy Invasion". | 0.000 | 0.62 |
| 2 | 0 | Spying Bill Clears Hurdle Despite Growing Concerns Over Privacy Invasion". | 0.000 | 0.63 |
| 2 | 1 | Spying Bill Clears Hurdle Despite Growing Concerns Over Privacy Invasion". | 0.000 | 0.63 |
| 2 | 2 | Spying Bill Clears Hurdle Despite Growing Concerns Over Privacy Invasion". | 0.000 | 0.63 |
| 2 | 4 | Spying Bill Clears Hurdle Despite Growing Concerns Over Privacy Invasion". | 0.000 | 0.63 |
| 4 | 0 | Is the Price of National Security Worth the Cost of Your Privacy?" The House intel panel pushes forward a controversial bill, sparking concerns over the government's reach into personal lives. | 0.003 | 0.68 |
| 4 | 1 | Does Congress' Rush to Pass Surveillance Bill Silence the Whispers of Privacy Fears?". | 0.002 | 0.48 |
| 4 | 2 | Spying Bill Clears Hurdle Despite Growing Concerns Over Privacy Invasion". | 0.000 | 0.63 |
| 4 | 4 | Spying Bill Clears Hurdle Despite Growing Concerns Over Privacy Invasion". | 0.000 | 0.63 |

---

## Headline 7

**Source:** "Facebook's Sandberg says has no plan to work in government"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | What's Behind Facebook's Sheryl Sandberg's Decision to Rule Out a Government Role? | 0.481 | 0.75 |
| 0 | 1 | Sandberg's Future: Will She Stay in the Private Sector or Take a Leap into Public Service?". | 0.004 | 0.57 |
| 0 | 2 | Sandberg's Future Unclear: Will She Leave Facebook's Boardroom for a Seat of Power? | 0.005 | 0.69 |
| 0 | 4 | Facebook's Sandberg Rules Out a Future in Public Office. | 0.002 | 0.82 |
| 1 | 0 | What's Behind Facebook's Top Executive's Silence on a Potential Government Role? | 0.041 | 0.70 |
| 1 | 1 | Is Facebook's Powerhouse Leader Ruling Out a Future in Politics?". | 0.166 | 0.58 |
| 1 | 2 | Sandberg's Future Unclear: Will Facebook's Top Executive Stay Out of Politics? | 0.001 | 0.74 |
| 1 | 4 | Facebook's Sandberg Rules Out a Future in Public Office. | 0.002 | 0.82 |
| 2 | 0 | What's Behind Facebook's Top Executive's Silence on a Potential Government Role? | 0.041 | 0.70 |
| 2 | 1 | Is Facebook's Powerhouse Leader Ruling Out a Future in Politics?". | 0.166 | 0.58 |
| 2 | 2 | Sandberg's Future in Government Unclear: Has Facebook's Top Executive Ruled Out a Political Career for Good? | 0.001 | 0.79 |
| 2 | 4 | Facebook's Sandberg Rules Out a Future in Public Office. | 0.002 | 0.82 |
| 4 | 0 | What's Behind Facebook's Top Executive's Silence on a Potential Government Role? | 0.041 | 0.70 |
| 4 | 1 | Is Facebook's Powerhouse Leader Ruling Out a Future in Politics?". | 0.166 | 0.58 |
| 4 | 2 | Sandberg's Future: Will Facebook's No. 2 Continue to Keep Politics at Arm's Length? | 0.003 | 0.69 |
| 4 | 4 | Facebook's Sandberg Raises Questions About What Lies Ahead After Ruling Out a Government Role. | 0.018 | 0.76 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Shocker: Facebook's Top Exec Shuts Down Government Job Rumors. | 0.003 | 0.61 |
| 0 | 1 | Facebook's Powerhouse Shuns Political Spotlight, Staying Focused on Tech. | 0.001 | 0.50 |
| 0 | 2 | Facebook's Powerhouse Shuns Political Spotlight, Staying Focused on Tech. | 0.001 | 0.50 |
| 0 | 4 | Facebook's Sandberg Rules Out a Career in Public Office. | 0.005 | 0.82 |
| 1 | 0 | Shocker: Facebook's Top Exec Shuts Down Government Job Rumors. | 0.003 | 0.61 |
| 1 | 1 | Facebook's Powerhouse Leader Shakes Off Political Rumors: No Plans to Trade in Her Role. | 0.002 | 0.61 |
| 1 | 2 | Facebook's Powerhouse Leader Shakes Off Political Rumors: No Plans to Trade in Her Role. | 0.002 | 0.61 |
| 1 | 4 | Facebook's Powerhouse Leader Shakes Off Political Rumors: No Plans to Trade in Her Role. | 0.002 | 0.61 |
| 2 | 0 | Shocker: Facebook's Top Exec Shuts Down Government Job Rumors. | 0.003 | 0.61 |
| 2 | 1 | Facebook's Powerhouse Leader Shakes the Political Establishment with Surprising Admission. | 0.007 | 0.44 |
| 2 | 2 | Facebook's Powerhouse Leader Shakes the Political Sphere with Surprising Admission. | 0.004 | 0.43 |
| 2 | 4 | Facebook's Powerhouse Rules Out a Political Play for Now. | 0.011 | 0.59 |
| 4 | 0 | Shocker: Facebook's High-Ranking Exec Suddenly Rules Out a Career Change to Politics! | 0.878 | 0.62 |
| 4 | 1 | Facebook's Powerhouse Rules Out a Political Play: Sheryl Sandberg's Future is Clear - and it's Not in Public Office. | 0.893 | 0.73 |
| 4 | 2 | Facebook's Powerhouse Rules Out a Political Play: Sheryl Sandberg's Future is Clear - and it's Not in Public Office. | 0.893 | 0.73 |
| 4 | 4 | Facebook's Powerhouse Rules Out a Political Play for Now  (Semantic sensationalism and emotional trigger: "Powerhouse" and "Play"). | 0.811 | 0.36 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Sandberg's Silence: Will Facebook's Top Executive Ever Trade in Her Tech Throne for a Seat of Power?". | 0.008 | 0.64 |
| 0 | 1 | Sandberg's Future: Will She Stay in the Private Sector or Make a Surprising Leap into Public Office?". | 0.004 | 0.56 |
| 0 | 2 | Sandberg's Future: No Public Office in Sight?". | 0.001 | 0.57 |
| 0 | 4 | Sandberg's Future in Government: No Plans to Trade in Her Facebook Badge for a Political Post". | 0.009 | 0.67 |
| 1 | 0 | Sandberg's Silence: Is Facebook's Top Executive Ruling Out a Future in Politics?". | 0.012 | 0.75 |
| 1 | 1 | Sandberg's Silence: Is Facebook's Top Exec Ruling Out a Future in Politics?". | 0.014 | 0.73 |
| 1 | 2 | Sandberg's Future: Is Facebook's Top Executive Ruling Out a Career in Politics?". | 0.036 | 0.77 |
| 1 | 4 | Sandberg's Future: Will She Stay in the Private Sector or Make a Surprisng Leap into Public Office?". | 0.003 | 0.57 |
| 2 | 0 | Is Facebook's Powerhouse Ready to Hang Up Her Hat for a Seat of Power?". | 0.824 | 0.49 |
| 2 | 1 | Sandberg's Silence: Is Facebook's Top Exec Ruling Out a Future in Politics?". | 0.014 | 0.73 |
| 2 | 2 | Sandberg's Silence: Is Facebook's Top Exec Ruling Out a Future in Politics?". | 0.014 | 0.73 |
| 2 | 4 | Sandberg's Future: Is Facebook's Top Executive Ruler of the Board - or Cabinet - Bound for Government Service?". | 0.045 | 0.73 |
| 4 | 0 | Is Facebook's Powerhouse Ready to Trade in the Boardroom for the Beltway?". | 0.008 | 0.46 |
| 4 | 1 | Is Facebook's Powerhouse Leader Shying Away from Public Office or Just Focused on Social Media Dominance?". | 0.567 | 0.57 |
| 4 | 2 | Is Facebook's Powerhouse Leader Shying Away from Public Office..."? | 0.733 | 0.57 |
| 4 | 4 | Is Facebook's Powerhouse Politically Powerless: Sandberg Rules Out Government Role". | 0.046 | 0.74 |

---

## Headline 8

**Source:** "Betting on a Trump win or a North Korea H-bomb: a predictions site gains traction"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | What's driving the surge in bets on a high-stakes outcome? | 0.002 | 0.51 |
| 0 | 1 | As Odds Shift, One Website's Unconventional Bets Spark Growing Interest. | 0.001 | 0.61 |
| 0 | 2 | As Odds of Unlikely Outcomes Draw In, One Website's Forecasts Capture Public Imagination. | 0.002 | 0.66 |
| 0 | 4 | As Odds of Unlikely Outcomes Draw In, One Website's Forecasts Capture Public Imagination. | 0.002 | 0.66 |
| 1 | 0 | What's driving the surge in bets on a high-stakes outcome? | 0.002 | 0.51 |
| 1 | 1 | As Odds Shift, One Platform's Unconventional Forecasts Spark Widespread Interest. | 0.000 | 0.60 |
| 1 | 2 | As Odds of Unpredictable Events Spark Online Frenzy, One Site's Forecasts Gain Unlikely Attention. | 0.002 | 0.62 |
| 1 | 4 | As Odds of Unpredictable Events Spark Online Frenzy, One Site's Forecasts Gain Unlikely Attention. | 0.002 | 0.62 |
| 2 | 0 | What's driving the surge in bets on a high-stakes outcome? | 0.002 | 0.51 |
| 2 | 1 | As Odds Shift, Are Experts Betting on Unlikely Outcomes or Is There More to the Story? | 0.001 | 0.48 |
| 2 | 2 | As Odds of Chaos Shift, One Site's Unconventional Forecasts Spark Widespread Intrigue. | 0.001 | 0.54 |
| 2 | 4 | As speculation swirls, where investors dare to put their money: will it be on a presidential upset or a nuclear gamble? | 0.001 | 0.65 |
| 4 | 0 | What's driving the surge in popularity for a website that's taking bold bets on the unpredictable? | 0.588 | 0.58 |
| 4 | 1 | As Odds Shift, Are Experts Betting on Unlikely Outcomes or Is There More to the Story? | 0.001 | 0.48 |
| 4 | 2 | As speculation swirls, where do fortunes lie: a predictions site's unlikely appeal? | 0.001 | 0.66 |
| 4 | 4 | As speculation swirls, where investors dare to put their money: does a Trump victory or a North Korean nuclear test hold the keys? | 0.001 | 0.65 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Gambling on Chaos: A Website's Unsettling Rise to Fame". | 0.015 | 0.55 |
| 0 | 1 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 0 | 2 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 0 | 4 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 1 | 0 | Gambling on Chaos: A Bold Bet on the Unpredictable". | 0.002 | 0.46 |
| 1 | 1 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 1 | 2 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 1 | 4 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 2 | 0 | Gambling on Chaos: A Bold New Platform Sees a Surge in Interest". | 0.001 | 0.56 |
| 2 | 1 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Spark Fear and Fascination". | 0.406 | 0.65 |
| 2 | 2 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 2 | 4 | Gambling on Chaos: One Website's Unsettlingly Accurate Forecasts Send Shockwaves". | 0.058 | 0.62 |
| 4 | 0 | Gambling on Chaos: A Bold New Platform Sees a Surge in Interest Amid Global Uncertainty". | 0.001 | 0.59 |
| 4 | 1 | Gambling on Chaos: A Bold New Platform Sees a Surge in Interest". | 0.001 | 0.56 |
| 4 | 2 | Gambling on Chaos: How Fear and Hope are Fueling Wild Speculations on Global Upheaval". | 0.041 | 0.49 |
| 4 | 4 | Gambling on Chaos: How Fear and Hope are Fueling Wild Speculations". | 0.020 | 0.50 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Will the Odds Favor a Trump Victory or a Nuclear Showdown with North Korea?". | 0.002 | 0.68 |
| 0 | 1 | Will Gamblers Cash In on Chaos? One Site's Unconventional Bets Spark Controversy". | 0.004 | 0.60 |
| 0 | 2 | Will Gamblers Cash In on Chaos? One Site's Unconventional Bets Steal Spotlight". | 0.004 | 0.61 |
| 0 | 4 | Will Gamblers Cash In on Chaos? Predictions Site Sees Surge in Bets on High-Stakes Outcomes". | 0.000 | 0.66 |
| 1 | 0 | Will the Odds Favor a Shocking Upset or a Nuclear Showdown?". | 0.003 | 0.48 |
| 1 | 1 | Will You Strike Gold or Face Nuclear Fallout? One Site's Unconventional Bets Are Rattling the Nation". | 0.005 | 0.64 |
| 1 | 2 | Will Gamblers' Fears or Hopes Come True: Mysterious Site Soars to Popularity". | 0.067 | 0.60 |
| 1 | 4 | Will Gamblers' Fears or Hopes Come True: Mysterious Site Puts Odds on Trump Victory and North Korean Nuke Threat". | 0.049 | 0.80 |
| 2 | 0 | Will the Odds Favor a Shocking Upset or a Nuclear Showdown?". | 0.003 | 0.48 |
| 2 | 1 | Will You Strike Gold or Face Nuclear Fallout? The Odds Are In, But Can You Handle the Truth?". | 0.001 | 0.29 |
| 2 | 2 | Will You Strike Gold or Face Nuclear Fallout? The Bets Are On, But Can You Afford to Lose?". | 0.001 | 0.39 |
| 2 | 4 | Will Gamblers' Fears or Dreams Come True: Mysterious Site Predicts High-Stakes Outcomes". | 0.850 | 0.63 |
| 4 | 0 | Will the Odds Favor a Shocking Upset or a Nuclear Showdown? One Site's Unconventional Predictions Have the World Watching". | 0.022 | 0.73 |
| 4 | 1 | Will You Strike It Rich or Face Nuclear Fallout? A Popular Predictions Site's Unlikely Betters Are Placing Their Chips". | 0.259 | 0.68 |
| 4 | 2 | Will You Strike It Rich or Face Nuclear Fallout? Experts Weigh In on Out-of-the-Ordinary Market Shift". | 0.001 | 0.46 |
| 4 | 4 | Will Gamers Risk It All on Doomsday Scenarios or Trump Triumphs?". | 0.696 | 0.50 |

---

## Headline 9

**Source:** "As clock ticks, Republicans try to move ahead on Obamacare repeal"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | As the deadline looms, what's behind the sudden push to revive the Obamacare overhaul? | 0.002 | 0.64 |
| 0 | 1 | As deadline looms, will Republicans succeed in revamping healthcare overhaul? | 0.000 | 0.66 |
| 0 | 2 | As deadline looms, will Republicans succeed in revamping Obama-era healthcare law? | 0.000 | 0.68 |
| 0 | 4 | As deadline looms, will Republicans succeed where others have stalled on healthcare overhaul? | 0.000 | 0.69 |
| 1 | 0 | As the deadline looms, what's behind the sudden push to revive the Obamacare overhaul? | 0.002 | 0.64 |
| 1 | 1 | As deadline looms, will Republicans succeed in their bid to overhaul healthcare reform? | 0.000 | 0.67 |
| 1 | 2 | As deadline looms, will Republicans succeed in revamping healthcare overhaul? | 0.000 | 0.66 |
| 1 | 4 | As deadline looms, will Republicans succeed where others have stalled on healthcare overhaul? | 0.000 | 0.69 |
| 2 | 0 | As the clock ticks away, what's behind the sudden push to revive the Obamacare overhaul? | 0.009 | 0.67 |
| 2 | 1 | As deadline looms, will Republicans succeed in their quest to overhaul healthcare reform? | 0.000 | 0.68 |
| 2 | 2 | As deadline looms, will Republicans succeed where others have stalled on healthcare overhaul efforts? | 0.000 | 0.70 |
| 2 | 4 | As deadline looms, will Republicans succeed where others have stalled on healthcare overhaul efforts? | 0.000 | 0.70 |
| 4 | 0 | As the clock ticks away, what will it take for Republicans to finally make progress on healthcare reform? | 0.002 | 0.65 |
| 4 | 1 | As deadline looms, are Republicans on track to revive Obamacare overhaul efforts, or will last-minute hurdles derail their plans? | 0.000 | 0.73 |
| 4 | 2 | As deadline looms, will Republicans succeed where others have stalled on healthcare reform efforts? | 0.000 | 0.70 |
| 4 | 4 | As deadline looms near, will Republicans crack open doors to Obamacare's future or slam them shut forever? | 0.001 | 0.68 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Time Running Out: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.001 | 0.67 |
| 0 | 1 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 0 | 2 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 0 | 4 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 1 | 0 | Time Running Out: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.001 | 0.67 |
| 1 | 1 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 1 | 2 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 1 | 4 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 2 | 0 | Time Running Out: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.001 | 0.67 |
| 2 | 1 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 2 | 2 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 2 | 4 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 4 | 0 | Time is running out: The fate of healthcare reform hangs in the balance as lawmakers scramble to make their move. | 0.002 | 0.67 |
| 4 | 1 | Deadline Looms: The Fate of Affordable Healthcare Hangs in the Balance as Republicans Push Forward. | 0.001 | 0.71 |
| 4 | 2 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |
| 4 | 4 | Deadline Looms: GOP Makes Last-Ditch Effort to Revamp Healthcare Law. | 0.000 | 0.64 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Will Obamacare Repeal Efforts Run Out of Time? | 0.000 | 0.65 |
| 0 | 1 | Time Running Out: Can Republicans Revive Obamacare Repeal Efforts?". | 0.001 | 0.73 |
| 0 | 2 | Time Running Out: Can Republicans Revive Obamacare Repeal Efforts?". | 0.001 | 0.73 |
| 0 | 4 | Time Running Out: Can Republicans Revive Obamacare Repeal Efforts?". | 0.001 | 0.73 |
| 1 | 0 | Will Obamacare Repeal Finally Get the Green Light as the Clock Ticks Down? | 0.001 | 0.64 |
| 1 | 1 | Will Republicans Seize the Moment to Revamp Healthcare or Let Time Run Out?". | 0.001 | 0.61 |
| 1 | 2 | Time Running Out: Can Republicans Revive Obamacare Repeal Efforts?". | 0.001 | 0.73 |
| 1 | 4 | Time Running Out: Can Republicans Revive Obamacare Repeal Efforts?". | 0.001 | 0.73 |
| 2 | 0 | Will Obamacare Repeal Finally Get the Green Light as the Clock Ticks Down? | 0.001 | 0.64 |
| 2 | 1 | Will the GOP's Last-Minute Push to Gut Healthcare Reform Succeed?". | 0.026 | 0.58 |
| 2 | 2 | Will Republicans Seize the Moment or Let Time Run Out on Obamacare Repeal Efforts?". | 0.003 | 0.69 |
| 2 | 4 | Will Republicans Seize the Clock or Let Obamacare Reform Slip Away?". | 0.001 | 0.69 |
| 4 | 0 | Will Obamacare Repeal Finally Come into Focus as Deadline Looms for Republicans' Next Move? | 0.001 | 0.70 |
| 4 | 1 | Will the GOP's Last-Ditch Effort to Gut Obamacare Succeed Before Time Runs Out?". | 0.015 | 0.63 |
| 4 | 2 | Will the GOP Keep the Clock Ticking or Hit the Reset Button on Obamacare Repeal?". | 0.004 | 0.71 |
| 4 | 4 | Will They Succeed or Fail the Clock is Ticking: GOP's Obamacare Repeal Push". | 0.765 | 0.69 |

---

## Headline 10

**Source:** "Planned Parenthood sues Indiana for abortion law"

### Curiosity + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | What's behind Indiana's latest abortion law, and why is Planned Parenthood taking a stand? | 0.570 | 0.80 |
| 0 | 1 | Abortion rights clash: Planned Parenthood takes Indiana to court. | 0.000 | 0.90 |
| 0 | 2 | Abortion rights clash: Planned Parenthood takes Indiana to court. | 0.000 | 0.90 |
| 0 | 4 | Abortion rights clash: Planned Parenthood takes Indiana to court. | 0.000 | 0.90 |
| 1 | 0 | What's behind Indiana's latest abortion law, and why is Planned Parenthood taking a stand? | 0.570 | 0.80 |
| 1 | 1 | Abortion rights clash: Can a state's strict regulations silence a healthcare provider's voice? | 0.002 | 0.61 |
| 1 | 2 | Abortion rights clash: Can a state's strict new law silence a healthcare provider's voice forever? | 0.005 | 0.63 |
| 1 | 4 | Abortion rights clash: Can Indiana's new restrictions stand up to legal scrutiny? | 0.001 | 0.76 |
| 2 | 0 | What's behind Indiana's latest abortion law, and why is Planned Parenthood taking a stand? | 0.570 | 0.80 |
| 2 | 1 | Abortion rights clash: Can a state's strict regulations silence a healthcare provider's voice forever? | 0.005 | 0.61 |
| 2 | 2 | Abortion rights clash: Can a state's strict regulations silence a healthcare provider's voice forever? | 0.005 | 0.61 |
| 2 | 4 | Abortion rights clash: Can a state's strict regulations silence a healthcare provider's voice forever? | 0.005 | 0.61 |
| 4 | 0 | What's behind Indiana's latest abortion law, and why is Planned Parenthood taking a stand? | 0.570 | 0.80 |
| 4 | 1 | Abortion rights are under fire: Can a new lawsuit save reproductive freedom in Indiana's latest controversy? | 0.005 | 0.77 |
| 4 | 2 | Abortion rights are put to the test: Can state legislatures dictate reproductive healthcare? | 0.001 | 0.61 |
| 4 | 4 | Abortion rights clash: Is Indiana's strict law worth fighting for or infringing on freedoms of all women's health providers? | 0.009 | 0.76 |

### Emotional + Sensational (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Indiana's Abortion Ban Under Fire: Planned Parenthood Takes a Stand". | 0.004 | 0.79 |
| 0 | 1 | Indiana's Abortion Ban Under Fire as Planned Parenthood Takes Legal Action". | 0.001 | 0.77 |
| 0 | 2 | Indiana's Abortion Ban Under Fire as Planned Parenthood Fights Back". | 0.002 | 0.79 |
| 0 | 4 | Indiana's Abortion Ban Under Fire as Planned Parenthood Fights Back". | 0.002 | 0.79 |
| 1 | 0 | Indiana's Extreme Abortion Ban Sparks Fury as Planned Parenthood Fights Back". | 0.302 | 0.79 |
| 1 | 1 | Indiana's Abortion Ban Under Fire: Planned Parenthood Takes Legal Action". | 0.001 | 0.81 |
| 1 | 2 | Indiana's Abortion Ban Under Fire as Planned Parenthood Takes Legal Action". | 0.001 | 0.77 |
| 1 | 4 | Indiana's Abortion Ban Under Fire as Planned Parenthood Takes Legal Action". | 0.001 | 0.77 |
| 2 | 0 | Indiana's Extreme Abortion Ban Sparks Fury as Planned Parenthood Takes a Stand". | 0.299 | 0.78 |
| 2 | 1 | Indiana's Abortion Ban Sparks Emotional Battle for Women's Rights". | 0.011 | 0.64 |
| 2 | 2 | Indiana's Abortion Ban Under Fire as Planned Parenthood Takes Legal Action". | 0.001 | 0.77 |
| 2 | 4 | Indiana's Abortion Ban Under Fire as Planned Parenthood Takes Legal Action". | 0.001 | 0.77 |
| 4 | 0 | Indiana's Extreme Abortion Ban Sparks Emotional Battle as Planned Parenthood Fights Back in Court". | 0.245 | 0.79 |
| 4 | 1 | Indiana's Extreme Abortion Ban Sparks Emotional Battle for Women's Rights". | 0.094 | 0.66 |
| 4 | 2 | Indiana's Abortion Ban Sparks Emotional Battle for Women's Rights". | 0.011 | 0.64 |
| 4 | 4 | Indiana's Abortion Ban Ignites Emotional Battle for Women's Rights as Planned Parenthood Takes Legal Action". | 0.024 | 0.77 |

### Sensational + Question (top-k = 50)

| alpha | beta | Generated headline | cb (ext.) | STS |
|---|---|---|---|---|
| 0 | 0 | Abortion Battle Ignites as Planned Parenthood Takes on Indiana's Controversial Legislation". | 0.001 | 0.76 |
| 0 | 1 | Abortion Battle Ignites as Planned Parenthood Takes Indiana to Court". | 0.000 | 0.76 |
| 0 | 2 | Abortion Battle Ignites as Planned Parenthood Takes Indiana to Court". | 0.000 | 0.76 |
| 0 | 4 | Abortion Battle Ignites as Planned Parenthood Takes Indiana to Court". | 0.000 | 0.76 |
| 1 | 0 | Abortion Battle Ignites as Planned Parenthood Takes on Indiana's Controversial New Law". | 0.002 | 0.77 |
| 1 | 1 | Abortion Battle Ignites as Planned Parenthood Takes on Indiana's Controversial Legislation". | 0.001 | 0.76 |
| 1 | 2 | Abortion Battle Ignites as Planned Parenthood Takes on Indiana's Controversial Legislation". | 0.001 | 0.76 |
| 1 | 4 | Abortion Battle Ignites as Planned Parenthood Takes on Indiana's Controversial Legislation". | 0.001 | 0.76 |
| 2 | 0 | Is Indiana's Abortion Law Under Fire as Planned Parenthood Takes a Stand?". | 0.041 | 0.69 |
| 2 | 1 | Abortion Battle Ignites: Planned Parenthood Takes on Indiana's Controversial Law". | 0.004 | 0.80 |
| 2 | 2 | Abortion Battle Ignites: Planned Parenthood Takes on Indiana's Controversial Law". | 0.004 | 0.80 |
| 2 | 4 | Abortion Battle Ignites as Planned Parenthood Takes on Indiana's Controversial Legislation". | 0.001 | 0.76 |
| 4 | 0 | Is Indiana's Abortion Law Under Fire as Planned Parenthood Takes a Stand?". | 0.041 | 0.69 |
| 4 | 1 | Is Indiana's Abortion Law Under Fire as Planned Parenthood Takes Legal Action?". | 0.011 | 0.70 |
| 4 | 2 | Is Indiana's Abortion Law a Threat to Women's Rights? Planned Parenthood Takes Legal Action". | 0.004 | 0.74 |
| 4 | 4 | Is Indiana's Abortion Restrictor-riendly Law Target for Legal Firepower from Planned Parenthood Forces Now Fighting Back in Courtroom Showdowns?". | 0.955 | 0.76 |

---
