# NBADraft
Predictive Models to analyze outcomes for drafted players

Player stats from [Tankathon](https://www.tankathon.com/) and [Basketball Reference](https://www.basketball-reference.com/)

Files:
- `Tankathon_scraper.ipynb`: webscrape college/international stats for draft prospects since 2004
- `allstars.ipynb`: code used to create models to predict future All-Stars
- `skills.ipynb`: code used to create models for creation, shooting, and defense
- `model_helpers.py`: contains functions for data cleaning and visualization that could be used for future projects that utilize the data scraped here
- `outcomes_scraper.ipynb`: code to webscrape NBA outcomes data, which can be used as response variables in future projects for draft projection
- `results2024.csv`: All-star probabilities for 2024 NBA Draft prospects after running the code in `allstar.ipynb`
- `skill_results2024.csv`: predicted PTS/36, TS%, AST/36, TOV%, 3PM/36, and DBPM for 2024 NBA Draft prospects after running the code in `skills.ipynb`

Data: 
- `draft_boards24.csv`: contains draft board ranks for 5 major draft ranking sites (Tankathon, ESPN, CBS Sports, The Ringer, Rookie Scale) [LAST UPDATED: 06/22/2024]
- `draft_players.csv`: contains college/international stats for prospects between 2004 and 2023
- `draft_players24.csv`: contains college/international stats for 2024 NBA Draft prospects [LAST UPDATED: 06/22/2024]
- `outcomes.csv`: contains NBA career outcomes for draft prospects between 2004 and 2020
