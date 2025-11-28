'''
Collects the HR data for each week in the 2023/2024 season
'''

from datetime import datetime, timedelta
from pybaseball import statcast, playerid_reverse_lookup
import pandas as pd
import time
import pandas as pd
import unicodedata

weeks = []
start = datetime(2024, 3, 28)
end = datetime(2024, 9, 29)

week_start = start

while week_start < end:
    week_end = week_start + timedelta(days=6)
    
    try:
        print(f"Fetching Statcast data for {week_start.date()} to {week_end.date()}...")
        
        # Get all batted ball data for the week
        df = statcast(start_dt=str(week_start.date()), end_dt=str(week_end.date()))
        
        if df is not None and not df.empty:
            # Filter for home runs only
            hr_df = df[df['events'] == 'home_run'].copy()
            
            print(f"  Found {len(hr_df)} home runs")
            
            # Get unique batter IDs
            unique_batters = hr_df['batter'].unique()
            
            # Look up batter names using playerid_reverse_lookup
            try:
                batter_names = playerid_reverse_lookup(unique_batters, key_type='mlbam')
                # Create a mapping of batter ID to name
                id_to_name = dict(zip(batter_names['key_mlbam'], 
                                     batter_names['name_last'] + ', ' + batter_names['name_first']))
            except:
                print("  Warning: Could not look up all batter names, using IDs only")
                id_to_name = {bid: f"Player_{bid}" for bid in unique_batters}
            
            # Count HRs per batter
            hr_counts = hr_df.groupby('batter').size().reset_index(name='HR')
            hr_counts['Name'] = hr_counts['batter'].map(id_to_name)
            hr_counts.columns = ['playerid', 'HR', 'Name']
            hr_counts = hr_counts[['Name', 'playerid', 'HR']] 
            hr_counts['week_start'] = week_start.date()
            hr_counts['week_end'] = week_end.date()
            
            weeks.append(hr_counts)
        
        time.sleep(2) 
        
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
    
    week_start += timedelta(days=7)

if weeks:
    full_weekly_hr = pd.concat(weeks, ignore_index=True)
    full_weekly_hr.to_csv('weekly_hr_2024_batters.csv', index=False)
    print("\nTop HR hitters:")
    print(full_weekly_hr.groupby('Name')['HR'].sum().sort_values(ascending=False).head(10))
else:
    print("Failed :(")



weekly_hr = pd.read_csv('weekly_hr_2024_batters.csv')

# Top 50 players (including some more since the names were not easily extracted)
top_50_names = [
    "Ronald Acuna Jr.", "Freddie Freeman", "Mookie Betts", "Shohei Ohtani",
    "Matt Olson", "Corey Seager", "Marcus Semien", "Juan Soto",
    "Bobby Witt Jr.", "William Contreras", "Julio Rodriguez", "Adley Rutschman",
    "Francisco Lindor", "Corbin Carroll", "Austin Riley", "Yandy Diaz",
    "Kyle Tucker", "Luis Robert Jr.", "J.P. Crawford", "Dansby Swanson",
    "Adolis Garcia", "Gunnar Henderson", "Jose Ramirez", "Xander Bogaerts",
    "Nico Hoerner", "Cody Bellinger", "Alex Bregman", "Ketel Marte",
    "Cal Raleigh", "Isaac Paredes", "Brandon Nimmo", "Ha-seong Kim",
    "Ozzie Albies", "TJ Friedl", "Trea Turner", "Bryson Stott",
    "Fernando Tatis Jr.", "Christian Walker", "Bo Bichette", "Michael Harris II",
    "James Outman", "Christian Yelich", "Will Smith", "Andres Gimenez",
    "Gleyber Torres", "Bryce Harper", "Rafael Devers", "Thairo Estrada",
    "Eugenio Suarez", "Luis Arraez", "Aaron Judge", "Vladimir Guerrero Jr.",
    "Mike Trout"
]

def normalize_name(name):
    """Normalize names for matching - handles accents, punctuation, case"""
    
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    # Remove all punctuation and convert to lowercase
    name = name.replace('.', '').replace(',', '').replace('-', '').strip().lower()
    return ' '.join(name.split())  # Normalize whitespace

def reverse_name_normalized(name):
    """Convert 'First Last' to 'last first' (normalized)"""
    parts = name.split()
    if len(parts) == 2:
        # "First Last" -> "last first"
        return f"{parts[1]} {parts[0]}"
    elif len(parts) == 3:
        # Handle Jr., II, etc OR middle names
        if parts[2].lower() in ['jr', 'jr.', 'ii', 'iii', 'sr', 'sr.']:
            # "First Last Jr." -> "last first jr"
            return f"{parts[1]} {parts[0]} {parts[2]}"
        else:
            # Assume compound last name: "First Middle Last" -> "last first middle"
            return f"{parts[2]} {parts[0]} {parts[1]}"
    elif len(parts) == 4:
        # "First Middle Last Jr." -> "last first middle jr"
        return f"{parts[2]} {parts[0]} {parts[1]} {parts[3]}"
    return name

# Normalize both datasets
weekly_hr['Name_normalized'] = weekly_hr['Name'].apply(normalize_name)

# Create mapping of normalized names
top_50_mapping = {}
for original_name in top_50_names:
    normalized = normalize_name(original_name)
    reversed_normalized = reverse_name_normalized(normalized)
    top_50_mapping[reversed_normalized] = original_name

# Filter for top 50 players
filtered_hr = weekly_hr[weekly_hr['Name_normalized'].isin(top_50_mapping.keys())].copy()

# Add back the original standardized name
filtered_hr['Player_Name'] = filtered_hr['Name_normalized'].map(top_50_mapping)

# Group by player name (id are not consistent over both datasets)
player_weekly_hr = filtered_hr.groupby(['Player_Name', 'week_start', 'week_end']).agg({
    'HR': 'sum',
    'playerid': 'first' 
}).reset_index()

# Sort by week and player
player_weekly_hr = player_weekly_hr.sort_values(['week_start', 'Player_Name'])

# Reorder columns
player_weekly_hr = player_weekly_hr[['playerid', 'Player_Name', 'week_start', 'week_end', 'HR']]

unique_weeks = sorted(player_weekly_hr['week_start'].unique())
week_number_map = {week: i+1 for i, week in enumerate(unique_weeks)}

# Add week number column (for DQN rewards)
player_weekly_hr['week_number'] = player_weekly_hr['week_start'].map(week_number_map)

# Save to a csv file
player_weekly_hr.to_csv('top_50_weekly_hr_2024.csv', index=False)

print(player_weekly_hr.head(20))