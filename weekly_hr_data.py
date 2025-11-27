'''
Collects the HR data for each week in the 2023 season
'''
from datetime import datetime, timedelta
from pybaseball import statcast
import pandas as pd
import time

# Still need to implement this into the player selection reward portion for accurate player rewards. 
weeks = []
start = datetime(2023, 3, 30)
end = datetime(2023, 10, 1)

week_start = start

while week_start < end:
    week_end = week_start + timedelta(days=6)
    
    try:
        print(f"Fetching Statcast data for {week_start.date()} to {week_end.date()}...")
        
        # Get all batted ball data for the week
        df = statcast(start_dt=str(week_start.date()), end_dt=str(week_end.date()))
        
        if df is not None and not df.empty:
            # Filter for home runs only
            hr_df = df[df['events'] == 'home_run']
            
            # Count HRs per player
            hr_counts = hr_df.groupby(['player_name', 'batter']).size().reset_index(name='HR')
            hr_counts.columns = ['Name', 'playerid', 'HR']
            hr_counts['week_start'] = week_start.date()
            hr_counts['week_end'] = week_end.date()
            
            weeks.append(hr_counts)
        
        time.sleep(2)  # Be nice to the API
        
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
    
    week_start += timedelta(days=7)

if weeks:
    full_weekly_hr = pd.concat(weeks, ignore_index=True)
    full_weekly_hr.to_csv('weekly_hr_2023.csv', index=False)
    print("Data saved successfully!")