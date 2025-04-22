import requests
from bs4 import BeautifulSoup
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import re
import numpy as np
import unicodedata
import datetime

# Step 1: Fetch the webpage content
url = "https://338canada.com/districts.htm"
request_time = datetime.datetime.now()
timestamp_str = request_time.strftime("%Y-%m-%d %H:%M:%S")
response = requests.get(url)

# Step 2: Parse the HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Step 3: Locate the table with id="myTable"
table = soup.find("table", id="myTable")

# Step 4: Extract table headers
thead = table.find("thead")
if thead:
    header_row = thead
else:
    header_row = table.find("tr")

headers = [cell.get_text(strip=True) for cell in header_row.find_all(["th", "td"])]
headers.append("BgColor_LatestProjection")  # Add column for background colors

# Step 5: Extract table rows with background colors
rows = []
for tr in table.find("tbody").find_all("tr"):
    cells_text = []
    td_elements = tr.find_all("td")
    
    # Get text from all cells
    for td in td_elements:
        cells_text.append(td.get_text(strip=True))
    
    if cells_text:
        # Find the "Latest projection" column index
        try:
            latest_proj_idx = headers.index("Latest projection")
            
            # Extract background color from the "Latest projection" cell
            if latest_proj_idx < len(td_elements):
                td = td_elements[latest_proj_idx]
                
                # Try to get bgcolor attribute first
                bg_color = td.get('bgcolor')
                
                # If no bgcolor, try extracting from style attribute
                if not bg_color and td.get('style'):
                    style = td.get('style')
                    bg_match = re.search(r'background-color:\s*(#[0-9a-fA-F]+|rgb\([^)]+\))', style)
                    if bg_match:
                        bg_color = bg_match.group(1)
                
                # If still no color found, set to None
                if not bg_color:
                    bg_color = None
                    
                # Append the background color to the row data
                cells_text.append(bg_color)
            else:
                cells_text.append(None)
        except ValueError:
            # "Latest projection" column not found in headers
            cells_text.append(None)
        
        rows.append(cells_text)

# Step 6: Create a DataFrame
df = pd.DataFrame(rows)
df.columns = headers

def normalize_district_name(name):
    """
    Normalize district names to make them comparable despite encoding differences
    """
    if not isinstance(name, str):
        return ""
    
    # Try to fix any encoding issues
    try:
        # Handle potential double-encoded UTF-8
        name = name.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    
    # Normalize Unicode characters
    name = unicodedata.normalize('NFKD', name)
    
    # Replace various dash types with standard hyphen
    name = name.replace('\u2014', '-')  # em dash
    name = name.replace('\u2013', '-')  # en dash
    name = name.replace('\u2019', "'")  # right single quotation mark
    name = name.replace('\u2018', "'")  # left single quotation mark
    name = re.sub(r'â\x80[\x93\x94\x99]', '-', name)  # common encoding errors for dashes
    
    # Make lowercase for easier comparison
    name = name.lower()
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name

# Clean the Electoral district column
df['Electoral district'] = df['Electoral district'].str.replace(r'^\d+\s+', '', regex=True).str.strip()

# Step 7: Load the shapefile
shapefile_path = "shapefiles/CF_CA_2023_FR.shp"  # Update path as needed
gdf = gpd.read_file(shapefile_path)

# Apply normalization to both datasets
gdf['normalized_name'] = gdf['CF_NOMAN'].apply(normalize_district_name)
df['normalized_district'] = df['Electoral district'].apply(normalize_district_name)

# Merge the GeoDataFrame with the DataFrame
merged = gdf.merge(df[['normalized_district', 'Latest projection']], 
                 left_on='normalized_name',
                 right_on='normalized_district',
                 how='left')

# Step 8: Define party colors
party_colors = {
    "LPC": {
        "safe": "#e31a1c",    # dark red
        "likely": "#fc9272",  # light red
        "leaning": "#fcbba1"  # in-between red
    },
    "CPC": {
        "safe": "#3182bd",    # dark blue
        "likely": "#9ecae1",  # light blue
        "leaning": "#c6dbef"   # in-between blue
    },
    "NDP": {
        "safe": "#31a354",    # dark green
        "likely": "#a1d99b",  # light green
        "leaning": "#c7e9c0"   # in-between green
    },
    "BQ": {
        "safe": "#762a83",    # dark purple
        "likely": "#d7b5d8",  # light purple
        "leaning": "#e7d4e8"   # in-between purple
    },
    "GPC": {
        "safe": "#006400",    # dark green
        "likely": "#90ee90",  # light green
        "leaning": "#a1e3a1"   # in-between green
    }
}

# Helper functions for mixing hex colors
def hex_to_rgb(hexcolor):
    hexcolor = hexcolor.lstrip("#")
    return tuple(int(hexcolor[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb

def mix_colors(hex_colors):
    rgbs = [hex_to_rgb(h) for h in hex_colors]
    avg_rgb = tuple(int(sum(channels)/len(rgbs)) for channels in zip(*rgbs))
    return rgb_to_hex(avg_rgb)

def get_projection_color(projection):
    # Handle missing values
    if not projection or (isinstance(projection, float) and np.isnan(projection)):
        return "grey"
    
    projection = projection.strip()
    
    # If the projection indicates a tossup, mix the safe colors of the parties involved.
    if projection.startswith("Toss up"):
        parties = projection.replace("Toss up", "").strip().split("/")
        safe_colors = []
        for party in parties:
            party = party.strip().upper()
            if party in party_colors:
                safe_colors.append(party_colors[party]["safe"])
        if safe_colors:
            return mix_colors(safe_colors)
        else:
            return "grey"
    else:
        # Match projections like "LPC safe", "CPC likely", "NDP leaning", etc.
        m = re.match(r'^(LPC|CPC|NDP|BQ|GPC)\s+(safe|likely|leaning)$', projection, re.IGNORECASE)
        if m:
            party, rating = m.groups()
            party = party.upper()
            rating = rating.lower()
            if party in party_colors and rating in party_colors[party]:
                return party_colors[party][rating]
        return "grey"

# Update the merged GeoDataFrame with a new column "mapped_color" based on Latest projection.
merged["mapped_color"] = merged["Latest projection"].apply(get_projection_color)

# Step 9: Create the map
fig, ax = plt.subplots(figsize=(15, 15))
merged.plot(ax=ax, color=merged["mapped_color"], edgecolor="black", linewidth=0.5)
ax.set_title(f"Canadian Electoral Districts\nData queried at {timestamp_str}", fontsize=16)

# Create legend entries
legend_entries = []

# Add individual party projections
for party, ratings in party_colors.items():
    for rating, color in ratings.items():
        legend_entries.append((f"{party} {rating}", color))

# Add toss-up combinations
tossup_combinations = [proj for proj in merged["Latest projection"].dropna().unique() 
                       if "Toss up" in str(proj)]
for combo in tossup_combinations:
    legend_entries.append((combo, get_projection_color(combo)))

# Add grey for no data
legend_entries.append(("No data", "grey"))

# Add legend
for label, color in legend_entries:
    ax.scatter([], [], color=color, label=label, s=100)

# Position the legend
legend = ax.legend(
    title="Party Projections", 
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=10,
    frameon=True
)

# Adjust layout for legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)

# Save images - regular and high resolution versions
plt.savefig("docs/canada_electoral_map.png", dpi=150, bbox_inches="tight")
plt.savefig("docs/canada_electoral_map_highres.png", dpi=300, bbox_inches="tight")
plt.savefig("docs/canada_electoral_map_vector.pdf", bbox_inches="tight")
plt.savefig("docs/canada_electoral_map_vector.svg", bbox_inches="tight")

print(f"Map generated successfully at {timestamp_str}")