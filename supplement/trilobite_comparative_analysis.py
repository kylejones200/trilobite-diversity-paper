"""
Comparative biology analysis of trilobite occurrence data from the Paleobiology Database.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 3
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean trilobite occurrence data from PBDB CSV.

    Args:
        filepath: Path to the PBDB CSV file.

    Returns:
        DataFrame with cleaned occurrence data including calculated midpoint ages.
    """
    logger.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath, skiprows=19)
    df.columns = df.columns.str.strip('"')
    
    numeric_cols = ['max_ma', 'min_ma', 'lng', 'lat', 'paleolng', 'paleolat']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['mid_ma'] = (df['max_ma'] + df['min_ma']) / 2
    
    df = df[df['accepted_name'].notna()]
    df = df[df['accepted_name'] != 'Trilobita']
    
    logger.info(f"Loaded {len(df)} occurrences, {df['accepted_name'].nunique()} unique taxa")
    logger.info(f"Time range: {df['max_ma'].max():.1f} - {df['min_ma'].min():.1f} Ma")
    
    return df


def prepare_taxonomic_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Extract taxonomic diversity metrics.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        Tuple of (rank_counts, top_genera) where rank_counts is occurrences by
        taxonomic rank and top_genera is the top 15 genera by occurrence count.
    """
    df['genus'] = df['accepted_name'].str.split().str[0]
    rank_counts = df['accepted_rank'].value_counts()
    top_genera = df['genus'].value_counts().head(15)
    
    return rank_counts, top_genera


def plot_taxonomic_diversity(
    rank_counts: pd.Series,
    top_genera: pd.Series,
    output_path: str = 'taxonomic_diversity.png'
) -> None:
    """Create taxonomic diversity visualization.

    Args:
        rank_counts: Occurrences by taxonomic rank.
        top_genera: Top 15 genera by occurrence count.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(range(len(rank_counts)), rank_counts.values, color='black', width=0.6)
    axes[0].set_xticks(range(len(rank_counts)))
    axes[0].set_xticklabels(rank_counts.index, rotation=45, ha='right')
    axes[0].set_xlabel('Taxonomic Rank')
    axes[0].set_ylabel('Occurrences')
    axes[0].set_title('Occurrences by Taxonomic Rank')
    
    axes[1].barh(range(len(top_genera)), top_genera.values, color='black', height=0.6)
    axes[1].set_yticks(range(len(top_genera)))
    axes[1].set_yticklabels(top_genera.index)
    axes[1].set_xlabel('Occurrences')
    axes[1].set_title('Top 15 Genera')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def prepare_temporal_data(df: pd.DataFrame, bin_size: float = 10.0) -> pd.DataFrame:
    """Prepare temporal diversity data.

    Args:
        df: DataFrame with occurrence data.
        bin_size: Size of time bins in millions of years.

    Returns:
        DataFrame with temporal diversity metrics by time bin.
    """
    df['time_bin'] = (df['mid_ma'] // bin_size) * bin_size
    
    temporal_diversity = df.groupby('time_bin').agg({
        'accepted_name': 'nunique',
        'occurrence_no': 'count'
    }).rename(columns={'accepted_name': 'unique_taxa', 'occurrence_no': 'occurrences'})
    
    return temporal_diversity.sort_index(ascending=False)


def plot_temporal_patterns(
    temporal_diversity: pd.DataFrame,
    output_path: str = 'temporal_patterns.png'
) -> None:
    """Create temporal diversity visualization.

    Args:
        temporal_diversity: DataFrame with temporal diversity metrics.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(temporal_diversity.index, temporal_diversity['unique_taxa'],
                 marker='o', color='black', linestyle='-', markersize=3)
    axes[0].set_xlabel('Time (Ma)')
    axes[0].set_ylabel('Unique Taxa')
    axes[0].set_title('Trilobite Diversity Through Time')
    axes[0].invert_xaxis()
    
    axes[1].bar(temporal_diversity.index, temporal_diversity['occurrences'],
                color='black', width=8, alpha=0.6)
    axes[1].set_xlabel('Time (Ma)')
    axes[1].set_ylabel('Occurrences')
    axes[1].set_title('Trilobite Occurrences Through Time')
    axes[1].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def prepare_geographic_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Prepare geographic distribution data.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        Tuple of (country_counts, geo_df) where country_counts is occurrences
        by country and geo_df contains coordinates and ages.
    """
    country_counts = df['cc'].value_counts().head(10)
    geo_df = df[['lng', 'lat', 'mid_ma']].dropna()
    
    return country_counts, geo_df


def plot_geographic_distribution(
    country_counts: pd.Series,
    geo_df: pd.DataFrame,
    output_path: str = 'geographic_distribution.png'
) -> None:
    """Create geographic distribution visualization.

    Args:
        country_counts: Top 10 countries by occurrence count.
        geo_df: DataFrame with longitude, latitude, and age data.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].barh(range(len(country_counts)), country_counts.values, color='black', height=0.6)
    axes[0].set_yticks(range(len(country_counts)))
    axes[0].set_yticklabels(country_counts.index)
    axes[0].set_xlabel('Occurrences')
    axes[0].set_title('Occurrences by Country')
    
    scatter = axes[1].scatter(geo_df['lng'], geo_df['lat'], c=geo_df['mid_ma'],
                             cmap='gray', alpha=0.4, s=8, edgecolors='none')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Geographic Distribution')
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Age (Ma)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def prepare_top_taxa_comparison(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Prepare comparative data for top taxa.

    Args:
        df: DataFrame with occurrence data.
        n: Number of top taxa to analyze.

    Returns:
        DataFrame with comparative statistics for top taxa.
    """
    top_taxa = df['accepted_name'].value_counts().head(n).index.tolist()
    
    comparison_data = []
    for taxon in top_taxa:
        taxon_data = df[df['accepted_name'] == taxon]
        comparison_data.append({
            'taxon': taxon,
            'occurrences': len(taxon_data),
            'mean_age': taxon_data['mid_ma'].mean(),
            'age_range': taxon_data['mid_ma'].max() - taxon_data['mid_ma'].min(),
            'lat_range': taxon_data['lat'].max() - taxon_data['lat'].min() if taxon_data['lat'].notna().any() else np.nan,
            'unique_countries': taxon_data['cc'].nunique(),
        })
    
    return pd.DataFrame(comparison_data)


def plot_top_taxa_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = 'top_taxa_comparison.png'
) -> None:
    """Create comparative visualization for top taxa.

    Args:
        comparison_df: DataFrame with comparative statistics.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    n = len(comparison_df)
    y_pos = range(n)
    
    axes[0, 0].barh(y_pos, comparison_df['occurrences'], color='black', height=0.6)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels([t[:25] + '...' if len(t) > 25 else t for t in comparison_df['taxon']], fontsize=7)
    axes[0, 0].set_xlabel('Occurrences')
    axes[0, 0].set_title('Occurrence Count')
    axes[0, 0].invert_yaxis()
    
    axes[0, 1].barh(y_pos, comparison_df['age_range'], color='black', height=0.6)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels([t[:25] + '...' if len(t) > 25 else t for t in comparison_df['taxon']], fontsize=7)
    axes[0, 1].set_xlabel('Temporal Range (Myr)')
    axes[0, 1].set_title('Temporal Range')
    axes[0, 1].invert_yaxis()
    
    axes[1, 0].barh(y_pos, comparison_df['lat_range'], color='black', height=0.6)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels([t[:25] + '...' if len(t) > 25 else t for t in comparison_df['taxon']], fontsize=7)
    axes[1, 0].set_xlabel('Latitudinal Range (degrees)')
    axes[1, 0].set_title('Geographic Spread')
    axes[1, 0].invert_yaxis()
    
    axes[1, 1].barh(y_pos, comparison_df['unique_countries'], color='black', height=0.6)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([t[:25] + '...' if len(t) > 25 else t for t in comparison_df['taxon']], fontsize=7)
    axes[1, 1].set_xlabel('Number of Countries')
    axes[1, 1].set_title('Geographic Distribution')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def prepare_environmental_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Prepare environmental preference data.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        Tuple of (env_counts, lith_counts) with occurrence counts by environment
        and lithology.
    """
    env_counts = df['environment'].value_counts()
    lith_counts = df['lithology1'].value_counts().head(10)
    
    return env_counts, lith_counts


def plot_environmental_preferences(
    env_counts: pd.Series,
    lith_counts: pd.Series,
    output_path: str = 'environmental_preferences.png'
) -> None:
    """Create environmental preferences visualization.

    Args:
        env_counts: Occurrences by environment type.
        lith_counts: Top 10 lithologies by occurrence count.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if len(env_counts) > 0:
        top_env = env_counts.head(10)
        axes[0].barh(range(len(top_env)), top_env.values, color='black', height=0.6)
        axes[0].set_yticks(range(len(top_env)))
        axes[0].set_yticklabels([e[:30] + '...' if len(e) > 30 else e for e in top_env.index])
        axes[0].set_xlabel('Occurrences')
        axes[0].set_title('Occurrences by Environment')
    
    if len(lith_counts) > 0:
        axes[1].barh(range(len(lith_counts)), lith_counts.values, color='black', height=0.6)
        axes[1].set_yticks(range(len(lith_counts)))
        axes[1].set_yticklabels([l[:30] + '...' if len(l) > 30 else l for l in lith_counts.index])
        axes[1].set_xlabel('Occurrences')
        axes[1].set_title('Occurrences by Lithology')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def prepare_clustering_features(df: pd.DataFrame, n: int = 50) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """Prepare feature matrix for hierarchical clustering of taxa.

    Args:
        df: DataFrame with occurrence data.
        n: Number of top taxa to include in clustering.

    Returns:
        Tuple of (feature_df, feature_matrix, taxon_labels) where feature_df
        contains the features, feature_matrix is the normalized feature array,
        and taxon_labels is the list of taxon names.
    """
    top_taxa = df['accepted_name'].value_counts().head(n).index.tolist()
    
    feature_data = []
    for taxon in top_taxa:
        taxon_data = df[df['accepted_name'] == taxon]
        
        mean_age = taxon_data['mid_ma'].mean()
        age_range = taxon_data['mid_ma'].max() - taxon_data['mid_ma'].min()
        lat_mean = taxon_data['lat'].mean() if taxon_data['lat'].notna().any() else np.nan
        lat_range = taxon_data['lat'].max() - taxon_data['lat'].min() if taxon_data['lat'].notna().any() else 0.0
        lng_mean = taxon_data['lng'].mean() if taxon_data['lng'].notna().any() else np.nan
        unique_countries = taxon_data['cc'].nunique()
        occurrences = len(taxon_data)
        
        feature_data.append({
            'taxon': taxon,
            'mean_age': mean_age,
            'age_range': age_range,
            'lat_mean': lat_mean,
            'lat_range': lat_range,
            'lng_mean': lng_mean,
            'unique_countries': unique_countries,
            'occurrences': occurrences,
        })
    
    feature_df = pd.DataFrame(feature_data)
    
    # Remove rows with missing geographic data
    feature_df = feature_df.dropna(subset=['lat_mean', 'lng_mean'])
    
    # Select features for clustering
    feature_cols = ['mean_age', 'age_range', 'lat_mean', 'lat_range', 'lng_mean', 'unique_countries']
    feature_matrix = feature_df[feature_cols].values
    
    # Normalize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    taxon_labels = feature_df['taxon'].tolist()
    
    return feature_df, feature_matrix_scaled, taxon_labels


def plot_dendrogram(
    feature_matrix: np.ndarray,
    taxon_labels: list,
    output_path: str = 'hierarchical_clustering_dendrogram.png'
) -> None:
    """Create hierarchical clustering dendrogram.

    Args:
        feature_matrix: Normalized feature matrix for clustering.
        taxon_labels: List of taxon names corresponding to rows.
        output_path: Path to save the figure.
    """
    # Compute distance matrix and linkage
    distances = pdist(feature_matrix, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')
    
    # Create dendrogram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dendrogram(
        linkage_matrix,
        labels=taxon_labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )
    
    ax.set_xlabel('Taxon')
    ax.set_ylabel('Distance')
    ax.set_title('Hierarchical Clustering of Trilobite Taxa')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def prepare_formation_zone_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare formation and zone relationship data.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        Tuple of (formation_data, zone_data) with occurrence counts and
        temporal ranges for formations and zones.
    """
    formation_data = []
    formations = df[df['formation'].notna()]['formation'].unique()
    
    for formation in formations:
        form_df = df[df['formation'] == formation]
        formation_data.append({
            'formation': formation,
            'occurrences': len(form_df),
            'unique_taxa': form_df['accepted_name'].nunique(),
            'mean_age': form_df['mid_ma'].mean(),
            'age_range': form_df['mid_ma'].max() - form_df['mid_ma'].min(),
            'zones': form_df['zone'].notna().sum(),
            'unique_zones': form_df[form_df['zone'].notna()]['zone'].nunique(),
        })
    
    formation_df = pd.DataFrame(formation_data).sort_values('occurrences', ascending=False)
    
    zone_data = []
    zones = df[df['zone'].notna()]['zone'].unique()
    
    for zone in zones:
        zone_df = df[df['zone'] == zone]
        zone_data.append({
            'zone': zone,
            'occurrences': len(zone_df),
            'unique_taxa': zone_df['accepted_name'].nunique(),
            'mean_age': zone_df['mid_ma'].mean(),
            'age_range': zone_df['mid_ma'].max() - zone_df['mid_ma'].min(),
            'formations': zone_df['formation'].notna().sum(),
            'unique_formations': zone_df[zone_df['formation'].notna()]['formation'].nunique(),
        })
    
    zone_df = pd.DataFrame(zone_data).sort_values('occurrences', ascending=False)
    
    return formation_df, zone_df


def prepare_biostratigraphic_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate biostratigraphic ranges for taxa within zones.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        DataFrame with first and last appearances of taxa in each zone.
    """
    zone_df = df[df['zone'].notna()].copy()
    
    biostrat_data = []
    
    for zone in zone_df['zone'].unique():
        zone_taxa = zone_df[zone_df['zone'] == zone]
        
        for taxon in zone_taxa['accepted_name'].unique():
            taxon_zone = zone_taxa[zone_taxa['accepted_name'] == taxon]
            
            biostrat_data.append({
                'zone': zone,
                'taxon': taxon,
                'first_appearance_ma': taxon_zone['mid_ma'].max(),
                'last_appearance_ma': taxon_zone['mid_ma'].min(),
                'occurrences': len(taxon_zone),
                'mean_age': taxon_zone['mid_ma'].mean(),
            })
    
    biostrat_df = pd.DataFrame(biostrat_data)
    
    return biostrat_df


def prepare_zone_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate diversity patterns by zone.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        DataFrame with diversity metrics for each zone.
    """
    zone_df = df[df['zone'].notna()].copy()
    
    zone_diversity = zone_df.groupby('zone').agg({
        'accepted_name': 'nunique',
        'occurrence_no': 'count',
        'mid_ma': ['mean', 'min', 'max'],
    }).reset_index()
    
    zone_diversity.columns = ['zone', 'unique_taxa', 'occurrences', 'mean_age', 'min_age', 'max_age']
    zone_diversity['age_range'] = zone_diversity['max_age'] - zone_diversity['min_age']
    zone_diversity = zone_diversity.sort_values('mean_age', ascending=False)
    
    return zone_diversity


def plot_formation_zone_relationships(
    formation_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    output_path: str = 'formation_zone_relationships.png'
) -> None:
    """Create visualization of formation and zone relationships.

    Args:
        formation_df: DataFrame with formation data.
        zone_df: DataFrame with zone data.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    top_formations = formation_df.head(15)
    top_zones = zone_df.head(15)
    
    axes[0, 0].barh(range(len(top_formations)), top_formations['occurrences'], 
                     color='black', height=0.6)
    axes[0, 0].set_yticks(range(len(top_formations)))
    axes[0, 0].set_yticklabels([f[:25] + '...' if len(f) > 25 else f 
                               for f in top_formations['formation']], fontsize=7)
    axes[0, 0].set_xlabel('Occurrences')
    axes[0, 0].set_title('Top 15 Formations by Occurrences')
    axes[0, 0].invert_yaxis()
    
    axes[0, 1].barh(range(len(top_formations)), top_formations['unique_taxa'], 
                    color='black', height=0.6)
    axes[0, 1].set_yticks(range(len(top_formations)))
    axes[0, 1].set_yticklabels([f[:25] + '...' if len(f) > 25 else f 
                               for f in top_formations['formation']], fontsize=7)
    axes[0, 1].set_xlabel('Unique Taxa')
    axes[0, 1].set_title('Taxonomic Diversity by Formation')
    axes[0, 1].invert_yaxis()
    
    axes[1, 0].barh(range(len(top_zones)), top_zones['occurrences'], 
                     color='black', height=0.6)
    axes[1, 0].set_yticks(range(len(top_zones)))
    axes[1, 0].set_yticklabels([z[:30] + '...' if len(z) > 30 else z 
                               for z in top_zones['zone']], fontsize=7)
    axes[1, 0].set_xlabel('Occurrences')
    axes[1, 0].set_title('Top 15 Zones by Occurrences')
    axes[1, 0].invert_yaxis()
    
    axes[1, 1].barh(range(len(top_zones)), top_zones['unique_taxa'], 
                     color='black', height=0.6)
    axes[1, 1].set_yticks(range(len(top_zones)))
    axes[1, 1].set_yticklabels([z[:30] + '...' if len(z) > 30 else z 
                               for z in top_zones['zone']], fontsize=7)
    axes[1, 1].set_xlabel('Unique Taxa')
    axes[1, 1].set_title('Taxonomic Diversity by Zone')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def plot_biostratigraphic_ranges(
    biostrat_df: pd.DataFrame,
    df: pd.DataFrame,
    output_path: str = 'biostratigraphic_ranges.png'
) -> None:
    """Create biostratigraphic range chart showing zone and key taxon ranges.

    Args:
        biostrat_df: DataFrame with biostratigraphic range data.
        df: Original DataFrame with occurrence data.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Top plot: Zone ranges through time
    zone_df = df[df['zone'].notna()].copy()
    zone_ranges = zone_df.groupby('zone').agg({
        'mid_ma': ['min', 'max', 'mean'],
        'accepted_name': 'nunique'
    }).reset_index()
    zone_ranges.columns = ['zone', 'min_age', 'max_age', 'mean_age', 'taxa_count']
    zone_ranges = zone_ranges.sort_values('mean_age', ascending=False).head(25)
    
    min_time_zones = zone_ranges['min_age'].min()
    max_time_zones = zone_ranges['max_age'].max()
    
    # Calculate taxon ranges first to get combined time range
    top_taxa = df['accepted_name'].value_counts().head(20).index
    taxon_ranges = []
    
    for taxon in top_taxa:
        taxon_data = df[df['accepted_name'] == taxon]
        taxon_ranges.append({
            'taxon': taxon,
            'first_appearance': taxon_data['mid_ma'].max(),
            'last_appearance': taxon_data['mid_ma'].min(),
            'mean_age': taxon_data['mid_ma'].mean(),
            'occurrences': len(taxon_data),
        })
    
    taxon_ranges_df = pd.DataFrame(taxon_ranges).sort_values('mean_age', ascending=False)
    min_time_taxa = taxon_ranges_df['last_appearance'].min()
    max_time_taxa = taxon_ranges_df['first_appearance'].max()
    
    # Use the same time range for both panels
    min_time = min(min_time_zones, min_time_taxa)
    max_time = max(max_time_zones, max_time_taxa)
    time_range = max_time - min_time
    label_offset = time_range * 0.05
    
    y_pos = 0
    for _, row in zone_ranges.iterrows():
        zone = row['zone']
        min_age = row['min_age']
        max_age = row['max_age']
        
        axes[0].plot([min_age, max_age], [y_pos, y_pos], 
                    color='black', linewidth=1.5, solid_capstyle='round')
        axes[0].plot(min_age, y_pos, marker='|', color='black', markersize=6, markeredgewidth=1.5)
        axes[0].plot(max_age, y_pos, marker='|', color='black', markersize=6, markeredgewidth=1.5)
        axes[0].text(min_age - label_offset, y_pos, f"{int(row['taxa_count'])} taxa", 
                    fontsize=6, va='center', ha='right')
        y_pos += 1
    
    axes[0].set_xlim([min_time - time_range * 0.20, max_time + time_range * 0.05])
    axes[0].set_yticks(range(len(zone_ranges)))
    axes[0].set_yticklabels([z[:35] + '...' if len(z) > 35 else z 
                             for z in zone_ranges['zone']], fontsize=7)
    axes[0].set_xlabel('Time (Ma)')
    axes[0].set_ylabel('Zone')
    axes[0].set_title('Biostratigraphic Zone Ranges Through Time')
    axes[0].invert_xaxis()
    
    # Bottom plot: Key taxon ranges across all zones
    y_pos = 0
    for _, row in taxon_ranges_df.iterrows():
        taxon = row['taxon']
        first = row['first_appearance']
        last = row['last_appearance']
        
        axes[1].plot([first, last], [y_pos, y_pos], 
                     color='black', linewidth=1.5, solid_capstyle='round')
        axes[1].plot(first, y_pos, marker='|', color='black', markersize=6, markeredgewidth=1.5)
        axes[1].plot(last, y_pos, marker='|', color='black', markersize=6, markeredgewidth=1.5)
        axes[1].text(last - label_offset, y_pos, f"{int(row['occurrences'])} occ", 
                    fontsize=6, va='center', ha='right')
        y_pos += 1
    
    axes[1].set_xlim([min_time - time_range * 0.20, max_time + time_range * 0.05])
    axes[1].set_yticks(range(len(taxon_ranges_df)))
    axes[1].set_yticklabels([t[:30] + '...' if len(t) > 30 else t 
                             for t in taxon_ranges_df['taxon']], fontsize=7)
    axes[1].set_xlabel('Time (Ma)')
    axes[1].set_ylabel('Taxon')
    axes[1].set_title('Key Taxon Ranges Through Time')
    axes[1].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def plot_zone_diversity(
    zone_diversity: pd.DataFrame,
    output_path: str = 'zone_diversity_patterns.png'
) -> None:
    """Create zone-based diversity pattern visualization.

    Args:
        zone_diversity: DataFrame with zone diversity metrics.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    zone_diversity_sorted = zone_diversity.sort_values('mean_age', ascending=False)
    
    axes[0].plot(zone_diversity_sorted['mean_age'], zone_diversity_sorted['unique_taxa'],
                  marker='o', color='black', linestyle='-', markersize=3, linewidth=0.5)
    axes[0].set_xlabel('Time (Ma)')
    axes[0].set_ylabel('Unique Taxa per Zone')
    axes[0].set_title('Taxonomic Diversity by Zone Through Time')
    axes[0].invert_xaxis()
    
    axes[1].bar(zone_diversity_sorted['mean_age'], zone_diversity_sorted['occurrences'],
                color='black', width=2, alpha=0.6)
    axes[1].set_xlabel('Time (Ma)')
    axes[1].set_ylabel('Occurrences per Zone')
    axes[1].set_title('Occurrence Count by Zone Through Time')
    axes[1].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def plot_geographic_map(df: pd.DataFrame, output_path: str = 'geographic_map.png') -> None:
    """Create geographic map showing occurrence locations.

    Args:
        df: DataFrame with occurrence data.
        output_path: Path to save the figure.
    """
    geo_df = df[['lng', 'lat', 'cc', 'mid_ma']].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by country
    country_colors = {'CN': 'black', 'KR': 'black', 'IN': 'black', 'KP': 'black', 'BT': 'black'}
    for country in geo_df['cc'].unique():
        country_data = geo_df[geo_df['cc'] == country]
        ax.scatter(country_data['lng'], country_data['lat'], 
                  c=country_colors.get(country, 'gray'), 
                  alpha=0.4, s=10, edgecolors='none', label=country)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic Distribution of Trilobite Occurrences')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def prepare_first_last_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate first and last appearances for each taxon.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        DataFrame with first and last appearances, and temporal ranges.
    """
    taxon_ranges = df.groupby('accepted_name').agg({
        'mid_ma': ['min', 'max', 'count']
    }).reset_index()
    taxon_ranges.columns = ['taxon', 'last_appearance', 'first_appearance', 'occurrences']
    taxon_ranges['duration'] = taxon_ranges['first_appearance'] - taxon_ranges['last_appearance']
    taxon_ranges = taxon_ranges.sort_values('first_appearance', ascending=False)
    
    return taxon_ranges


def plot_first_last_appearances(
    taxon_ranges: pd.DataFrame,
    output_path: str = 'first_last_appearances.png'
) -> None:
    """Create first-last appearance plot showing taxon ranges.

    Args:
        taxon_ranges: DataFrame with first and last appearance data.
        output_path: Path to save the figure.
    """
    # Select top taxa by occurrences for clarity
    top_taxa = taxon_ranges.sort_values('occurrences', ascending=False).head(50)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = 0
    for _, row in top_taxa.iterrows():
        taxon = row['taxon']
        first = row['first_appearance']
        last = row['last_appearance']
        
        ax.plot([first, last], [y_pos, y_pos], 
               color='black', linewidth=1.0, solid_capstyle='round')
        ax.plot(first, y_pos, marker='|', color='black', markersize=5, markeredgewidth=1.5)
        ax.plot(last, y_pos, marker='|', color='black', markersize=5, markeredgewidth=1.5)
        y_pos += 1
    
    ax.set_yticks(range(len(top_taxa)))
    ax.set_yticklabels([t[:30] + '...' if len(t) > 30 else t 
                        for t in top_taxa['taxon']], fontsize=7)
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Taxon')
    ax.set_title('First and Last Appearances of Top 50 Taxa')
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def calculate_turnover_rates(df: pd.DataFrame, bin_size: float = 1.0) -> pd.DataFrame:
    """Calculate origination and extinction rates per time bin.

    Args:
        df: DataFrame with occurrence data.
        bin_size: Size of time bins in millions of years.

    Returns:
        DataFrame with origination and extinction rates per bin.
    """
    df['time_bin'] = (df['mid_ma'] // bin_size) * bin_size
    
    # Get first and last appearances for each taxon
    taxon_ranges = df.groupby('accepted_name').agg({
        'mid_ma': ['min', 'max']
    })
    taxon_ranges.columns = ['last_appearance', 'first_appearance']
    
    # Bin the appearances
    taxon_ranges['first_bin'] = (taxon_ranges['first_appearance'] // bin_size) * bin_size
    taxon_ranges['last_bin'] = (taxon_ranges['last_appearance'] // bin_size) * bin_size
    
    # Count originations and extinctions per bin
    bins = sorted(df['time_bin'].unique())
    turnover_data = []
    
    for bin_age in bins:
        originations = len(taxon_ranges[taxon_ranges['first_bin'] == bin_age])
        extinctions = len(taxon_ranges[taxon_ranges['last_bin'] == bin_age])
        diversity = len(df[df['time_bin'] == bin_age]['accepted_name'].unique())
        
        turnover_data.append({
            'time_bin': bin_age,
            'originations': originations,
            'extinctions': extinctions,
            'diversity': diversity,
            'origination_rate': originations / bin_size if diversity > 0 else 0,
            'extinction_rate': extinctions / bin_size if diversity > 0 else 0,
        })
    
    turnover_df = pd.DataFrame(turnover_data).sort_values('time_bin', ascending=False)
    
    return turnover_df


def plot_turnover_rates(
    turnover_df: pd.DataFrame,
    output_path: str = 'turnover_rates.png'
) -> None:
    """Create turnover rate visualization.

    Args:
        turnover_df: DataFrame with turnover rate data.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(turnover_df['time_bin'], turnover_df['originations'],
                 marker='o', color='black', linestyle='-', markersize=3, linewidth=1, label='Originations')
    axes[0].plot(turnover_df['time_bin'], turnover_df['extinctions'],
                 marker='s', color='black', linestyle='--', markersize=3, linewidth=1, label='Extinctions')
    axes[0].set_xlabel('Time (Ma)')
    axes[0].set_ylabel('Count per Myr')
    axes[0].set_title('Origination and Extinction Counts Through Time')
    axes[0].invert_xaxis()
    axes[0].legend()
    
    axes[1].plot(turnover_df['time_bin'], turnover_df['origination_rate'],
                 marker='o', color='black', linestyle='-', markersize=3, linewidth=1, label='Origination Rate')
    axes[1].plot(turnover_df['time_bin'], turnover_df['extinction_rate'],
                 marker='s', color='black', linestyle='--', markersize=3, linewidth=1, label='Extinction Rate')
    axes[1].set_xlabel('Time (Ma)')
    axes[1].set_ylabel('Rate (per Myr)')
    axes[1].set_title('Origination and Extinction Rates Through Time')
    axes[1].invert_xaxis()
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def perform_rarefaction(df: pd.DataFrame, n_samples: int = 100, sample_size: int = 100) -> pd.DataFrame:
    """Perform rarefaction analysis to test robustness of diversity peak.

    Args:
        df: DataFrame with occurrence data.
        n_samples: Number of rarefaction iterations.
        sample_size: Number of occurrences to sample per iteration.

    Returns:
        DataFrame with rarefaction results.
    """
    df['time_bin'] = (df['mid_ma'] // 1) * 1
    
    bins = sorted(df['time_bin'].unique())
    rarefaction_data = []
    
    for bin_age in bins:
        bin_data = df[df['time_bin'] == bin_age]
        if len(bin_data) < sample_size:
            continue
        
        diversities = []
        for _ in range(n_samples):
            sample = bin_data.sample(n=sample_size, replace=False)
            diversity = sample['accepted_name'].nunique()
            diversities.append(diversity)
        
        rarefaction_data.append({
            'time_bin': bin_age,
            'mean_diversity': np.mean(diversities),
            'std_diversity': np.std(diversities),
            'min_diversity': np.min(diversities),
            'max_diversity': np.max(diversities),
            'raw_diversity': bin_data['accepted_name'].nunique(),
            'raw_occurrences': len(bin_data),
        })
    
    rarefaction_df = pd.DataFrame(rarefaction_data).sort_values('time_bin', ascending=False)
    
    return rarefaction_df


def plot_rarefaction(
    rarefaction_df: pd.DataFrame,
    output_path: str = 'rarefaction_analysis.png'
) -> None:
    """Create rarefaction analysis visualization.

    Args:
        rarefaction_df: DataFrame with rarefaction results.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rarefaction_df['time_bin'], rarefaction_df['raw_diversity'],
            marker='o', color='black', linestyle='-', markersize=4, linewidth=1.5, label='Raw Diversity')
    ax.plot(rarefaction_df['time_bin'], rarefaction_df['mean_diversity'],
            marker='s', color='black', linestyle='--', markersize=3, linewidth=1, label='Rarefied Diversity (n=100)')
    ax.fill_between(rarefaction_df['time_bin'],
                    rarefaction_df['mean_diversity'] - rarefaction_df['std_diversity'],
                    rarefaction_df['mean_diversity'] + rarefaction_df['std_diversity'],
                    alpha=0.2, color='black')
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Diversity')
    ax.set_title('Rarefaction Analysis: Raw vs. Standardized Diversity')
    ax.invert_xaxis()
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def plot_alpha_diversity_through_time(df: pd.DataFrame, output_path: str = 'alpha_diversity_through_time.png') -> None:
    """Create alpha diversity through time plot.

    Args:
        df: DataFrame with occurrence data.
        output_path: Path to save the figure.
    """
    df['time_bin'] = (df['mid_ma'] // 1) * 1
    alpha_diversity = df.groupby('time_bin').agg({
        'accepted_name': 'nunique',
        'occurrence_no': 'count'
    }).rename(columns={'accepted_name': 'unique_taxa', 'occurrence_no': 'occurrences'})
    alpha_diversity = alpha_diversity.sort_index(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(alpha_diversity.index, alpha_diversity['unique_taxa'],
            marker='o', color='black', linestyle='-', markersize=3, linewidth=1)
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Alpha Diversity (Unique Taxa)')
    ax.set_title('Trilobite Alpha Diversity Through Time')
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {output_path}")


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset.

    Args:
        df: DataFrame with occurrence data.

    Returns:
        Dictionary with summary statistics.
    """
    return {
        'total_occurrences': len(df),
        'unique_taxa': df['accepted_name'].nunique(),
        'time_span_myr': df['max_ma'].max() - df['min_ma'].min(),
        'oldest_ma': df['max_ma'].max(),
        'youngest_ma': df['min_ma'].min(),
        'countries': df['cc'].nunique(),
        'genera': df['accepted_name'].str.split().str[0].nunique(),
    }


def main() -> None:
    """Run comparative biology analysis on trilobite data."""
    df = load_data('pbdb_data (1).csv')
    
    rank_counts, top_genera = prepare_taxonomic_data(df)
    plot_taxonomic_diversity(rank_counts, top_genera)
    
    temporal_diversity = prepare_temporal_data(df)
    plot_temporal_patterns(temporal_diversity)
    
    country_counts, geo_df = prepare_geographic_data(df)
    plot_geographic_distribution(country_counts, geo_df)
    
    comparison_df = prepare_top_taxa_comparison(df, n=10)
    plot_top_taxa_comparison(comparison_df)
    
    env_counts, lith_counts = prepare_environmental_data(df)
    plot_environmental_preferences(env_counts, lith_counts)
    
    feature_df, feature_matrix, taxon_labels = prepare_clustering_features(df, n=50)
    plot_dendrogram(feature_matrix, taxon_labels)
    
    formation_df, zone_df = prepare_formation_zone_data(df)
    plot_formation_zone_relationships(formation_df, zone_df)
    
    biostrat_df = prepare_biostratigraphic_ranges(df)
    plot_biostratigraphic_ranges(biostrat_df, df)
    
    zone_diversity = prepare_zone_diversity(df)
    plot_zone_diversity(zone_diversity)
    
    plot_geographic_map(df)
    plot_alpha_diversity_through_time(df)
    
    taxon_ranges = prepare_first_last_appearances(df)
    plot_first_last_appearances(taxon_ranges)
    
    turnover_df = calculate_turnover_rates(df, bin_size=1.0)
    plot_turnover_rates(turnover_df)
    
    rarefaction_df = perform_rarefaction(df, n_samples=100, sample_size=100)
    plot_rarefaction(rarefaction_df)
    
    stats = generate_summary_statistics(df)
    logger.info("Analysis complete")
    logger.info(f"Summary: {stats['total_occurrences']} occurrences, "
                f"{stats['unique_taxa']} taxa, "
                f"{stats['time_span_myr']:.1f} Myr span")


if __name__ == "__main__":
    main()
