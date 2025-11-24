# Supplemental Materials

## Code Files

### trilobite_comparative_analysis.py
Main analysis script containing all functions for:
- Data loading and cleaning
- Taxonomic diversity analysis
- Temporal pattern analysis
- Geographic distribution analysis
- Formation and zone relationship analysis
- Biostratigraphic range calculations
- First-last appearance analysis
- Turnover rate calculations
- Rarefaction analysis
- Hierarchical clustering
- All visualization functions

### requirements.txt
Python package dependencies required to run the analysis code.

## Data Provenance

The raw occurrence data are available from the Paleobiology Database (https://paleobiodb.org).

PBDB Query Parameters:
- base_id: tid:19100 (Trilobita)
- lngmin: 76.4209
- lngmax: 150.0293
- latmin: 17.0148
- latmax: 42.0656
- interval_id: 22 (Cambrian)
- show: coords,attr,loc,prot,time,strat,stratext,lith,lithext,geo,rem,ent,entname,crmod,paleoloc

## Reproduction Instructions

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Download data from PBDB using the query parameters above, or use the provided CSV file.

3. Run the analysis:
   ```bash
   python trilobite_comparative_analysis.py
   ```

4. All figures will be generated in the current directory.

## Code Sections

### Rarefaction Code
The rarefaction analysis is implemented in the `perform_rarefaction()` and `plot_rarefaction()` functions (lines ~890-960).

### Turnover Calculations
Turnover rate calculations are in `calculate_turnover_rates()` and `plot_turnover_rates()` functions (lines ~810-870).

### Clustering Scripts
Hierarchical clustering is implemented in `prepare_clustering_features()` and `plot_dendrogram()` functions (lines ~360-425).

