# Stratigraphic Patterns of Cambrian Trilobite Diversity in Eastern Asia

This repository contains the complete submission package for the manuscript "Stratigraphic Patterns of Cambrian Trilobite Diversity in Eastern Asia: A Middle Cambrian Diversity Peak" submitted to the Journal of Asian Earth Sciences.

## Repository Structure

```
jaes-trilobite-diversity-paper/
├── manuscript/
│   ├── main.tex          # Main LaTeX manuscript
│   ├── main.pdf          # Compiled manuscript PDF
│   └── references.bib    # Bibliography in BibTeX format
├── figures/
│   ├── Fig1.png through Fig9.png  # All manuscript figures
├── supplement/
│   ├── trilobite_comparative_analysis.py  # Analysis code
│   ├── requirements.txt                   # Python dependencies
│   ├── diversity_timeseries.csv           # Time-binned diversity data
│   └── README.md                         # Supplemental materials documentation
├── metadata/
│   ├── cover_letter.tex/.pdf             # Submission cover letter
│   ├── highlights.tex/.pdf               # Journal highlights
│   └── [other metadata files]
└── README.md                             # This file
```

## Data Source

The raw occurrence data analyzed in this study are available from the Paleobiology Database (https://paleobiodb.org). The specific dataset used in this analysis can be accessed via PBDB query parameters provided in the Methods section of the manuscript.

## Reproducing the Analysis

1. Install Python dependencies:
   ```bash
   pip install -r supplement/requirements.txt
   ```

2. Download data from PBDB or use the provided CSV file.

3. Run the analysis:
   ```bash
   python supplement/trilobite_comparative_analysis.py
   ```

4. All figures will be generated in the current directory.

## Compiling the Manuscript

To compile the LaTeX manuscript:

```bash
cd manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

If you use this work, please cite:

Jones, K.T. (submitted). Stratigraphic Patterns of Cambrian Trilobite Diversity in Eastern Asia: A Middle Cambrian Diversity Peak. Journal of Asian Earth Sciences.

## License

[Add your license here]

## Contact

Kyle T. Jones  
American University  
kylejones@american.edu
