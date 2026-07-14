# LC-NE_Register_Annotations_retrograde_cells

Code Ocean capsule for extracting manually curated cell annotations from a neuroglancer JSON, registering to CCF space, and identifying annotions that exist within the Pons. 

**GitHub:** https://github.com/AllenNeuralDynamics/LC-NE_Register_Annotations_retrograde_cells
**Code Ocean:** https://codeocean.allenneuraldynamics.org/capsule/1936843

---

## 1. Pipeline overview

This capsule runs a single script 'pipeline.py' (see [run](run)) that identifies which point annotations from a nuroglancer JSON exist with in the Pons, using a signed distance algorithm. The points are then registered into CCF space and tagged as whether they are inside, outside, or bordering the Pons. Outputs from this script are written to '/results/:

the reproducible run is driven by [run](run):

```bash
set -ex
python -u pipeline.py
```

## 2. Input data assets

- '/data/LC-NE-manual-annotations_2026-07-09_00-03-36/': Per-brain cell locations. Annotations are automatically detected through internal processing pipeline and manually proofread to remove false positives and false negatives 
- '/data/SmartSPIM-template_2024-05-16_11-26-14/': static transforms for moving points from the SmartSPIM template space to CCF space 
- '/data/SmartSPIM_{LabTracks_ID}_{imaging_date}_stiched_{stitched_date}/: Per-brain data folder containing dataset specific registration transforms for moving points from raw imaging space to SmartPIM template space  

## 3. Output ('/results/')

- 'registered/': folder containing annotations taken from the manually annotated layer of the neuroglancer JSON registered into CCF space and saved to a CSV. Each dataset has its own file {LabTracks_ID}_ccf.csv
- 'raw_space/': folder containing annotations taken from the manually annotated layer of the neuroglancer JSON in raw imaging space with identification of location relative to the Pons
- 'final_results/': folder containing annotations taken from the manually annotated layer of the neuroglancer JSON in CCF space with identification of location relative to the Pons

## 4. Coordinate conventions

- Per-Brain locations in raw image space ('/results/raw_space/') are identified using columns 'Z', 'Y', 'X' in agreement with imaging axes identified within neuroglancer visualization

- Per-Brain locations within CCF space are identified using columns 'ML', 'DV', 'AP' in agreement with the Allen Brain Atlas orientation

- Per-Brain CSVs that include Pons localization information include a 'Location' column. 
    - inside: cells that are located within the Pons identified using signed distance
    - outside: cells that are located outside the Pons identified using signed distance
    - border: cells that are not located within the Pons identified using signed distance, but are within a thresheld radius from the regions boundary (see pipeline.py params)

### Environment
Python environment.
SmartSPIM processing pipeline completes Registration: creates transforms and Cell Detection and Classification: After registration but is not dependent on the output from registration.
From then on, this capsule performs Quantification: Brings together the Registration transforms with classification output to place cells into CCF space and count cells by region. We specifically isolate PONS and PONS adjacent cells as additional post-hoc processing for automated soma segmentation in Dbh-Cre;Ai65 animals where tdTomato labels soma and proximal dendrites.
