### Environment
Python environment.
smartSPIM processing pipeline completes Registration: creates transforms and Cell Detection and Classification: After registration but is not dependent on the output from registration.
From then on, this capsule performs Quantification: Brings together the Registration transforms with classification output to place cells into CCF space and count cells by region. We specifically isolate PONS and PONS adjacent cells as additional post-hoc processing for automated soma segmentation in Dbh-Cre;Ai65 animals where tdTomato labels soma and proximal dendrites.

### Notes

GitHub: https://github.com/AllenNeuralDynamics/LC-NE_Register_Annotations_retrograde_cells
Code Ocean: TBD