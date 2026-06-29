CPG v1 — Cellular Performance Gauge — complete run folder
==========================================================

This folder contains everything the chain needs to read a patient IDAT and
produce a report, EXCEPT the atlas file (which you add — see step 1). Keep the
folder structure exactly as it is; the conductor finds every file by its
relative path and auto-detects this folder as the root.


------------------------------------------------------------------
1. ADD THE ATLAS (the one file not included)
------------------------------------------------------------------
Copy your atlas into the IAM_Atlas folder:

    IAM_Atlas/IAMAtlasREBUILD.csv

(If you only have IAMAtlasREBUILD.csv.xz, put that there instead — it will be
decompressed once on the first run.) Nothing else needs to be placed anywhere.


------------------------------------------------------------------
2. ONE-TIME SETUP ON A NEW MACHINE
------------------------------------------------------------------
Install Anaconda, then make one dedicated environment for the chain:

    conda create -n cpg python=3.11 -y
    conda activate cpg
    pip install methylprep numpy pandas scipy scikit-learn matplotlib

That is all the chain depends on. Use this same environment every time.
(If methylprep is missing, the chain will try to install it automatically, but
installing it cleanly up front avoids any dependency conflicts on a fresh PC.)


------------------------------------------------------------------
3. RUN A PATIENT
------------------------------------------------------------------
a) Fill in the intake. Either:
   - open  cpg_intake_form.html  in a browser, complete it, and save
     questionnaire.json into this folder; OR
   - just run the chain (step b) with no questionnaire.json present and answer
     the four prompts at the terminal — it writes the file for you.

b) Put the patient's IDAT pair in this folder:
       <sample>_Grn.idat   and   <sample>_Red.idat

c) From this folder, with the cpg environment active:
       python walther_clinical.py

The report is written here as  CPG_report_<patient>_<timestamp>.html.

Stage 1 calibrates the IDAT per-sample (dye-bias + probe-type normalization /
noob) using only that patient's own probes — no cohort, no reference, no
manifest needed. 450k vs EPIC is auto-detected.


------------------------------------------------------------------
4. WHAT THE CHAIN READS FROM THE INTAKE
------------------------------------------------------------------
Only four fields: patient_id, age, sex, substrate (plus optional family_history).
Every questionnaire field is report context and NEVER enters the score — there
is a hard firewall. The full intake is kept under an "intake" key for the record.


------------------------------------------------------------------
5. ADDING THE SECOND CHAIN LATER (Mahalanobis)
------------------------------------------------------------------
The second chain (global Mahalanobis departure + per-disease residual maps)
runs only when the disease matrix flags something. When we wire it in, the new
files go under:

    Runtime Matrices/Mahalanobis_healthy_reference/
    Disease Cards : Residual Maps/

You will be told the exact filenames and folders at that time; just drop them in
and the conductor picks them up. Nothing in this current folder changes.


------------------------------------------------------------------
FOLDER MAP (do not rename or move)
------------------------------------------------------------------
walther_clinical.py            <- run this
cpg_report_builder.py
stage_1_idat_calibration.py    <- Stage 1 per-sample noob calibration
cpg_intake_form.html           <- office intake form (makes questionnaire.json)

IAM_Atlas/
  IAMAtlasREBUILD.csv          <- YOU ADD THIS
  IAMAtlasREBUILD_celltype_to_class.json
  IAMAtlasREBUILD_provenance.json
  iamatlas_class_archives/     <- 8 per-class brightness archives (report CIs)

Runtime Matrices/
  A_Scoring_Module/iamatlas_a_scoring.py
  A_Scoring_Module/test_a_score_canonical.py   <- startup safety gate
  Celltype_Marker/iamatlas_celltype_markers_v0_2.json
  Tier_breakpoints/tier_breakpoints.json

Walther_iam_deconvolver/walther_iam_deconvolver.py
NILC Deconvolver/nilc_deconvolver-2.py
CPG_Report_Generator/cpg_gauge.py

Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_8.csv
Disease Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json
