 -------------------------------------------------------------------------------------------------------------------------------------
| Originally located at EHR-Clustering-RA/notebooks/ | 
 -------------------------------------------------------------------------------------------------------------------------------------
 
 - additional_files/
    This directory stores additional code, which might be necessary to run in some but not all cases:
        - to either compute extra DAS (in case DAS is missing)
        - to differentiate between joint assesement by research nurse vs Physician
        - to add Symptom duration (based on physician's annotation)
 
- 1_patient_selection.ipynb (Jupyter Notebook)
    This notebook is tasked with identifying the first visit date and extracting the physician’s 
    verdict from the free text by employing a dedicated ML-model.
    
- 2_extract_features.ipynb (Jupyter Notebook)
    This notebook is tasked with truncating many entries to a single row that comprises the 
    clinical details of a patient at baseline. It is the first step in the preprocessing procedure  

- 3_process_lab.ipynb (Jupyter Notebook)
    This notebook is tasked with preprocessing the Lab data 
    
- 4_process_mannequin.ipynb (Jupyter Notebook)
    This notebook is tasked with preprocessing the Mannequin data 
    
- 5_clustering.ipynb (Jupyter Notebook)
    This notebook performs PhenoGraph graph clustering on the latent space created with MMAE and 
    creates a metadata table. Several steps are taken to define the ideal number of clusters and to 
    evaluate the robustness of the clustering technique.
    
- 6_process_medication.ipynb (Jupyter Notebook)
    This notebook processes the medication information and generates a baseline table.
    
- 7_survival_analysis.ipynb (Jupyter Notebook)
    This notebook performs survival analysis to assess the treatment response of the different clusters
    
- 8_replication_setB_IMPROVED.ipynb (Jupyter Notebook)
    This notebook is tasked w/ replicating our findings in historic trial data. It performs clustering
    on an independent replication set, namely: IMPROVED patients, that were included between 2007-2010.
    
- 8_replication_setC_RZWN.ipynb (Jupyter Notebook)
    This notebook is tasked w/ replicating our findings in external hospital data. It performs clustering
    on patients that visited any of the 9 hospitals affilitated with Reumazorg zuid west nederland (included between 2015-2022). 
    
- 9_Treatment_analysis.ipynb (Jupyter Notebook)
    This notebook is tasked with visualizing the treatment exposure and DMARD failure / successes 
    across patients. As well as studying some potential confounder, such as treating physician.
    
- 10_build_MTX_predictor.rmd (R Markdown)
    This R-notebook is tasked with building an predictor for MTX treatment failure. Furthermore, it 
    also creates some nice visualizations for downstream analysis, like checking for 
    potential confounders (treating physician) as well as showing an overview of the missingness.
    
- 11_Downstream_histopathological_analysis.ipynb (Jupyter Notebook)
    This notebook is tasked w/ conducting a downstream analysis to examine if clusters correspond to different histopathological
    features. This set includes patients from SYNGem Biopsy Unit cohort of the Fondazione Policlinico Universitario A.
    Gemelli IRCCS–Università Cattolica del Sacro Cuore – Rome - Italy,
