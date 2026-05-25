# Volvo pipeline
### Start with:
- Dataset/G2AAS_performance_december.csv
- Dataset/G2AAS_performance_november.csv

### Run:
1. preprocessing.ipynb
2. KMeans Windowed.ipynb
3. Clustering_samples_sorting.ipynb
4. postprocessing.ipynb

### End up with:
- Dataset/clustering.csv (clustered windows)
- Dataset/clustering_samples.csv
- Clustering/clustering_samples.xlsx
- Clustering/clustering_samples_with_clusters.xlsx
- Dataset/Processed Dataset/volvoTrain.csv
- Dataset/Processed Dataset/volvoTest.csv

### From models:
1. Call data_loader.load_volvo()
2. Call data_loader.load_optuna_val()
3. Call hyperparameter_search.run_optuna_study()
4. Call evaluation.cross_validate()
5. Fit model
6. Call evaluation.evaluate_model()
7. Save scores as ./results/Volvo {model}/scores.csv

# HAI pipeline
### No preprocessing standalone files

### From models:
1. Call hyperparameter_search.run_optuna_study_hai()
    1. Calls data_loader_hai.load_hai_optuna()
        1. Calls data_loader_hai.load_hai_data()
2. Call evaluation_hai.cross_validate_hai()
    1. Calls data_loader_hai.load_hai_data()
3. Call data_loader_hai.load_hai_train()
    1. Calls data_loader_hai.load_hai_data()
4. Fit model
5. Call evaluation.evaluate_model()
6. Save scores as ./results/HAI {model}/scores.csv
