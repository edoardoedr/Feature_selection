# FeSCAE - Feature Selection with Clustering and AutoEncoders

Framework per la selezione di feature da dati di espressione genica utilizzando clustering e autoencoder.

## 📋 Indice

- [Installazione](#installazione)
- [Utilizzo Base](#utilizzo-base)
- [Configurazione](#configurazione)
- [Dataset Supportati](#dataset-supportati)
- [Output](#output)

## 🚀 Installazione

1. Clona il repository e naviga nella directory:
```bash
cd genomics
```

2. Crea l'ambiente conda con le dipendenze:
```bash
conda env create -f environment.yml
conda activate genomics_env
```

## 💻 Utilizzo Base

### Esecuzione Rapida

Per eseguire l'analisi completa con le configurazioni di default:

```bash
python main.py
```

Il programma eseguirà automaticamente:
1. **Caricamento dati**: Preparazione del dataset configurato
2. **Clustering**: Raggruppamento delle features basato su correlazione o distanza
3. **Filtering**: Selezione delle feature più rilevanti tramite Autoencoder o Medoid

### Flusso di Esecuzione

```python
# 1. Caricamento configurazione
config = Config("config_yaml/base_config.yaml")

# 2. Preparazione dati
data_preparation = DataPreparationFactory(
    dataset_name=config.dataset.name,
    input_folder=config.dataset.input_folder,
    output_folder=config.output.output_folder,
    label_column=config.dataset.label_column
)
data = data_preparation.export_data()

# 3. Clustering delle features
clustering = GetClustering(
    data=data['features_data'],
    cluster_strategy=config.clustering.strategy,
    number_of_clusters=config.clustering.number_of_clusters
)
clusters = clustering.get_clusters()

# 4. Filtraggio e selezione features
FeatureFilter = create_filter(config.filters, iteration, output_folder, scaler)
selected_features = FeatureFilter.filter(features_data, clusters)
```

## ⚙️ Configurazione

Il file di configurazione principale si trova in `config_yaml/base_config.yaml`. Ecco una spiegazione dettagliata dei parametri:

### Dataset

```yaml
dataset:
  name: "Leukemia"                          # Nome del dataset da utilizzare
  input_folder: "Raw_Datasets/Leukemia_ttft"  # Cartella con i dati raw
  label_column: "TTFT"                      # Nome della colonna con le label
  scaler: "standard"                        # Tipo di normalizzazione (standard, minmax, robust)
```

**Parametri:**
- `name`: Nome del preprocessing da utilizzare (vedi [Dataset Supportati](#dataset-supportati))
- `input_folder`: Path alla cartella contenente i file del dataset
- `label_column`: Nome della colonna contenente le label binarie (0/1)
- `scaler`: Tipo di normalizzazione delle feature
  - `standard`: StandardScaler (media 0, deviazione standard 1)
  - `minmax`: MinMaxScaler (range [0,1])
  - `robust`: RobustScaler (robusto agli outlier)

### Output

```yaml
output:
  output_folder: "prova_output"  # Cartella per salvare i risultati
  logs_folder: "prova_output"    # Cartella per i log
```

### Clustering

```yaml
clustering:
  strategy: "KMeans"              # Algoritmo di clustering
  cluster_on_correlation: True    # Clustering basato su correlazione
  number_of_clusters: 20          # Numero di cluster desiderato
  parameters: "searching"         # "searching" o dizionario parametri
```

**Strategie disponibili:**
- `KMeans`: K-Means clustering
- `KMedoids`: K-Medoids clustering
- `Hierarchical`: Clustering gerarchico
- `DBSCAN`: Density-based clustering

**Parametri:**
- `cluster_on_correlation`: 
  - `True`: raggruppa features correlate tra loro
  - `False`: raggruppa features per distanza nello spazio dei campioni
- `parameters`: 
  - `"searching"`: ricerca automatica dei parametri ottimali
  - Dizionario con parametri specifici (es. `{distance_metric: "euclidean"}`)

### Filtri

```yaml
filters:
  filter_type: 'AE'  # Tipo di filtro: 'AE' (AutoEncoder) o 'Medoid'
  
  # Parametri per AutoEncoder
  autoencoder_parameters:
    lr: 0.001          # Learning rate
    epochs: 100        # Numero di epoche di training
    AE_name: 'AE'  # Tipo di autoencoder
  
  # Parametri per Medoid
  medoid_parameters:
    distance_metric: "euclidean"  # Metrica di distanza
```

**Tipi di filtro:**
- `AE`: Utilizza un autoencoder per selezionare features rappresentative da ogni cluster
- `Medoid`: Seleziona il medoid (elemento più centrale) di ogni cluster

Per il filtro `AE` è inoltre possibile effettuare la ricerca degli iperparametri dei
layer dell'autoencoder, così da individuare la configurazione dell'architettura più
adatta al dataset.

## 📊 Dataset Supportati

Il framework include preprocessing già implementati per i seguenti dataset:

### 1. Leukemia (CLL - Leucemia Linfocitica Cronica)

**Dataset**: [Leukemia Gene Expression CuMiDa](https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida/data)

**Configurazione:**
```yaml
dataset:
  name: "Leukemia"
  input_folder: "Raw_Datasets/Leukemia_ttft"
  label_column: "TTFT"
  scaler: "standard"
```

**Descrizione:**
- Dati di espressione genica per leucemia linfocitica cronica
- Target: predizione del tempo al primo trattamento (TTFT)
- Features: migliaia di geni espressi
- Label binarie: necessità di trattamento precoce (1) o tardivo (0)

**Struttura file richiesti nella cartella input:**
- `data_merged.csv`: file con dati di espressione genica e label
- Colonna `cod_pz`: identificativo paziente
- Colonna `TTFT` (o altra specificata): label binarie

### 2. Leukemia ALL/AML

**Configurazione:**
```yaml
dataset:
  name: "leukemia_ALL_AML"
  input_folder: "Raw_Datasets/Leukemia_ALL_AML_data"
  label_column: "cancer"
  scaler: "standard"
```

**Descrizione:**
- Dataset classico per classificazione ALL (Leucemia Linfoblastica Acuta) vs AML (Leucemia Mieloide Acuta)
- Features: espressione di 7129 geni
- Label: tipo di leucemia (0=ALL, 1=AML)

**Struttura file richiesti:**
- File con dati di training/test separati
- Preprocessing automatico per merge e preparazione

### Aggiungere Nuovi Dataset

Per aggiungere supporto a un nuovo dataset:

1. Creare una nuova classe in `FeSCAE/DataPreparation/` (es. `MyDatasetPreparation.py`)
2. Estendere `BaseDataPreparation`
3. Implementare il metodo `process_data()`
4. Aggiungere il nuovo dataset nella factory (`DataPreparationFactory.py`)

## 📁 Output

Dopo l'esecuzione, la cartella di output conterrà:

```
prova_output/
├── data.csv                    # Dataset completo (features + label)
├── features_data.csv           # Solo le features
├── features.csv                # Nome delle features
├── labels.csv                  # Label dei campioni
├── clusters.csv                # Assegnazione features ai cluster
├── dataset_info.txt            # Informazioni sul dataset
├── featuresel_log.txt          # Log del processo di selezione
└── output_iteration_0/         # Risultati per iterazione
    └── filter_log.txt          # Log dettagliato del filtro
```

### File Principali

- **data.csv**: Dataset completo con colonna identificativa, features e label
- **features_data.csv**: Matrice features (righe=campioni, colonne=geni)
- **clusters.csv**: Assegnazione di ogni feature al cluster di appartenenza
- **featuresel_log.txt**: Log del processo con metriche e dettagli

## 🔄 Workflow Completo

1. **Preparazione**: Il sistema carica/preprocessa i dati del dataset specificato
2. **Clustering**: Le features vengono raggruppate in cluster basati su correlazione/distanza
3. **Filtering**: Da ogni cluster viene selezionata la feature più rappresentativa
4. **Output**: Le feature selezionate e i risultati vengono salvati

## 📝 Note

- I dati preprocessati vengono salvati automaticamente: alle esecuzioni successive verranno ricaricati direttamente
- Il sistema supporta ricerca automatica dei parametri ottimali di clustering
- Per gli autoencoder è disponibile anche la ricerca degli iperparametri dei layer dell'architettura
- Gli autoencoder utilizzati sono personalizzabili (Linear, Convolutional, ecc.)
- Il framework è estendibile con nuovi algoritmi di clustering e filtri

## 🛠️ Troubleshooting

**Problema**: "Dataset sconosciuto"
- Verifica che il nome del dataset nel config corrisponda a uno supportato
- Controlla che i file di input siano nella cartella specificata

**Problema**: "I label devono essere 0 o 1"
- Assicurati che la colonna label contenga solo valori binari
- Potrebbe essere necessario preprocessare le label prima

**Problema**: Errori di shape nei dati
- Verifica che i dati non contengano valori mancanti
- Controlla che tutte le features siano numeriche
