"""
Script per il parsing dei risultati degli esperimenti FeSCAE.
Cerca ricorsivamente tutti i file 'featuresel_log.txt' e crea un CSV summary.
"""

import ast
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_timestamp(text):
    """Converte una stringa timestamp nel formato usato nei log."""
    if not text:
        return None

    try:
        return datetime.strptime(text, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _extract_line_timestamp(line):
    """Estrae il timestamp da una riga del log, se presente."""
    match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
    if match:
        return _parse_timestamp(match.group(1))

    match = re.search(r'Esecuzione iniziata:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if match:
        return _parse_timestamp(match.group(1))

    return None


def _format_seconds(delta):
    """Restituisce la durata in secondi, arrotondata al secondo più vicino."""
    if delta is None:
        return None

    return round(delta.total_seconds(), 3)


def parse_log_file(log_path):
    """
    Estrae le informazioni rilevanti da un file featuresel_log.txt.
    
    Args:
        log_path: Path al file di log
        
    Returns:
        dict: Dizionario con le informazioni estratte
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    content = ''.join(lines)

    timestamps = []
    pending_timestamp = None
    execution_start = None
    load_start = None
    clustering_start = None
    filtering_start = None
    evaluation_start = None
    
    result = {
        'log_path': str(log_path),
        'dataset': None,
        'clustering_strategy': None,
        'number_of_clusters': None,
        'num_selected_features': None,
        'selected_features': None,
        'nmi_mean': None,
        'nmi_std': None,
        'acc_mean': None,
        'acc_std': None,
        'load_data_duration_sec': None,
        'clustering_duration_sec': None,
        'filtering_duration_sec': None,
        'evaluation_duration_sec': None,
        'total_duration_sec': None
    }

    for line in lines:
        line_timestamp = _extract_line_timestamp(line)
        if line_timestamp is not None:
            timestamps.append(line_timestamp)
            pending_timestamp = line_timestamp

        if 'Esecuzione iniziata:' in line and execution_start is None:
            execution_start = line_timestamp
        elif 'CARICAMENTO DATI' in line and load_start is None:
            load_start = pending_timestamp
        elif 'CLUSTERING...' in line and clustering_start is None:
            clustering_start = pending_timestamp
        elif 'FILTERING...' in line and filtering_start is None:
            filtering_start = pending_timestamp
        elif 'VALUTAZIONE...' in line and evaluation_start is None:
            evaluation_start = pending_timestamp
    
    # Estrai dataset
    dataset_match = re.search(r'Dataset:\s*(\S+)', content)
    if dataset_match:
        result['dataset'] = dataset_match.group(1)
    
    # Estrai strategia di clustering
    strategy_match = re.search(r'Clustering strategy:\s*(\S+)', content)
    if strategy_match:
        result['clustering_strategy'] = strategy_match.group(1)
    
    # Estrai numero di cluster dal path (più affidabile)
    path_parts = str(log_path).split('/')
    for part in path_parts:
        # Cerca pattern tipo: output_colon_10_hierarchical
        match = re.search(r'_(\d+)_(hierarchical|kmeans)', part, re.IGNORECASE)
        if match:
            result['number_of_clusters'] = int(match.group(1))
            break
    
    # Estrai features selezionate
    features_match = re.search(r'Selected features:\s*(\[.*?\])', content)
    if features_match:
        features_str = features_match.group(1)
        # Converti la stringa in lista
        features_list = ast.literal_eval(features_str)
        result['num_selected_features'] = len(features_list)
        result['selected_features'] = ', '.join(features_list)
    
    # Estrai NMI (formato: NMI: 0.0021 ± 0.0000)
    nmi_match = re.search(r'NMI:\s*([\d.]+)\s*±\s*([\d.]+)', content)
    if nmi_match:
        result['nmi_mean'] = float(nmi_match.group(1))
        result['nmi_std'] = float(nmi_match.group(2))
    
    # Estrai ACC (formato: ACC: 0.5484 ± 0.0000)
    acc_match = re.search(r'ACC:\s*([\d.]+)\s*±\s*([\d.]+)', content)
    if acc_match:
        result['acc_mean'] = float(acc_match.group(1))
        result['acc_std'] = float(acc_match.group(2))

    # Calcola i tempi di esecuzione dai timestamp del log
    last_timestamp = timestamps[-1] if timestamps else None

    if load_start and clustering_start:
        result['load_data_duration_sec'] = _format_seconds(clustering_start - load_start)

    if clustering_start and filtering_start:
        result['clustering_duration_sec'] = _format_seconds(filtering_start - clustering_start)

    if filtering_start and evaluation_start:
        result['filtering_duration_sec'] = _format_seconds(evaluation_start - filtering_start)

    if evaluation_start and last_timestamp:
        evaluation_duration = last_timestamp - evaluation_start
        if evaluation_duration.total_seconds() >= 0:
            result['evaluation_duration_sec'] = _format_seconds(evaluation_duration)

    if execution_start and last_timestamp:
        total_duration = last_timestamp - execution_start
        if total_duration.total_seconds() >= 0:
            result['total_duration_sec'] = _format_seconds(total_duration)
    
    return result


def find_log_files(root_folder):
    """
    Trova ricorsivamente tutti i file 'featuresel_log.txt' nella cartella e sottocartelle.
    
    Args:
        root_folder: Cartella root da cui iniziare la ricerca
        
    Returns:
        list: Lista di Path ai file di log trovati
    """
    root_path = Path(root_folder)
    log_files = list(root_path.rglob('featuresel_log.txt'))
    return log_files


def parse_all_logs(root_folder, output_csv='results_summary.csv'):
    """
    Parsa tutti i log nella cartella e crea un CSV summary.
    
    Args:
        root_folder: Cartella root da cui iniziare la ricerca
        output_csv: Nome del file CSV di output
    """
    print(f"Ricerca dei file 'featuresel_log.txt' in '{root_folder}'...")
    
    log_files = find_log_files(root_folder)
    
    if not log_files:
        print(f"⚠ Nessun file 'featuresel_log.txt' trovato in '{root_folder}'")
        return
    
    print(f"✓ Trovati {len(log_files)} file di log\n")
    
    results = []
    for i, log_path in enumerate(log_files, 1):
        print(f"[{i}/{len(log_files)}] Parsing: {log_path.relative_to(root_folder)}")
        try:
            result = parse_log_file(log_path)
            results.append(result)
        except Exception as e:
            print(f"  ⚠ Errore nel parsing: {e}")
    
    if not results:
        print("\n⚠ Nessun risultato estratto")
        return
    
    # Crea DataFrame e ordina
    df = pd.DataFrame(results)
    
    # Ordina per dataset, numero cluster e strategia
    sort_columns = ['dataset', 'number_of_clusters', 'clustering_strategy']
    sort_columns = [col for col in sort_columns if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    
    # Salva CSV
    df.to_csv(output_csv, index=False)
    df.to_excel(output_csv.replace('.csv', '.xlsx'), index=False)  # Salva anche in Excel
    print(f"\n✓ CSV salvato in '{output_csv}'")
    
    # Stampa statistiche
    print(f"\n{'='*80}")
    print("RIEPILOGO")
    print(f"{'='*80}")
    print(f"Totale esperimenti: {len(df)}")
    
    if 'dataset' in df.columns:
        print(f"\nDataset:")
        for dataset, count in df['dataset'].value_counts().items():
            print(f"  - {dataset}: {count} esperimenti")
    
    if 'clustering_strategy' in df.columns:
        print(f"\nStrategie di clustering:")
        for strategy, count in df['clustering_strategy'].value_counts().items():
            print(f"  - {strategy}: {count} esperimenti")
    
    if 'number_of_clusters' in df.columns:
        print(f"\nNumero di cluster:")
        print(f"  - Min: {df['number_of_clusters'].min()}")
        print(f"  - Max: {df['number_of_clusters'].max()}")
        print(f"  - Valori unici: {sorted(df['number_of_clusters'].dropna().unique())}")
    
    if 'nmi_mean' in df.columns:
        print(f"\nNMI medio:")
        print(f"  - Min: {df['nmi_mean'].min():.4f}")
        print(f"  - Max: {df['nmi_mean'].max():.4f}")
        print(f"  - Media: {df['nmi_mean'].mean():.4f}")
    
    if 'acc_mean' in df.columns:
        print(f"\nACC medio:")
        print(f"  - Min: {df['acc_mean'].min():.4f}")
        print(f"  - Max: {df['acc_mean'].max():.4f}")
        print(f"  - Media: {df['acc_mean'].mean():.4f}")
    
    print(f"\n{'='*80}\n")
    
    # Mostra prime righe del DataFrame
    print("Prime righe del CSV:")
    cols_to_show = ['dataset', 'clustering_strategy', 'number_of_clusters', 
                    'num_selected_features', 'nmi_mean', 'acc_mean']
    cols_to_show = [col for col in cols_to_show if col in df.columns]
    print(df[cols_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parsa i risultati degli esperimenti FeSCAE e crea un CSV summary'
    )
    parser.add_argument(
        '-rf',
        '--root_folder',
        default='output_experiments',
        help='Cartella root contenente i risultati (default: output_experiments)'
    )
    parser.add_argument(
        '-o', '--output',
        default='results_summary.csv',
        help='Nome del file CSV di output (default: results_summary.csv)'
    )
    
    args = parser.parse_args()
    
    parse_all_logs(args.root_folder, args.output)
