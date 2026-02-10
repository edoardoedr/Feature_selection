from M.UTILS.utils import set_seed
set_seed(seed=42, deterministic=True)

import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
from M.MODELS import MCFS
from M.MODELS import NDFS
from M.UTILS import construct_W
from M.UTILS import unsupervised_evaluation
from M.UTILS import sparse_learning
from M.MODELS.DRAE import DRAE
from M.MODELS.SMLAE import SMLAE
from M.MODELS.RFAE import cal 
from M.MODELS.CAE import ConcreteAutoencoderFeatureSelector
from M.MODELS.AEFS import AEFS
from M.MODELS.DGA import DGA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import datetime
from M.UTILS.conf_models import _drae_config_for_num_features, _smlae_config_for_num_features, _aefs_config_for_num_features, _dga_config_for_num_features




# Para metodos MCFS, NDFS
num_features = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def resultsMCFS(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Devuelve por cada num_features:
    accuracy, std_acc, nmi, std_nmi
    (calculados sobre 'repeat' repeticiones) y guarda CSV en:
    Model_results/MCFS/results_MCFS_{dataset_name}
    Incluye el tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)
    y = mat["Y"][:, 0]

    num_cluster = len(np.unique(y))

    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, "t": 1}
    W = construct_W.construct_W(X, **kwargs)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()  # <-- tiempo por este num_features

        weight = MCFS.mcfs(X, n_selected_features=num_fea, W=W, n_clusters=num_cluster)
        idx = MCFS.feature_ranking(weight)
        selected_features = X[:, idx[:num_fea]]

        acc_runs, nmi_runs = [], []
        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=num_cluster,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0  # <-- segundos

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),  # <-- guardado en CSV
        })

    df = pd.DataFrame(results)

    results_dir = Path(__file__).resolve().parents[1] / "Model_results" / "MCFS" / f"results_MCFS_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"MCFS_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"MCFS_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df


def resultsNDFS(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre NDFS y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/NDFS/results_NDFS_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)
    y = mat["Y"][:, 0]

    num_cluster = len(np.unique(y))

    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, "t": 1}
    W = construct_W.construct_W(X, **kwargs)

    Weight = NDFS.ndfs(X, W=W, n_clusters=num_cluster)
    idx = sparse_learning.feature_ranking(Weight)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()

        selected_features = X[:, idx[:num_fea]]

        acc_runs, nmi_runs = [], []
        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=num_cluster,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = Path(__file__).resolve().parents[1] / "Model_results" / "NDFS" / f"results_NDFS_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"NDFS_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"NDFS_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df


def resultsDRAE(dataset_name: str, repeats: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre DRAE y evalúa con KMeans 'repeats' repeticiones.
    Guarda CSV en: Model_results/DRAE/results_DRAE_{dataset}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)   # data
    y = mat["Y"][:, 0]           # label

    XT = X.T                     # (d, n)
    n_clusters = len(np.unique(y))


    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        set_seed(42, deterministic=True)
        t0 = time.time()

        cfg = _drae_config_for_num_features(dataset_name, num_fea)

        # --- Entrenar DRAE con su config (SIN CAMBIOS) ---
        drae = DRAE(
            d_hidden=cfg["d_hidden"],
            n_clusters=n_clusters,
            alpha=1e8,
            beta=cfg["beta"],
            gamma=cfg["gamma"],
            zeta=cfg["zeta"],
            lam=cfg["lam"],
        )
        drae.fit(XT, epochs=100)

        # Seleccionar top features
        idx, scores = drae.select_top_genes(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        # Evaluación (KMeans repeats)
        for _ in range(repeats):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y,
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": int(num_fea),
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "DRAE"
        / f"results_DRAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"DRAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"DRAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df



def resultsSMLAE(dataset_name: str, repeats: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre SMLAE y evalúa con KMeans 'repeats' repeticiones.
    Guarda CSV en: Model_results/SMLAE/results_SMLAE_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)   # data
    y = mat["Y"][:, 0]           # label

    X = normalize_zscore(X)

    XT = X.T                     # (d, n)
    d, n = XT.shape
    n_clusters1 = len(np.unique(y))   # clusters reales para evaluación
    n_clusters = np.sqrt(n / 2)    # definición usada por SMLAE

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()

        cfg = _smlae_config_for_num_features(dataset_name, num_fea)

        # --- Entrenar SMLAE con su config ---
        smale = SMLAE(
            n_clusters=n_clusters,
            d_hidden=cfg["d_hidden"],
            n_neighbors=5,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            gamma=cfg["gamma"],
            omega=cfg["omega"],
            rho1=cfg["rho1"],
            rho2=cfg["rho2"],
        )
        smale.fit(XT, epochs=100)

        # Seleccionar top features
        idx, scores = smale.select_top_genes(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        # Evaluación (KMeans repeats)
        for _ in range(repeats):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters1,
                y=y,
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": int(num_fea),
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "SMLAE"
        / f"results_SMLAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"SMLAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"SMLAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df


def resultsCAE(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre CAE y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/CAE/results_CAE_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))
    d = X.shape[1]

    X = normalize_zscore(X)

    # output_function mapea (n, K) -> (n, d)  (decoder SIN CAMBIOS)
    def output_function(selected_features):
        model = Sequential([
            Dense(3*num_fea//2, activation="relu", input_shape=(selected_features.shape[1],)),
            Dense(d, activation="linear")
        ])
        return model(selected_features)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()

        cae = ConcreteAutoencoderFeatureSelector(
            num_fea,
            output_function=output_function,
            num_epochs=200,
            batch_size=None,
            learning_rate=0.001,
            start_temp=10.0,
            min_temp=0.1,
            tryout_limit=5,
        )

        cae.fit(X)

        idx = cae.get_support(indices=True).astype(int)
        selected_features = X[:, idx]

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y,
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": int(num_fea),
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "CAE"
        / f"results_CAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"CAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"CAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)  

    return df


def resultsRFAE(dataset_name: str, epochs: int = 1000, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre RFAE y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/RFAE/results_RFAE_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))

    X = normalize_zscore(X)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()

        # -------- RFAE training & feature selection --------
        idx = cal(
            p_data_arr=X,
            p_label_arr_onehot=y,
            p_key_feture_number=num_fea,
            datasetname=dataset_name,
            p_epochs_number=epochs,
            p_batch_size_value=128,
            clf=False,
            p_seed=0,
            device=0
        )

        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "RFAE"
        / f"results_RFAE_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"RFAE_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"RFAE_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df

def normalize_zscore(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std 
    return X_norm

def resultsAEFS(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre AEFS y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/AEFS/results_AEFS_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    
    # AEFS settings EXACTLY as in the paper
    # - hidden ∈ {128, 256, 512, 1024}
    # - sigmoid encoder, linear decoder (handled inside AEFS)

    hidden = 256        # one valid paper value
    alpha = 1e8         # sparsity parameter (to be tuned in grid in experiments)
    beta = 0.001        # weight decay
    lr = 1e-3
    epochs = 100
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))

    X = normalize_zscore(X)

    results = []
    indexfeatures_all = {}


    for num_fea in num_features:
        t0 = time.time()

        cfg = _aefs_config_for_num_features(dataset_name, num_fea)

        aefs = AEFS(hidden=cfg["hidden"], alpha=cfg["alpha"], beta=cfg["beta"], lr=cfg["lr"])

        aefs.fit(X, epochs=100)
        
        # Seleccionar top features
        idx, scores = aefs.select_top_features(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0

        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "AEFS"
        / f"results_AEFS_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"AEFS_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"AEFS_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df


def resultsDGA(dataset_name: str, repeat: int = 20) -> pd.DataFrame:
    """
    Carga {dataset_name}.mat con keys: 'X', 'Y'
    Corre DGA y evalúa con KMeans 'repeat' repeticiones.
    Guarda CSV en: Model_results/DGA/results_DGA_{dataset_name}
    Incluye tiempo (segundos) por cada num_features.
    """

    data_path = Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)
    y = mat["Y"].reshape(-1)
    n_clusters = len(np.unique(y))
    n_samples, d_features = X.shape

    X = normalize_zscore(X)

    results = []
    indexfeatures_all = {}

    for num_fea in num_features:
        t0 = time.time()
        cfg = _dga_config_for_num_features(dataset_name, num_fea)

        dga = DGA( 
            hidden=cfg["hidden"],
            gamma=cfg["gamma"],
            lam=cfg["lam"],
            lr=1e-3)

        dga.fit(X, epochs=100, B=n_samples)
        
        # Seleccionar top features
        idx, scores = dga.select_top_features(num_fea)
        idx = np.array(idx, dtype=int)
        selected_features = X[:, idx]  # (n, num_fea)

        acc_runs, nmi_runs = [], []

        for _ in range(repeat):
            nmi, acc = unsupervised_evaluation.evaluation(
                X_selected=selected_features,
                n_clusters=n_clusters,
                y=y
            )
            acc_runs.append(float(acc))
            nmi_runs.append(float(nmi))

        elapsed = time.time() - t0
        indexfeatures_all[num_fea] = idx[:num_fea]

        results.append({
            "num_features": num_fea,
            "accuracy": float(np.mean(acc_runs)),
            "std_acc": float(np.std(acc_runs, ddof=1)),
            "nmi": float(np.mean(nmi_runs)),
            "std_nmi": float(np.std(nmi_runs, ddof=1)),
            "time_sec": float(elapsed),
        })

    df = pd.DataFrame(results)

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "Model_results"
        / "DGA"
        / f"results_DGA_{dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"DGA_{dataset_name}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Guardar los indices de las features seleccionadas para todos los num_features
    indexfeatures_path = results_dir / f"DGA_{dataset_name}_{ts}_selected_features.npy"
    np.save(indexfeatures_path, indexfeatures_all)

    return df