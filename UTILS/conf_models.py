

#DICT con hiperparámetros para DRAE según número de features

#PROSTATE


def _drae_config_for_num_features(dataset: str, num_fea: int):
    """
    Devuelve dict de hiperparámetros DRAE según num_fea.
    """
    if dataset.lower() == "prostate-ge":
        if num_fea == 10:
            return dict(d_hidden=40, alpha=1e8, beta=10, gamma=1000, zeta=0.001, lam=0.001)
        elif num_fea == 20:
            return dict(d_hidden=40, alpha=1e8, beta=10, gamma=1000, zeta=0.001, lam=0.1)
        elif num_fea == 30:
            return dict(d_hidden=40, alpha=1e8, beta=10, gamma=1000, zeta=0, lam=0.01)
        elif num_fea in (40, 50, 90):
            return dict(d_hidden=40, alpha=1e8, beta=0, gamma=10000, zeta=0.001, lam=10)
        elif num_fea == 60:
            return dict(d_hidden=10, alpha=1e8, beta=0.001, gamma=10000, zeta=0.01, lam=0.01)
        elif num_fea == 70:
            return dict(d_hidden=30, alpha=1e8, beta=0.001, gamma=100000, zeta=0, lam=0.01)
        elif num_fea == 80:
            return dict(d_hidden=30, alpha=1e8, beta=0.001, gamma=1000, zeta=0.001, lam=0.1)
        elif num_fea == 100:
            return dict(d_hidden=50, alpha=1e8, beta=0.01, gamma=1000, zeta=0.001, lam=0)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "colon":

        if num_fea == 10:
            return dict(d_hidden=20, alpha=1e8, beta=0, gamma=100, zeta=0.001, lam=1000)
        elif num_fea == 20:
            return dict(d_hidden=50, alpha=1e8, beta=0.001, gamma=100, zeta=0.001, lam=10)
        elif num_fea == 30:
            return dict(d_hidden=50, alpha=1e8, beta=10, gamma=1000, zeta=0.001, lam=0)
        elif num_fea == 40:
            return dict(d_hidden=40, alpha=1e8, beta=0.001, gamma=1000, zeta=0.001, lam=0.001)
        elif num_fea == 50:
            return dict(d_hidden=20, alpha=1e8, beta=0, gamma=100, zeta=0, lam=0)
        elif num_fea == 60:
            return dict(d_hidden=30, alpha=1e8, beta=10, gamma=100, zeta=0.001, lam=0.001)
        elif num_fea == 70:
            return dict(d_hidden=40, alpha=1e8, beta=0.001, gamma=100000, zeta=0, lam=0)
        elif num_fea == 80:
            return dict(d_hidden=50, alpha=1e8, beta=10, gamma=1000, zeta=0, lam=0)
        elif num_fea in (90, 100):
            return dict(d_hidden=50, alpha=1e8, beta=0, gamma=100, zeta=0, lam=100)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "gli-85":
        if num_fea == 10:
            return dict(d_hidden=10, alpha=1e8, beta=0.01, gamma=100000, zeta=0.001, lam=0.01)
        elif num_fea == 20:
            return dict(d_hidden=50, alpha=1e8, beta=1000, gamma=1000, zeta=10, lam=0.01)
        elif num_fea in (30, 40):
            return dict(d_hidden=10, alpha=1e8, beta=0, gamma=100000, zeta=0.01, lam=100)
        elif num_fea == 50:
            return dict(d_hidden=30, alpha=1e8, beta=0, gamma=100000, zeta=0.01, lam=1000)
        elif num_fea == 60:
            return dict(d_hidden=10, alpha=1e8, beta=0.1, gamma=10000, zeta=0.01, lam=0.1)
        elif num_fea == 70:
            return dict(d_hidden=20, alpha=1e8, beta=0.1, gamma=10000, zeta=0.1, lam=10)
        elif num_fea == 80:
            return dict(d_hidden=10, alpha=1e8, beta=0.001, gamma=10000, zeta=0.01, lam=100)
        elif num_fea == 90:
            return dict(d_hidden=50, alpha=1e8, beta=0.01, gamma=100, zeta=0.001, lam=0.01)
        elif num_fea == 100:
            return dict(d_hidden=10, alpha=1e8, beta=0.1, gamma=10000, zeta=0, lam=10)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "leukemia":
        if num_fea == 10:
            return dict(d_hidden=50, alpha=1e8, beta=0, gamma=100000, zeta=0.1, lam=10)
        elif num_fea == 20:
            return dict(d_hidden=50, alpha=1e8, beta=0.001, gamma=100000, zeta=0.1, lam=10)
        elif num_fea == 30:
            return dict(d_hidden=50, alpha=1e8, beta=10, gamma=100000, zeta=0, lam=0)
        elif num_fea == 40:
            return dict(d_hidden=20, alpha=1e8, beta=0.001, gamma=100000, zeta=0.1, lam=0.1)
        elif num_fea == 50:
            return dict(d_hidden=50, alpha=1e8, beta=1000, gamma=1000, zeta=10, lam=0.01)
        elif num_fea == 60:
            return dict(d_hidden=30, alpha=1e8, beta=10, gamma=100000, zeta=0, lam=0.001)
        elif num_fea == 70:
            return dict(d_hidden=50, alpha=1e8, beta=0, gamma=100000, zeta=0.1, lam=10)
        elif num_fea == 80:
            return dict(d_hidden=50, alpha=1e8, beta=10, gamma=10000, zeta=0, lam=0)
        elif num_fea == 90:
            return dict(d_hidden=40, alpha=1e8, beta=10, gamma=10000, zeta=0.1, lam=0.001)
        elif num_fea == 100:
            return dict(d_hidden=30, alpha=1e8, beta=100, gamma=100000, zeta=0, lam=0.001)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "tox-171":
        if num_fea == 10:
            return dict(d_hidden=10, alpha=1e8, beta=0.001, gamma=100000, zeta=0, lam=100)
        elif num_fea == 20:
            return dict(d_hidden=40, alpha=1e8, beta=0.1, gamma=10000, zeta=0.01, lam=100)
        elif num_fea == 30:
            return dict(d_hidden=50, alpha=1e8, beta=10, gamma=10000, zeta=0.1, lam=10)
        elif num_fea == 40:
            return dict(d_hidden=40, alpha=1e8, beta=0, gamma=100000, zeta=100, lam=0.1)
        elif num_fea == 50:
            return dict(d_hidden=30, alpha=1e8, beta=10, gamma=10000, zeta=0.1, lam=100)
        elif num_fea == 60:
            return dict(d_hidden=10, alpha=1e8, beta=10, gamma=100000, zeta=0.001, lam=0.01)
        elif num_fea == 70:
            return dict(d_hidden=20, alpha=1e8, beta=0, gamma=100000, zeta=100, lam=100)
        elif num_fea == 80:
            return dict(d_hidden=10, alpha=1e8, beta=0.1, gamma=10000, zeta=0.001, lam=0.01)
        elif num_fea == 90:
            return dict(d_hidden=20, alpha=1e8, beta=0.01, gamma=10000, zeta=0.01, lam=100)
        elif num_fea == 100:
            return dict(d_hidden=20, alpha=1e8, beta=0.1, gamma=100000, zeta=0.01, lam=1000)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "lymphoma":

        if num_fea == 10:
            return dict(d_hidden=30, alpha=1e8, beta=0.1, gamma=10000, zeta=0.001, lam=10)
        elif num_fea == 20:
            return dict(d_hidden=30, alpha=1e8, beta=0.1, gamma=10000, zeta=0.1, lam=10)
        elif num_fea == 30:
            return dict(d_hidden=50, alpha=1e8, beta=100, gamma=100000, zeta=100, lam=1000)
        elif num_fea == 40:
            return dict(d_hidden=40, alpha=1e8, beta=0.01, gamma=100000, zeta=0.1, lam=10)
        elif num_fea == 50:
            return dict(d_hidden=50, alpha=1e8, beta=1000, gamma=100, zeta=1000, lam=0.001)
        elif num_fea == 60:
            return dict(d_hidden=30, alpha=1e8, beta=1000, gamma=1000, zeta=1000, lam=1000)
        elif num_fea == 70:
            return dict(d_hidden=50, alpha=1e8, beta=100, gamma=100, zeta=100, lam=0)
        elif num_fea == 80:
            return dict(d_hidden=40, alpha=1e8, beta=0.01, gamma=100000, zeta=0.1, lam=10)
        elif num_fea == 90:
            return dict(d_hidden=40, alpha=1e8, beta=1000, gamma=100, zeta=1000, lam=100)
        elif num_fea == 100:
            return dict(d_hidden=50, alpha=1e8, beta=100, gamma=100, zeta=100, lam=10)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
            
        

def _smlae_config_for_num_features(dataset:str, num_fea: int):
    """
    Devuelve dict de hiperparámetros SMLAE según num_fea
    """
    if dataset.lower() == "prostate-ge":
        if num_fea == 10:
            return dict(d_hidden=40, n_neighbors=5, alpha=1e-6, beta=1e-6, gamma=1e-6, omega=1e-6, rho1=1e-5, rho2=1e-5)
        elif num_fea == 20:
            return dict(d_hidden=30, n_neighbors=5, alpha=10, beta=1e-5, gamma=0.1, omega=1e-4, rho1=1e-5, rho2=1e-5)
        elif num_fea == 30:
            return dict(d_hidden=40, n_neighbors=5, alpha=100, beta=1e-6, gamma=1e-6, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 40:
            return dict(d_hidden=40, n_neighbors=5, alpha=100, beta=1e-6, gamma=1e-6, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 50:
            return dict(d_hidden=20, n_neighbors=5, alpha=1e-4, beta=1e-6, gamma=10000, omega=1e-4, rho1=1e-5, rho2=1e-5)
        elif num_fea == 60:
            return dict(d_hidden=40, n_neighbors=5, alpha=100, beta=1e-6, gamma=1000000, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 70:
            return dict(d_hidden=40, n_neighbors=5, alpha=10, beta=1e-6, gamma=1e-6, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 80:
            return dict(d_hidden=10, n_neighbors=5, alpha=1000, beta=1000, gamma=1e-4, omega=1e-5, rho1=1e-5, rho2=1e-5)
        elif num_fea == 90:
            return dict(d_hidden=40, n_neighbors=5, alpha=10, beta=1e-5, gamma=1000000, omega=1e-3, rho1=1e-5, rho2=1e-5)
        elif num_fea == 100:
            return dict(d_hidden=50, n_neighbors=5, alpha=100, beta=0, gamma=10, omega=10000, rho1=1e-5, rho2=1e-5)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    if dataset.lower() == "colon":
        if num_fea == 10:
            return dict(d_hidden=20, n_neighbors=5, alpha=10, beta=10000, gamma=10000, omega=100, rho1=1e-5, rho2=1e-5)
        elif num_fea == 20:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000, beta=1e-5, gamma=0.01, omega=0.001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 30:
            return dict(d_hidden=30, n_neighbors=5, alpha=0.1, beta=1e-6, gamma=1000000, omega=0.0001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 40:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000000, beta=10000, gamma=0, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 50:
            return dict(d_hidden=10, n_neighbors=5, alpha=0.01, beta=0, gamma=100, omega=1000000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 60:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000000, beta=1000, gamma=100, omega=1000000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 70:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000000, beta=10000, gamma=0, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 80:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000000, beta=10000, gamma=0, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 90:
            return dict(d_hidden=20, n_neighbors=5, alpha=1e-6, beta=0, gamma=0, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 100:
           return dict(d_hidden=30, n_neighbors=5, alpha=1000000, beta=10000, gamma=0, omega=10000, rho1=1e-5, rho2=1e-5)
        else:
           raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    if dataset.lower() == "gli-85":
        if num_fea == 10:
            return dict(d_hidden=10, n_neighbors=5, alpha=0.0001, beta=0.0001, gamma=1000000, omega=1e-6, rho1=1e-5, rho2=1e-5)
        elif num_fea == 20:
            return dict(d_hidden=40, n_neighbors=5, alpha=1000, beta=1000000, gamma=100, omega=0.01, rho1=1e-5, rho2=1e-5)
        elif num_fea == 30:
            return dict(d_hidden=40, n_neighbors=5, alpha=0.01, beta=10, gamma=1000, omega=10, rho1=1e-5, rho2=1e-5)
        elif num_fea == 40:
            return dict(d_hidden=50, n_neighbors=5, alpha=0.0001, beta=0.01, gamma=100, omega=0.01, rho1=1e-5, rho2=1e-5)
        elif num_fea == 50:
            return dict(d_hidden=50, n_neighbors=5, alpha=0, beta=100, gamma=0, omega=1e-6, rho1=1e-5, rho2=1e-5)
        elif num_fea == 60:
            return dict(d_hidden=20, n_neighbors=5, alpha=0.1, beta=1000, gamma=0.001, omega=0.0001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 70:
            return dict(d_hidden=50, n_neighbors=5, alpha=0.001, beta=10000, gamma=0.0001, omega=1e-5, rho1=1e-5, rho2=1e-5)
        elif num_fea == 80:
            return dict(d_hidden=20, n_neighbors=5, alpha=100, beta=100, gamma=10, omega=100, rho1=1e-5, rho2=1e-5)
        elif num_fea == 90:
            return dict(d_hidden=40, n_neighbors=5, alpha=0.1, beta=0.0001, gamma=0.001, omega=1000000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 100:
            return dict(d_hidden=20, n_neighbors=5, alpha=1000, beta=0.0001, gamma=1e-5, omega=1e-5, rho1=1e-5, rho2=1e-5)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")  
    if dataset.lower() == "leukemia":
        if num_fea == 10:
            return dict(d_hidden=10, n_neighbors=5, alpha=1e-6, beta=1000, gamma=0.01, omega=1000000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 20:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000, beta=1000000, gamma=0, omega=100, rho1=1e-5, rho2=1e-5)
        elif num_fea == 30:
            return dict(d_hidden=30, n_neighbors=5, alpha=0.001, beta=100, gamma=1000, omega=0.0001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 40:
            return dict(d_hidden=30, n_neighbors=5, alpha=0, beta=10, gamma=1000, omega=0, rho1=1e-5, rho2=1e-5)
        elif num_fea == 50:
            return dict(d_hidden=30, n_neighbors=5, alpha=1e-6, beta=1000, gamma=0.0001, omega=0.0001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 60:
            return dict(d_hidden=10, n_neighbors=5, alpha=10, beta=1e-5, gamma=1e-5, omega=100, rho1=1e-5, rho2=1e-5)
        elif num_fea == 70:
            return dict(d_hidden=40, n_neighbors=5, alpha=1e-6, beta=0.01, gamma=1e-6, omega=1000000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 80:
            return dict(d_hidden=40, n_neighbors=5, alpha=0.0001, beta=1000000, gamma=1000, omega=1000000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 90:
            return dict(d_hidden=20, n_neighbors=5, alpha=10000, beta=1e-5, gamma=1000000, omega=0.1, rho1=1e-5, rho2=1e-5)
        elif num_fea == 100:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000000, beta=10000, gamma=0, omega=10000, rho1=1e-5, rho2=1e-5)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")   
    if dataset.lower() == "lymphoma":
        if num_fea == 10:
            return dict(d_hidden=50, n_neighbors=5, alpha=0.01, beta=0.1, gamma=1000000, omega=0.001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 20:
            return dict(d_hidden=20, n_neighbors=5, alpha=1000, beta=100, gamma=10000, omega=0, rho1=1e-5, rho2=1e-5)
        elif num_fea == 30:
            return dict(d_hidden=40, n_neighbors=5, alpha=1e-6, beta=1e-5, gamma=1e-6, omega=0, rho1=1e-5, rho2=1e-5)
        elif num_fea == 40:
            return dict(d_hidden=50, n_neighbors=5, alpha=0.01, beta=0, gamma=1000, omega=0.0001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 50:
            return dict(d_hidden=10, n_neighbors=5, alpha=100, beta=0.1, gamma=0.1, omega=0, rho1=1e-5, rho2=1e-5)
        elif num_fea == 60:
            return dict(d_hidden=20, n_neighbors=5, alpha=0.01, beta=1e-5, gamma=1e-6, omega=1000000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 70:
            return dict(d_hidden=50, n_neighbors=5, alpha=1000, beta=0, gamma=0, omega=1000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 80:
            return dict(d_hidden=40, n_neighbors=5, alpha=1e-5, beta=1e-5, gamma=1000000, omega=10, rho1=1e-5, rho2=1e-5)
        elif num_fea == 90:
            return dict(d_hidden=40, n_neighbors=5, alpha=1e-5, beta=1e-5, gamma=100, omega=0.0001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 100:
            return dict(d_hidden=30, n_neighbors=5, alpha=1e-5, beta=100, gamma=0.01, omega=10, rho1=1e-5, rho2=1e-5)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    if dataset.lower() == "tox-171":
        if num_fea == 10:
            return dict(d_hidden=30, n_neighbors=5, alpha=0.01, beta=0.1, gamma=100, omega=10000,  rho1=1e-5, rho2=1e-5)
        elif num_fea == 20:
            return dict(d_hidden=10, n_neighbors=5, alpha=10, beta=10, gamma=1e-5, omega=0.1,  rho1=1e-5, rho2=1e-5)
        elif num_fea == 30:
            return dict(d_hidden=20, n_neighbors=5, alpha=1e-5, beta=100, gamma=0.1, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 40:
            return dict(d_hidden=30, n_neighbors=5, alpha=1e-6, beta=1e-6, gamma=0.01, omega=10000, rho1=1e-5, rho2=1e-5)
        elif num_fea == 50:
            return dict(d_hidden=20, n_neighbors=5, alpha=1000000, beta=0.01, gamma=0.01, omega=1e-6, rho1=1e-5, rho2=1e-5)
        elif num_fea == 60:
            return dict(d_hidden=20, n_neighbors=5, alpha=1e-5, beta=1000000, gamma=0.01, omega=10, rho1=1e-5, rho2=1e-5)
        elif num_fea == 70:
            return dict(d_hidden=10, n_neighbors=5, alpha=1000000, beta=0.0001, gamma=1e-5, omega=0.1, rho1=1e-5, rho2=1e-5)
        elif num_fea == 80:
            return dict(d_hidden=30, n_neighbors=5, alpha=0.1, beta=1e-5, gamma=0.1, omega=0.0001, rho1=1e-5, rho2=1e-5)
        elif num_fea == 90:
            return dict(d_hidden=30, n_neighbors=5, alpha=1000000, beta=10, gamma=0.1, omega=0.1, rho1=1e-5, rho2=1e-5)
        elif num_fea == 100:
            return dict(d_hidden=20, n_neighbors=5, alpha=0, beta=0.1, gamma=0.0001, omega=1e-6, rho1=1e-5, rho2=1e-5)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")

    
def _aefs_config_for_num_features(dataset: str, num_fea: int):
    """
    Devuelve dict de hiperparámetros AEFS según num_fea
    """

    if dataset.lower() == "prostate-ge":
        if num_fea in (10, 100):
            return dict(hidden=1024, alpha=0.5, beta=1, lr=1e-3)
        elif num_fea in (20, 40):
            return dict(hidden=1024, alpha=1, beta=1, lr=1e-3)
        elif num_fea ==30:
            return dict(hidden=1024, alpha=0.01, beta=0.01, lr=1e-3)
        elif num_fea ==50:
            return dict(hidden=1024, alpha=1, beta=0.1, lr=1e-3)
        elif num_fea == 60:
            return dict(hidden=1024, alpha=0.5, beta=0.1, lr=1e-3)
        elif num_fea == 70:
            return dict(hidden=1024, alpha=1, beta=10, lr=1e-3)
        elif num_fea == 80:
            return dict(hidden=1024, alpha=0.05, beta=0.01, lr=1e-3)
        elif num_fea ==  90:
            return dict(hidden=1024, alpha=1, beta=0.001, lr=1e-3)
        else:
            raise ValueError(f"No hay configuración definida para numeros de features={num_fea}")
    elif dataset.lower() == "colon":
        if num_fea == 10:
            return dict(hidden=1024, alpha=0.05, beta=10, lr=1e-3)
        elif num_fea == 20:
            return dict(hidden=512, alpha=10, beta=0.05, lr=1e-3)
        elif num_fea == 30:
            return dict(hidden=128, alpha=0.5, beta=0.05, lr=1e-3)
        elif num_fea == 40:
            return dict(hidden=128, alpha=0.5, beta=0.5, lr=1e-3)
        elif num_fea == 50:
            return dict(hidden=128, alpha=1, beta=0.5, lr=1e-3)
        elif num_fea == 60:
            return dict(hidden=128, alpha=1, beta=1, lr=1e-3)
        elif num_fea in (70, 100):
            return dict(hidden=256, alpha=1, beta=10, lr=1e-3)
        elif num_fea == 80:
            return dict(hidden=512, alpha=1, beta=10, lr=1e-3)
        elif num_fea == 90:
            return dict(hidden=128, alpha=1, beta=0.05, lr=1e-3)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    elif dataset.lower() == "leukemia":
        if num_fea in (10, 30):
            return dict(hidden=1024, alpha=0.05, beta=10, lr=1e-3)
        elif num_fea  in (20, 40, 50, 60, 70, 80, 90, 100):
            return dict(hidden=1024, alpha=0.5, beta=10, lr=1e-3)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    elif dataset.lower() == "lymphoma":
        if num_fea in (10, 30, 40, 50, 60, 70, 80, 90, 100):
            return dict(hidden=256, alpha=0.5, beta=0.5, lr=1e-3)
        elif num_fea == 20:
            return dict(hidden=1024, alpha=0.5, beta=1, lr=1e-3)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    elif dataset.lower() == "tox-171":
        if num_fea in (10, 20):
            return dict(hidden=1024, alpha=10, beta=0.05, lr=1e-3)
        elif num_fea == 30:
            return dict(hidden=256, alpha=0.05, beta=10, lr=1e-3)
        elif num_fea in (40, 50, 60, 70, 80, 90, 100):
            return dict(hidden=128, alpha=1, beta=1, lr=1e-3)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    elif dataset.lower() == "gli-85":
        if num_fea in (10, 20, 30, 40, 50, 60, 70, 80, 100):
            return dict(hidden=256, alpha=0.5, beta=1, lr=1e-3)
        elif num_fea == 90:
            return dict(hidden=256, alpha=0.5, beta=0.5, lr=1e-3)
        else:
            raise ValueError(f"No hay configuración definida para nuemros de features={num_fea}")
    
def _dga_config_for_num_features(dataset: str, num_fea: int):
    """
    Devuelve dict de hiperparámetros DGAE según num_fea
    """
    if dataset.lower() == "prostate-ge":

        if num_fea == 10:
            return dict(hidden=10, gamma=0.1, lam=0.05)
        elif num_fea == 20:
            return dict(hidden=10, gamma=0.001, lam=10)
        elif num_fea == 30:
            return dict(hidden=50, gamma=0.001, lam=0.05)
        elif num_fea == 40:
            return dict(hidden=40, gamma=0.001, lam=0.01)
        elif num_fea == 50:
            return dict(hidden=40, gamma=0.001, lam=0.001)
        elif num_fea == 60:
            return dict(hidden=20, gamma=0.5, lam=0.05)
        elif num_fea == 70:
            return dict(hidden=10, gamma=0.01, lam=0.05)
        elif num_fea == 80:
            return dict(hidden=20, gamma=0.1, lam=10)
        elif num_fea == 90:
            return dict(hidden=20, gamma=0.01, lam=10)
        elif num_fea == 100:
            return dict(hidden=40, gamma=0.05, lam=0.5)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "colon":

        if num_fea == 10:
            return dict(hidden=50, gamma=0.01, lam=10)
        elif num_fea == 20:
            return dict(hidden=50, gamma=1, lam=0.5)
        elif num_fea == 30:
            return dict(hidden=10, gamma=1, lam=0.05)
        elif num_fea == 40:
            return dict(hidden=20, gamma=0.01, lam=0.001)
        elif num_fea == 50:
            return dict(hidden=20, gamma=0.5, lam=0.1)
        elif num_fea == 60:
            return dict(hidden=20, gamma=0.5, lam=1)
        elif num_fea == 70:
            return dict(hidden=20, gamma=0.5, lam=1)
        elif num_fea == 80:
            return dict(hidden=20, gamma=0.5, lam=0.1)
        elif num_fea == 90:
            return dict(hidden=40, gamma=0.01, lam=10)
        elif num_fea == 100:
            return dict(hidden=20, gamma=1, lam=0.01)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "gli-85":

        if num_fea == 10:
            return dict(hidden=30, gamma=0.1, lam=0.01)
        elif num_fea == 20:
            return dict(hidden=30, gamma=0.1, lam=10)
        elif num_fea == 30:
            return dict(hidden=30, gamma=0.05, lam=0.1)
        elif num_fea == 40:
            return dict(hidden=10, gamma=0.001, lam=0.001)
        elif num_fea == 50:
            return dict(hidden=40, gamma=1, lam=0.5)
        elif num_fea == 60:
            return dict(hidden=30, gamma=1, lam=0.001)
        elif num_fea == 70:
            return dict(hidden=40, gamma=0.1, lam=0.1)
        elif num_fea == 80:
            return dict(hidden=40, gamma=10, lam=0.001)
        elif num_fea == 90:
            return dict(hidden=40, gamma=0.05, lam=0.05)
        elif num_fea == 100:
            return dict(hidden=50, gamma=1, lam=10)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
        
    if dataset.lower() == "leukemia":
        if num_fea == 10:
            return dict(hidden=40, gamma=0.01, lam=0.5)
        elif num_fea == 20:
            return dict(hidden=20, gamma=0.1, lam=0.001)
        elif num_fea == 30:
            return dict(hidden=30, gamma=0.1, lam=10)
        elif num_fea == 40:
            return dict(hidden=20, gamma=1, lam=0.05)
        elif num_fea == 50:
            return dict(hidden=20, gamma=0.5, lam=1)
        elif num_fea == 60:
            return dict(hidden=10, gamma=0.5, lam=1)
        elif num_fea == 70:
            return dict(hidden=40, gamma=1, lam=0.05)
        elif num_fea in (80, 90):
            return dict(hidden=10, gamma=0.1, lam=10)
        elif num_fea == 100:
            return dict(hidden=40, gamma=0.01, lam=0.5)

        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "tox-171":

        if num_fea == 10:
            return dict(hidden=20, gamma=10, lam=0.001)
        elif num_fea == 20:
            return dict(hidden=30, gamma=0.5, lam=0.5)
        elif num_fea == 30:
            return dict(hidden=20, gamma=10, lam=0.05)
        elif num_fea == 40:
            return dict(hidden=40, gamma=0.01, lam=0.05)
        elif num_fea == 50:
            return dict(hidden=40, gamma=0.1, lam=10)
        elif num_fea == 60:
            return dict(hidden=50, gamma=0.5, lam=0.1)
        elif num_fea == 70:
            return dict(hidden=50, gamma=10, lam=1)
        elif num_fea == 80:
            return dict(hidden=20, gamma=0.001, lam=0.001)
        elif num_fea == 90:
            return dict(hidden=10, gamma=10, lam=0.001)
        elif num_fea == 100:
            return dict(hidden=40, gamma=0.001, lam=0.01)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")
    if dataset.lower() == "lymphoma":

        if num_fea == 10:
            return dict(hidden=20, gamma=0.001, lam=0.1)
        elif num_fea == 20:
            return dict(hidden=20, gamma=10, lam=0.01)
        elif num_fea == 30:
            return dict(hidden=40, gamma=0.5, lam=1)
        elif num_fea == 40:
            return dict(hidden=50, gamma=0.001, lam=0.01)
        elif num_fea == 50:
            return dict(hidden=30, gamma=0.001, lam=0.5)
        elif num_fea == 60:
            return dict(hidden=50, gamma=0.1, lam=10)
        elif num_fea == 70:
            return dict(hidden=10, gamma=0.01, lam=0.1)
        elif num_fea == 80:
            return dict(hidden=30, gamma=0.05, lam=0.05)
        elif num_fea == 90:
            return dict(hidden=20, gamma=10, lam=0.05)
        elif num_fea == 100:
            return dict(hidden=50, gamma=1, lam=0.05)
        else:
            raise ValueError(f"No hay configuración definida para num_fea={num_fea}")