from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_scaler(self):
    if self.scaler_type == "standard":
        return StandardScaler()
    elif self.scaler_type == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Scaler '{self.scaler_type}' not supported.")