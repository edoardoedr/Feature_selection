import torch
import torch.nn as nn

class LinearAE(nn.Module):
    def __init__(self, n_input):
        super(LinearAE, self).__init__()
        
        # Definiamo i livelli
        self.fc1 = nn.Linear(n_input, n_input // 5)
        self.fc2 = nn.Linear(n_input // 5, n_input // 10)
        self.fc3 = nn.Linear(n_input // 10, n_input // 5)
        self.fc_out = nn.Linear(n_input // 5, n_input)
        
        # Funzione di attivazione
        self.relu = nn.ReLU()
        
        # Inizializzazione Xavier
        self._initialize_weights()

    def _initialize_weights(self):
        # Applica inizializzazione Xavier ai pesi dei livelli fully connected
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        
        # Imposta i bias a zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x):
        # Applicazione dei livelli e attivazioni
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)  # Livello finale con attivazione lineare (default)
        
        return x