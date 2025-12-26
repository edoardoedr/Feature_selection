import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import copy
import random
from .autoencoders import LinearAE
from .loggers import FilterLogger

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

class AEFilter:
    def __init__(self, lr, epoch, model_name, iteration, output_dir, scaler):
        self.model_name = model_name
        self.model = None
        self.lr = lr
        self.epoch = epoch
        self.iteration = iteration
        self.best_weights = None
        self.output_dir = output_dir
        self.logger = FilterLogger(iteration, output_dir=output_dir)
        self.scaler = scaler

    def get_data(self, data, clusters, i_clust):
        # Seleziona i geni del cluster corrente
        current_features = clusters['Feature'][clusters['Cluster'] == i_clust].values
        X = data[current_features].T.copy()
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            self.logger.log_message(f"Skipping training for cluster {i_clust} due to empty feature set.")
            X = torch.empty(0, 0)
            return X, X, X
        
        torch.manual_seed(SEED)
        random.seed(SEED)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), index=X.index)
        # Converte i dati in tensori
        X_tensor = torch.tensor(X_scaled.to_numpy(), dtype=torch.float32)
        
        return X_scaled.shape[1], X_tensor, X_scaled

    def create_model(self, input_size):
        if self.model_name == "LinearAE":
            model = LinearAE(input_size)  # Assumo che LinearAE sia definito altrove
        else:
            raise ValueError(f"Modello {self.model_name} non supportato.")
        return model

    def fit(self, data, clusters, i_clust):
        # Prepara i dati e il modello
        num_feat, X_tensor, df_scaled = self.get_data(data, clusters, i_clust)
        
        if X_tensor.shape[0] == 0 or X_tensor.shape[1] == 0:
            self.logger.log_message(f"Skipping training for cluster {i_clust} due to empty feature set.")
            return None
        
        self.logger.log_cluster_start(i_clust, num_feat)
        torch.manual_seed(SEED)
        random.seed(SEED)
        self.model = self.create_model(num_feat)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()
        best_acc = float('inf')

        for epoch in range(self.epoch):
            self.model.train()
            optimizer.zero_grad()

            outputs = self.model(X_tensor)
            loss = loss_fn(outputs, X_tensor)
            loss.backward()
            optimizer.step()

            # Valutazione dell'accuratezza (MAE tra input e output)
            accuracy = torch.mean(torch.abs(outputs - X_tensor)).item()

            # Log del progresso di addestramento
            self.logger.log_training_progress(epoch, self.epoch, loss.item(), accuracy)

            # Se è il miglior modello, salva i pesi
            if accuracy < best_acc:
                best_acc = accuracy
                self.best_weights = copy.deepcopy(self.model.state_dict())
                
            # Stampa opzionale del progresso
            # print(f"Epoca: {epoch}, Loss: {loss.item()}, Accurattezza: {accuracy}")

        # Carica i migliori pesi
        self.model.load_state_dict(self.best_weights)

        # Previsione e selezione del miglior gene
        preds = self.predict(X_tensor)
        selected_idx = torch.argmin(torch.mean(torch.abs(X_tensor - preds), dim=1)).item()
        selected_gene = df_scaled.index.values[selected_idx]

        # Calcola l'errore di ricostruzione per il gene selezionato
        selected_error = torch.mean(torch.abs(X_tensor[selected_idx] - preds[selected_idx])).item()
        
        # Log della feature selezionata
        self.logger.log_feature_selection(i_clust, selected_gene, selected_error)


        return selected_gene

    def predict(self, X_test):
        # Controlla se X_test è un DataFrame e converti in tensore
        if isinstance(X_test, pd.DataFrame):
            X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

        # Modalità valutazione (no gradiente)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test)

        return logits

    def filter(self, data, clusters):
        selection = []
        n_clusters = max(clusters['Cluster']) + 1

        self.logger.log_message(f'Iniziando filtraggio con autoencoder per {n_clusters-1} cluster')

        for i in range(1, n_clusters): #TODO: parallelizzare il codice
            selected_genes = self.fit(data, clusters, i)
            if selected_genes != None:
                selection.append(selected_genes)

            self.logger.log_message(f'Filtered Clusters {i}/{n_clusters - 1}')
        # Log riassuntivo finale
        self.logger.log_selected_features_summary(selection)
        self.logger.log_completion()
        
        return selection