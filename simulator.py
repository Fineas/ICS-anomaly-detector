from torch.optim import Adam
from torch import nn
import datetime
import random
import torch
import time
from torch.utils.data import DataLoader, TensorDataset

#############################################
# Simulation of Programmable Logic Controller
#############################################
class PLC:
    def __init__(self, id, name=None, active=True):
        self.id = id
        self.name = name or f"PLC-{id}"
        self.active = active  # Determines if the PLC participates in traffic.
        self.log = []         # Keep a log of received messages
        self.timestamp = None

    def send(self, dest_id, data, timestamp=None):
        if not self.active:
            print(f"{self.name} is inactive and cannot send messages.")
            return None
        message = Protocol(source=self.id, destination=dest_id, data=data, timestamp=timestamp)
        print(f"{self.name} sends message to PLC-{dest_id}: {message} | {timestamp}")
        return message

    def receive(self, message, timestamp=None):
        if not self.active:
            print(f"{self.name} is inactive and ignores the message: {message}")
            return
        print(f"{self.name} received message: {message} | {timestamp}")
        self.log.append(message)

    def __repr__(self):
        return f"<{self.name} {'Active' if self.active else 'Inactive'}>"


#########################################
# Simulation of custom Network Protocol
#########################################
class Protocol:
    def __init__(self, source, destination, data, timestamp):
        self.source = source
        self.destination = destination
        self.data = data
        self.timestamp = timestamp

    def __str__(self):
        if self.timestamp:
            time_str = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        else:
            time_str = "Unknown"
        return f"Protocol({self.source}, {self.destination}, {self.data}, {time_str})"

    def __repr__(self):
        return self.__str__()


######################################
# Traffic Simulation (with classification)
######################################
class Traffic:
    def __init__(self, plc_list, classifier=None):
        # Dictionary for easier lookup by id.
        self.plcs = {plc.id: plc for plc in plc_list}
        # Optional PyTorch classifier model (if provided)
        self.classifier = classifier

    def simulate(self, delay=1, event_callback=None):
        """
        Run simulation in an infinite loop. For every communication event, classify it
        using the provided neural network model (if any) and send an event via event_callback.
        """
        try:
            while True:
                active_plcs = [plc for plc in self.plcs.values() if plc.active]
                if len(active_plcs) < 2:
                    if event_callback:
                        event_callback({"event": "waiting", "message": "Not enough active PLCs to simulate communication."})
                    time.sleep(delay)
                    continue

                # Randomly select a sender and a receiver (ensuring they are not the same)
                sender = random.choice(active_plcs)
                receiver = random.choice(active_plcs)
                while sender.id == receiver.id:
                    receiver = random.choice(active_plcs)

                data = random.randint(1, 100)
                timestamp = time.time()
                message = sender.send(receiver.id, data, timestamp)
                if message:
                    receiver.receive(message, timestamp)

                    # Build features: normalized hour, normalized source, normalized destination, normalized data.
                    current_hour = datetime.datetime.fromtimestamp(timestamp).hour
                    features = [
                        current_hour / 23.0,       # normalized hour
                        (sender.id - 1) / 3.0,       # normalized source id
                        (receiver.id - 1) / 3.0,     # normalized destination id
                        data / 100.0               # normalized data
                    ]
                    prediction = 0.0
                    if self.classifier is not None:
                        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            output = self.classifier(input_tensor)
                        prediction = output.item()
                        print(f"Prediction: {prediction}")
                        event_type = "anomaly" if prediction > 0.5 else "normal"
                    else:
                        event_type = "message"

                    event = {
                        "event": event_type,
                        "sender": sender.name,
                        "receiver": receiver.name,
                        "data": data,
                        "timestamp": timestamp,
                        "prediction": prediction
                    }
                    if event_callback:
                        event_callback(event)
                time.sleep(delay)
        except KeyboardInterrupt:
            if event_callback:
                event_callback({"event": "stopped", "message": "Simulation stopped by user."})

    def trigger_anomaly(self, event_callback=None):
        """
        Activate PLC-4 to simulate an anomalous behavior.
        """
        plc4 = self.plcs.get(4)
        if plc4:
            plc4.active = True  # Activate PLC-4.
            print(f"\n--- Incident Triggered: {plc4.name} is now active and participating in the network! ---\n")
        else:
            print("PLC-4 not found in the network.")


def create_traffic_simulator(classifier=None):
    """
    Helper function to create and return a Traffic simulator with four PLC devices.
    PLC-1, PLC-2, and PLC-3 are active by default while PLC-4 is inactive.
    Optionally, a trained classifier can be provided.
    """
    plc1 = PLC(id=1, name="PLC-1", active=True)
    plc2 = PLC(id=2, name="PLC-2", active=True)
    plc3 = PLC(id=3, name="PLC-3", active=True)
    plc4 = PLC(id=4, name="PLC-4", active=False)
    return Traffic([plc1, plc2, plc3, plc4], classifier=classifier)


###############################################
# Dataset Generation and Neural Network Training
###############################################
def generate_dataset():
    """
    Generate a complete dataset simulating all possible communications between
    4 PLCs over 24 hours. There are 12 events per hour (directed communication)
    for a total of 24*12 = 288 samples.
    
    Label: 1 (anomaly) if the event involves PLC4 (as source or destination)
    and occurs between 7:00 (inclusive) and 21:00 (exclusive); otherwise 0.
    
    Features are: [normalized_hour, normalized_source, normalized_destination, normalized_data].
    """
    dataset = []
    base_time = datetime.datetime(2021, 1, 1, 0, 0, 0).timestamp()
    for hour in range(24):
        timestamp = base_time + hour * 3600
        current_hour = datetime.datetime.fromtimestamp(timestamp).hour
        for source in range(1, 5):
            for dest in range(1, 5):
                if source == dest:
                    continue
                data = random.randint(1, 100)
                # Label as anomaly if PLC4 is involved and hour is between 7 and 21.
                if (source == 4 or dest == 4) and (current_hour >= 7 and current_hour < 21):
                    label = 1
                else:
                    label = 0
                features = [current_hour / 23.0, (source - 1) / 3.0, (dest - 1) / 3.0, data / 100.0]
                dataset.append((features, label))
    return dataset


class AnomalyClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Binary classification output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def prepare_data(dataset):
    features = []
    labels = []
    for item in dataset:
        features.append(item[0])
        labels.append(item[1])
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return features, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_features, batch_labels in train_loader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)
            predictions = (outputs > 0.5).float()
            correct_train += (predictions == batch_labels).sum().item()
            total_train += batch_labels.size(0)
        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_features.size(0)
                predictions = (outputs > 0.5).float()
                correct_val += (predictions == batch_labels).sum().item()
                total_val += batch_labels.size(0)
        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")


####################################
# Main: Dataset Generation, Training and Simulation
####################################
def main():
    # ===== Generate Dataset =====
    dataset = generate_dataset()
    features, labels = prepare_data(dataset)
    num_samples = features.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)
    split = int(0.8 * num_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # ===== Train the Model =====
    input_dim = 4  # [normalized_hour, normalized_source, normalized_destination, normalized_data]
    model = AnomalyClassifier(input_dim)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    print("Training the classifier...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)

    # Save the model
    torch.save(model.state_dict(), "anomaly_model.pth")
    print("Model saved as anomaly_model.pth")

    # ===== Load the Model for Simulation =====
    model_loaded = AnomalyClassifier(input_dim)
    model_loaded.load_state_dict(torch.load("anomaly_model.pth"))
    model_loaded.eval()

    # Create a Traffic simulator instance that uses the classifier
    traffic = create_traffic_simulator(classifier=model_loaded)

    # ===== Run a Short Simulation (Demo) =====
    def event_callback(event):
        print("Event Callback:", event)

    print("\nSimulating 10 events with classification...\n")
    # Simulate a few events (instead of an infinite loop)
    for _ in range(10):
        active_plcs = [plc for plc in traffic.plcs.values() if plc.active]
        # If not enough active PLCs, activate all.
        if len(active_plcs) < 2:
            for plc in traffic.plcs.values():
                plc.active = True
            active_plcs = [plc for plc in traffic.plcs.values() if plc.active]
        sender = random.choice(active_plcs)
        receiver = random.choice(active_plcs)
        while sender.id == receiver.id:
            receiver = random.choice(active_plcs)
        data = random.randint(1, 100)
        timestamp = time.time()
        message = sender.send(receiver.id, data, timestamp)
        if message:
            receiver.receive(message, timestamp)
            current_hour = datetime.datetime.fromtimestamp(timestamp).hour
            features_event = [current_hour/23.0, (sender.id-1)/3.0, (receiver.id-1)/3.0, data/100.0]
            input_tensor = torch.tensor(features_event, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model_loaded(input_tensor).item()
            event_type = "anomaly" if prediction > 0.5 else "normal"
            event = {
                "event": event_type,
                "sender": sender.name,
                "receiver": receiver.name,
                "data": data,
                "timestamp": timestamp,
                "prediction": prediction
            }
            event_callback(event)
        time.sleep(1)

if __name__ == "__main__":
    main()
