import random
import time

# ---------------------------------------------
# Simulation of Programmable Logic Controller
# ---------------------------------------------
class PLC:
    def __init__(self, id, name=None, active=True):
        self.id = id
        self.name = name or f"PLC-{id}"
        self.active = active  # Determines if the PLC participates in traffic.
        self.log = []  # Keep a log of received messages

    def send(self, dest_id, data):
        if not self.active:
            print(f"{self.name} is inactive and cannot send messages.")
            return None
        message = Protocol(source=self.id, destination=dest_id, data=data)
        print(f"{self.name} sends message to PLC-{dest_id}: {message}")
        return message

    def receive(self, message):
        if not self.active:
            print(f"{self.name} is inactive and ignores the message: {message}")
            return
        print(f"{self.name} received message: {message}")
        self.log.append(message)

    def __repr__(self):
        return f"<{self.name} {'Active' if self.active else 'Inactive'}>"


# ---------------------------------------
# Simulation of custom Network Protocol
# ---------------------------------------
class Protocol:
    def __init__(self, source, destination, data):
        self.source = source
        self.destination = destination
        self.data = data

    def __str__(self):
        return f"[Source: {self.source} -> Destination: {self.destination} | Data: {self.data}]"

    def __repr__(self):
        return self.__str__()


# --------------------
# Traffic Simulation
# --------------------
class Traffic:
    def __init__(self, plc_list):
        # Dictionary for easier lookup by id.
        self.plcs = {plc.id: plc for plc in plc_list}

    def simulate(self, delay=1, event_callback=None):
        """
        Run simulation in an infinite loop. For every communication event, call event_callback
        with a dictionary containing event details.
        """
        try:
            while True:
                active_plcs = [plc for plc in self.plcs.values() if plc.active]
                if len(active_plcs) < 2:
                    event = {"event": "waiting", "message": "Not enough active PLCs to simulate communication."}
                    if event_callback:
                        event_callback(event)
                    time.sleep(delay)
                    continue

                # Randomly select a sender and a receiver (ensuring they are not the same)
                sender = random.choice(active_plcs)
                receiver = random.choice(active_plcs)
                while sender.id == receiver.id:
                    receiver = random.choice(active_plcs)

                data = random.randint(1, 100)
                message = sender.send(receiver.id, data)
                if message:
                    receiver.receive(message)

                    # Trigger event
                    event = {
                        "event": "message",
                        "sender": sender.name,
                        "receiver": receiver.name,
                        "data": data
                    }
                    if event_callback:
                        event_callback(event)
                time.sleep(delay)
        except KeyboardInterrupt:
            event = {"event": "stopped", "message": "Simulation stopped by user."}
            if event_callback:
                event_callback(event)

    def trigger_anomaly(self, event_callback=None):
        """
        Activate PLC-4 to simulate an anomalous behavior.
        """
        plc4 = self.plcs.get(4)
        if plc4:
            plc4.active = True  # Activate the fourth PLC.
            print(f"\n--- Anomaly Triggered: {plc4.name} is now active and participating in the network! ---\n")
            event = {"event": "anomaly", "message": f"Anomaly Triggered: {plc4.name} is now active."}
            if event_callback:
                event_callback(event)
        else:
            print("PLC-4 not found in the network.")
            event = {"event": "error", "message": "PLC-4 not found in the network."}
            if event_callback:
                event_callback(event)


def create_traffic_simulator():
    """
    Helper function to create and return a Traffic simulator with four PLC devices.
    PLC-1, PLC-2, and PLC-3 are active by default while PLC-4 is inactive.
    """
    plc1 = PLC(id=1, name="PLC-1", active=True)
    plc2 = PLC(id=2, name="PLC-2", active=True)
    plc3 = PLC(id=3, name="PLC-3", active=True)
    plc4 = PLC(id=4, name="PLC-4", active=False)
    traffic = Traffic([plc1, plc2, plc3, plc4])
    return traffic