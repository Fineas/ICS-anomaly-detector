from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS  # Import the extension
import threading
import json
from queue import Queue
from simulator import create_traffic_simulator

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global event queue to store simulation events
event_queue = Queue()

def event_callback(event):
    event_queue.put(event)

# Create our traffic simulator instance
traffic = create_traffic_simulator()

def simulation_thread():
    traffic.simulate(delay=1, event_callback=event_callback)

sim_thread = threading.Thread(target=simulation_thread, daemon=True)
sim_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            event = event_queue.get()
            yield f"data: {json.dumps(event)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/trigger_anomaly', methods=['POST'])
def trigger_anomaly():
    traffic.trigger_anomaly(event_callback=event_callback)
    return jsonify({"status": "Anomaly triggered"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
