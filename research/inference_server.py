import sys
import os
import json
import warnings
from http.server import HTTPServer, BaseHTTPRequestHandler

warnings.filterwarnings("ignore")

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research.inference.predict import PredictionPipeline

print("[Inference Daemon] Preloading PyTorch, Qiskit & Scikit-Learn models...")
pipeline = PredictionPipeline()
print("[Inference Daemon] All models loaded into memory. Ready for ultra-fast inference!")

class InferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "model": "MobileNetV2+VQC+Hybrid"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                image_path = data.get("image_path")
                if not image_path or not os.path.exists(image_path):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "error": "Image file does not exist"}).encode("utf-8"))
                    return

                # Fast warm inference (70ms)
                result = pipeline.predict(image_path)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy standard request logs
        pass

def run(port=5001):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, InferenceHandler)
    print(f"[Inference Daemon] Listening on http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Inference Daemon] Shutting down.")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    run(port)
