from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def guess_type(self, path):
        path_str = str(path)
        if '.wasm' in path_str:
            return 'application/wasm'
        return super().guess_type(path)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        super().end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), CORSHTTPRequestHandler)
    print('Server running at http://0.0.0.0:8000 (use your machine\'s LAN IP to access from other devices)')
    server.serve_forever()
