from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from ericbacktest import CryptoBacktester

class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(length) or b'{}') if length else {}
        if self.path == '/api/parse-strategy':
            strategy = data.get('strategy', '')
            if not strategy:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'strategy required'}).encode())
                return
            try:
                bt = CryptoBacktester()
                rules = bt.parse_strategy_with_ai(strategy)
                self._set_headers(200)
                self.wfile.write(json.dumps({'strategy_rules': rules}).encode())
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        elif self.path == '/api/run-backtest':
            rules = data.get('strategy_rules')
            excel = data.get('excel_names', [])
            initial = float(data.get('initial_capital', 100000))
            if not rules or not excel:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'strategy_rules and excel_names required'}).encode())
                return
            try:
                bt = CryptoBacktester()
                bt.fetch_data_from_pinecone(excel)
                bt.run_custom_strategy(rules, initial_capital=initial)
                bt.generate_report()
                self._set_headers(200)
                self.wfile.write(json.dumps({'performance': bt.performance_metrics}).encode())
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'not found'}).encode())


def run(addr='0.0.0.0', port=8000):
    server = HTTPServer((addr, port), RequestHandler)
    print(f'Server running on http://{addr}:{port}')
    server.serve_forever()

if __name__ == '__main__':
    run()
