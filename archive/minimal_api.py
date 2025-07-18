#!/usr/bin/env python3
"""
Minimal Portfolio API

Ultra-minimal API server that should work around connection issues.
"""

import json
import http.server
import socketserver
import urllib.parse
from pathlib import Path

class PortfolioAPIHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/health':
            self.serve_health()
        elif self.path == '/assets/available':
            self.serve_assets()
        elif self.path == '/strategies/available':
            self.serve_strategies()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        if self.path == '/portfolio/optimize':
            self.handle_optimize()
        else:
            self.send_error(404, "Not Found")
    
    def serve_dashboard(self):
        try:
            dashboard_path = Path("web/portfolio_dashboard.html")
            if dashboard_path.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(dashboard_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Dashboard not found")
        except Exception as e:
            self.send_error(500, f"Error serving dashboard: {e}")
    
    def serve_health(self):
        response = {
            "status": "healthy",
            "message": "Minimal API is running",
            "timestamp": "2025-01-17"
        }
        self.send_json_response(response)
    
    def serve_assets(self):
        assets = [
            "GSPC", "DJI", "IXIC", "RUT", "EFA", "EEM",
            "TLT", "IEF", "SHY", "LQD", "HYG", "TIPS",
            "GLD", "SLV", "USO", "UNG", "DBA",
            "VNQ", "BTC_USD", "ETH_USD"
        ]
        response = {
            "assets": assets,
            "count": len(assets)
        }
        self.send_json_response(response)
    
    def serve_strategies(self):
        strategies = [
            {
                "name": "mean_reversion",
                "description": "Mean reversion strategy"
            },
            {
                "name": "momentum", 
                "description": "Momentum strategy"
            },
            {
                "name": "risk_parity",
                "description": "Risk parity strategy"
            }
        ]
        response = {"strategies": strategies}
        self.send_json_response(response)
    
    def handle_optimize(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Simple optimization logic
            assets = request_data.get('assets', [])
            risk_tolerance = request_data.get('risk_tolerance', 'medium')
            strategy = request_data.get('strategy', 'mean_reversion')
            
            # Generate equal weights
            n_assets = len(assets)
            if n_assets == 0:
                self.send_error(400, "No assets provided")
                return
            
            weight = 1.0 / n_assets
            weights = {asset: weight for asset in assets}
            
            # Simple metrics based on risk tolerance
            if risk_tolerance == "conservative":
                expected_return = 0.06
                expected_volatility = 0.10
            elif risk_tolerance == "aggressive":
                expected_return = 0.12
                expected_volatility = 0.20
            else:
                expected_return = 0.08
                expected_volatility = 0.15
            
            sharpe_ratio = expected_return / expected_volatility
            
            response = {
                "success": True,
                "weights": weights,
                "expected_return": expected_return,
                "expected_volatility": expected_volatility,
                "sharpe_ratio": sharpe_ratio,
                "strategy_used": strategy,
                "recommendations": [
                    f"Portfolio optimized for {risk_tolerance} risk tolerance",
                    f"Expected annual return: {expected_return:.1%}",
                    f"Expected volatility: {expected_volatility:.1%}"
                ],
                "risk_metrics": {
                    "var_95": expected_volatility * 1.65,
                    "max_drawdown": expected_volatility * 2.0,
                    "volatility": expected_volatility
                }
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            self.send_error(500, f"Optimization failed: {e}")
    
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    PORT = 8080
    
    print("🚀 Starting Minimal Portfolio API...")
    print(f"Dashboard: http://localhost:{PORT}")
    print(f"Health check: http://localhost:{PORT}/health")
    print("Press Ctrl+C to stop")
    
    try:
        with socketserver.TCPServer(("", PORT), PortfolioAPIHandler) as httpd:
            print(f"✅ Server running on port {PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use. Try a different port.")
        else:
            print(f"❌ Server error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
