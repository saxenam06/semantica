#!/usr/bin/env python3
"""
Simple HTTP server to test documentation locally
Run: python test-docs.py
Then visit: http://localhost:8000/preview.html
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000

class DocsHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for local testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        super().end_headers()

def main():
    # Change to project root directory
    os.chdir(Path(__file__).parent)
    
    Handler = DocsHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"""
╔═══════════════════════════════════════════════════════╗
║     Semantica Documentation Test Server               ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║  Server running at:                                  ║
║  http://localhost:{PORT}                                    ║
║                                                       ║
║  Available pages:                                     ║
║  • http://localhost:{PORT}/preview.html              ║
║  • http://localhost:{PORT}/docs/index.md             ║
║  • http://localhost:{PORT}/docs/installation.md      ║
║  • http://localhost:{PORT}/docs/quickstart.md         ║
║  • http://localhost:{PORT}/docs/api.md               ║
║  • http://localhost:{PORT}/docs/examples.md          ║
║                                                       ║
║  Press Ctrl+C to stop the server                     ║
╚═══════════════════════════════════════════════════════╝
        """)
        
        # Open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}/docs/preview.html')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped. Goodbye!")

if __name__ == "__main__":
    main()

