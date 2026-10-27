"""
Convenience entry point for the MaternalCare AI Flask Web Application.
Allows running the web app directly from the project root directory.
"""
import os
import sys

# Ensure webapp directory is in path and working directory is set
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(ROOT_DIR, "webapp")

if WEBAPP_DIR not in sys.path:
    sys.path.insert(0, WEBAPP_DIR)


from app import app, init_db

if __name__ == "__main__":
    init_db()
    print("\n  MaternalCare AI — Flask Web Application")
    print("  Running at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, use_reloader=False, port=5000)

