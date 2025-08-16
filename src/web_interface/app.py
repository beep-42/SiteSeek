#!/usr/bin/env python3
import json

import numpy
from flask import Flask, request, Response, stream_with_context, render_template
import subprocess, shlex
from pathlib import Path
from timeit import default_timer as timer


def find_db_files(root_dir):
    root = Path(root_dir)
    return [p for p in root.rglob('*db*') if p.is_file()]

app = Flask(__name__)

def generate_search_command(database, pdb_id, residues):
    print(database, pdb_id, residues)
    residues = ','.join([x.replace(':', '') for x in residues])
    cmd = ["python3", "cli.py", "search", database, pdb_id, residues, '--output-json']
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        yield line
    proc.wait()
    yield f"\n[Search exited with code {proc.returncode}]\n"

def generate_search_function(database, pdb_id, residues):
    def convert_to_json(results) -> str:
        """Converts list of Results to a dict of dicts (index to object) and subsequently serializes it using json.dumps."""

        obj = {i: r.__dict__ for i, r in enumerate(results)}
        for one in obj:
            single_result = obj[one]
            for field in single_result:
                if isinstance(single_result[field], numpy.ndarray):
                    single_result[field] = single_result[field].tolist()

        return json.dumps(obj)

    site = [x.replace(':', '') for x in residues]

    try:
        t = timer()
        results = yield from app.config['database'].search(pdb_id, site=site, k_mer_similarity_threshold=11, lr=.9,
                            skip_clustering=False, skip_icp=False, ransac_min=15)
        yield f'Done in {timer() - t :.{2}f} seconds. Found {len(results)} results.\n'

        # output.write(json.dumps(results))

        yield convert_to_json(sorted(results, key=lambda x: x.score, reverse=True))

    except Exception as e:
        yield f"Failed! Exception: {e}"

@app.route("/")
def index():
    # db_files = find_db_files(Path(app.root_path).parent)
    return render_template("selection-ngl.html", db_files=[app.config['database_name']])

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    database = data.get("database")
    pdb_id = data.get("pdb_id")
    residues = data.get("residues", [])
    if not pdb_id or not residues or not database:
        return "Invalid request\n", 400

    return Response(
        stream_with_context(generate_search_function(database, pdb_id, residues)),
        mimetype="text/plain"
    )

if __name__ == "__main__":
    app.run(debug=True)