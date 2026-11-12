"""
Live in-place executor for Semantic_Search_NLP_Project.ipynb.
Executes all code cells in sequence in a persistent environment, captures real stdout/stderr,
converts DataFrames to HTML/text display outputs, renders matplotlib plots to base64 images,
and updates the notebook in-place.
"""

import sys
import io
import time
import base64
import nbformat as nbf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import pandas as pd

def run_notebook(nb_path="Semantic_Search_NLP_Project.ipynb"):
    print(f"=== DÉBUT DE L'EXÉCUTION RÉELLE DU NOTEBOOK : {nb_path} ===")
    t_start_total = time.time()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # Execution scope (persistent namespace across all cells)
    scope = {
        "__name__": "__main__",
        "__doc__": None,
        "__package__": None,
    }

    current_cell_outputs = []

    # Custom display hook for Jupyter emulation
    def custom_display(*args, **kwargs):
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                current_cell_outputs.append(nbf.v4.new_output(
                    output_type="display_data",
                    data={
                        "text/html": arg.to_html(),
                        "text/plain": arg.to_string()
                    },
                    metadata={}
                ))
            else:
                current_cell_outputs.append(nbf.v4.new_output(
                    output_type="display_data",
                    data={
                        "text/plain": repr(arg)
                    },
                    metadata={}
                ))

    # Custom plt.show hook to capture matplotlib figures as base64 display_data
    def custom_plt_show(*args, **kwargs):
        fig = plt.gcf()
        if fig and fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)
            b64_img = base64.b64encode(buf.read()).decode("utf-8")
            current_cell_outputs.append(nbf.v4.new_output(
                output_type="display_data",
                data={
                    "image/png": b64_img,
                    "text/plain": f"<Figure size {fig.get_size_inches()[0]*100:.0f}x{fig.get_size_inches()[1]*100:.0f} with {len(fig.get_axes())} Axes>"
                },
                metadata={}
            ))
            plt.close(fig)

    scope["display"] = custom_display

    execution_counter = 0

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            execution_counter += 1
            cell.execution_count = execution_counter
            current_cell_outputs = []
            
            # Monkeypatch plt.show for this execution
            plt.show = custom_plt_show

            source = cell.source
            print(f"\n[Cellule {idx} | Code #{execution_counter}] Exécution en cours...")
            t0 = time.time()

            # Redirect stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                # Execute cell code
                exec(source, scope)
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print(f"❌ ERREUR lors de l'exécution de la cellule {idx} :\n{e}")
                raise e
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Check if any active plot was not closed
            fig = plt.gcf()
            if fig and fig.get_axes():
                custom_plt_show()

            # Process captured stdout
            stdout_val = stdout_capture.getvalue()
            if stdout_val:
                current_cell_outputs.insert(0, nbf.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=stdout_val
                ))

            # Process captured stderr (ignore harmless matplotlib warnings)
            stderr_val = stderr_capture.getvalue()
            if stderr_val and "UserWarning" not in stderr_val:
                current_cell_outputs.append(nbf.v4.new_output(
                    output_type="stream",
                    name="stderr",
                    text=stderr_val
                ))

            cell.outputs = current_cell_outputs
            dt = time.time() - t0
            print(f"✓ Cellule {idx} exécutée avec succès en {dt:.2f}s (Sorties produites : {len(cell.outputs)})")
            if stdout_val:
                first_lines = "\n".join("    | " + line for line in stdout_val.strip().split("\n")[:3])
                print(f"    Sortie standard :\n{first_lines}")

    # Save executed notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    total_time = time.time() - t_start_total
    print(f"\n🎉 TOUTES LES CELLULES ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS EN {total_time:.1f} SECONDES !")
    print(f"Fichier mis à jour sur disque : {nb_path}")

if __name__ == "__main__":
    run_notebook()
