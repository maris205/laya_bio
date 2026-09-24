"""Rebuild paper figures, tables and PDF from saved results; no model execution."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
for script in ('gen_tables.py','gen_sequence_diagnostics.py','gen_candidate_order.py'):
    subprocess.run([sys.executable,str(HERE/'figures'/script)],cwd=HERE,check=True)
for command,log in [(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],'compile_pass1.log'),
                    (['bibtex','main'],'bibtex.log'),
                    (['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],'compile_pass2.log'),
                    (['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],'compile.log')]:
    with (HERE/log).open('w') as f:subprocess.run(command,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['pdftotext','-layout','main.pdf','main.txt'],cwd=HERE,check=True)
print('Built '+str(HERE/'main.pdf'))
