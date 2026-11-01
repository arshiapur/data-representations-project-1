"""Save actual command/stdout/stderr/exit-status transcript of verification."""
import os
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parent
env = dict(os.environ, MPLCONFIGDIR='/tmp/stage3-mpl', XDG_CACHE_HOME='/tmp/stage3-cache')
commands = [([sys.executable,'-m','unittest','discover','-s','tests','-v'],root,'tests.txt'),
            ([sys.executable,'../code/mds.py','--input','../data/Iris.csv','--q','2','--dimensions','2'],root/'results','Iris_q2_2D_diagnostics.json'),
            ([sys.executable,'../code/mds.py','--input','../data/Iris.csv','--q','2','--dimensions','3'],root/'results','Iris_q2_3D_diagnostics.json'),
            ([sys.executable,'tests/verify_iris.py'],root,'iris_alignment.json')]
with (root/'TASK_EXECUTION_TRANSCRIPT.txt').open('w') as transcript:
    transcript.write('Actual verification subprocess transcript for this Stage 3 task.\nThis records commands and complete process outputs, not private model reasoning or a platform chat export.\nNo separate task URL is available through the current tools.\n\n')
    for command,cwd,name in commands:
        p = subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True)
        transcript.write(f'CWD: {cwd}\nCOMMAND: {command!r}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}\nEXIT STATUS: {p.returncode}\n\n')
        (root/'results'/name).write_text(p.stdout if name.endswith('.json') else p.stdout+p.stderr)
        print(name, 'exit', p.returncode)
        if p.returncode:
            raise SystemExit(p.returncode)
