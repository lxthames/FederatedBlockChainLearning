"""Build the anonymous draft; final readiness is checked separately."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parents[1]


def main():
    subprocess.run([sys.executable,'-m','scripts.research.paper_artifacts'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','scripts.data_preparation.extension_artifacts'],cwd=ROOT,check=True)
    build=PAPER/'build'
    build.mkdir(exist_ok=True)
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex'],
              ['bibtex','build/main'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex']]
    for index,command in enumerate(commands):
        with (build/f'build-{index}.log').open('w',encoding='utf-8') as out:
            result=subprocess.run(command,cwd=PAPER,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'build failed: {command}; see build/build-{index}.log')
    log=(build/'main.log').read_text(encoding='utf-8',errors='replace')
    if 'There were undefined references' in log or 'There were undefined citations' in log:
        raise RuntimeError('unresolved references')
    shutil.copy2(build/'main.pdf',PAPER/'ACM_TRUST_2027_draft.pdf')
    import fitz
    document=fitz.open(PAPER/'ACM_TRUST_2027_draft.pdf')
    pages=[]
    for index,page in enumerate(document):
        page.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(build/f'page-{index+1}.png')
        pages.append({'page':index+1,'width':page.rect.width,'height':page.rect.height})
    report={'pages':pages,'page_count':len(document),'draft':True,
            'overfull_boxes':log.count('Overfull \\hbox'),
            'undefined_references':False,'visual_qa':'pending inspection of rendered pages'}
    (PAPER/'build-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
