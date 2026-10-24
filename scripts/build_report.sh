#!/bin/zsh
# Build report/report.md -> report/report.docx (Word, native equations) and report/report.pdf
# (pandoc HTML with MathML, printed by headless Chrome; no LaTeX needed).
set -euo pipefail
cd "$(dirname "$0")/../report"
PANDOC=/opt/homebrew/bin/pandoc   # native arm64 build (the Anaconda pandoc is x86-only)
"$PANDOC" report.md -o report.docx --resource-path=. --toc --toc-depth=1 --reference-doc=reference.docx
"$PANDOC" report.md -o report.html --standalone --embed-resources --math-method=mathml --css=print.css \
       --resource-path=. --toc --toc-depth=1 --metadata=lang:en-GB
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
          --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html" 2>/dev/null
ls -la report.docx report.pdf report.html
