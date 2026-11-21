
#!/bin/bash

# Interrompe lo script in caso di errore
set -e

echo "=== Git Update ==="

# Aggiunge tutte le modifiche tracciabili e non ignorate
git add -A

# Crea il commit
git commit -m "Update"

# Invia le modifiche a GitHub
git push

echo "=== Push completato! ==="
