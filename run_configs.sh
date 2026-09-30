#!/bin/bash
# Esegue main.py su tutti i config .yaml di una cartella (anche nelle sottocartelle),
# N alla volta in parallelo, con barra di progresso.
#
# Uso: ./run_configs.sh <cartella_config> [n_paralleli]
# Es.: ./run_configs.sh FeSCAE_configs_AE_noSearch 4

CONFIG_DIR="$1"
N_JOBS="${2:-1}"
export PYTHON_ENV="/home/edofroses/.conda/envs/genomics_env/bin/python"

if [ -z "$CONFIG_DIR" ]; then
    echo "Uso: $0 <cartella_config> [n_paralleli]"
    exit 1
fi

# I path nei config sono relativi alla root del progetto
cd "$(dirname "$0")"

mkdir -p run_logs

export TOTAL=$(find "$CONFIG_DIR" -name "*.yaml" | wc -l | tr -d ' ')
export DONE_FILE="run_logs/.done"
: > "$DONE_FILE"

echo "Config trovati: $TOTAL - job in parallelo: $N_JOBS"

run_one() {
    name=$(basename "$1" .yaml)
    if "$PYTHON_ENV" main.py --config "$1" > "run_logs/$name.log" 2>&1; then
        status="OK  "
    else
        status="FAIL"
    fi

    # Aggiorna il contatore e stampa la barra
    echo "$name" >> "$DONE_FILE"
    done=$(grep -n -x "$name" "$DONE_FILE" | cut -d: -f1)
    width=30
    filled=$(( done * width / TOTAL ))
    bar=$(printf "%${filled}s" | tr " " "#")$(printf "%$(( width - filled ))s" | tr " " "-")
    echo "[$bar] $done/$TOTAL ($(( done * 100 / TOTAL ))%) [$status] $name"
}
export -f run_one

find "$CONFIG_DIR" -name "*.yaml" | sort | xargs -P "$N_JOBS" -I {} bash -c 'run_one "$1"' _ {}

rm -f "$DONE_FILE"
echo "Finito. Log in run_logs/"
