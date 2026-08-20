# Harmony Grid Prototype

Prototype Python pour analyser un fichier audio local et produire une première grille harmonique lisible façon jazz/pop.

Ce prototype est volontairement simple. Il sert à valider la chaîne :

1. import audio local ;
2. estimation du tempo ;
3. extraction chroma ;
4. reconnaissance d’accords par templates ;
5. quantification sur les mesures ;
6. export Markdown et JSON.

Il ne fait pas d’analyse directe d’un flux Spotify. Un lien Spotify peut être utilisé plus tard pour récupérer les métadonnées, mais l’analyse audio doit venir d’un fichier local fourni légalement par l’utilisateur.

## Installation

Il faut Python 3.11 ou 3.12 et FFmpeg.

```bash
cd harmony_grid_prototype
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
python analyze_chords.py ~/Music/morceau.mp3 --out grille.md --json analyse.json
```

Options utiles :

```bash
python analyze_chords.py morceau.wav \
  --out grille.md \
  --json analyse.json \
  --beats-per-bar 4 \
  --min-segment 0.45 \
  --tempo 80.5 \
  --fallback-bpm 120 \
  --key "E minor"
```

`--fallback-bpm` est utilisé si la détection automatique du tempo échoue, par exemple sur un fichier très statique, un pad, ou un test synthétique sans percussion.

`--tempo` permet de forcer le tempo si l’algorithme détecte un tempo double ou moitié. Par exemple, si le script affiche 161 BPM alors que le morceau est plutôt à 80,5 BPM, relance avec `--tempo 80.5`.

`--key` permet de corriger manuellement la tonalité affichée si l’algorithme confond une tonalité majeure avec sa relative mineure, ce qui arrive fréquemment dans cette première version.

Si la tonalité forcée est mineure, par exemple `--key "E minor"` ou `--key "Mi mineur"`, le prototype applique aussi une correction simple sur la tonique : un accord `E` détecté comme majeur sera affiché `Em`. Cette correction reste volontairement prudente et ne remplace pas une vraie reconnaissance harmonique contextuelle.

## Sortie Markdown

Exemple :

```text
| C          | %          | Am         | F G        |
```

Le symbole `%` signifie que l’accord de la mesure précédente est répété.

## Sortie JSON

Le JSON contient :

- tempo estimé ;
- métrique supposée ;
- tonalité probable ;
- segments d’accords avec timestamps ;
- grille par mesures ;
- confiance par accord.

## Limites connues

- Les accords enrichis sont souvent simplifiés.
- Les renversements ne sont pas encore détectés de manière robuste.
- Le premier temps peut être mal placé.
- Le tempo peut être détecté en double ou moitié.
- Les morceaux très denses, très réverbérés ou bruitistes peuvent produire beaucoup d’incertitude.

## Prochaines améliorations

- Ajouter correction interactive du premier temps.
- Ajouter transposition.
- Ajouter export MusicXML/MIDI.
- Ajouter meilleur modèle de reconnaissance d’accords.
- Ajouter interface macOS SwiftUI.
