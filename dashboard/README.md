# dashboard

Placer ici le livrable de dashboard final. Ne garder que le(s) dossier(s) correspondant à l'outil réellement utilisé, supprimer les autres :

- `powerbi/` — projet Power BI au format **PBIP** (voir [powerbi/README.md](powerbi/README.md))
- `tableau/` — classeur Tableau (voir [tableau/README.md](tableau/README.md))
- ou le code d'une app Streamlit/Dash directement à la racine de `dashboard/`

## powerbi/

Projet Power BI Desktop enregistré en `.pbip` : structure texte (JSON/TMDL) versionnable, avec un dossier `.pbi/` local exclu du git. Détails dans [powerbi/README.md](powerbi/README.md).

## tableau/

Classeur Tableau (`.twbx` ou `.twb`). Détails dans [tableau/README.md](tableau/README.md).

## assets/

Ressources runtime importées dans le dashboard :

- `backgrounds/` — PNG utilisés comme fond de page (Power BI). Les fichiers éditables source (`.pptx`, `.fig`) qui ont généré ces PNG vont dans `backgrounds/source/`.
- `icons/` — icônes custom (KPI, navigation).
- `logo/` — logo(s) du projet/de l'organisation.
