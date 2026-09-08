# powerbi

Projet Power BI enregistré au format **PBIP** (Power BI Project) plutôt qu'en `.pbix` binaire — activer *File > Save as > Power BI project (.pbip)* dans Power BI Desktop.

Power BI Desktop génère ici une structure texte (JSON / TMDL), diffable et review-able dans les pull requests :

```
powerbi/
├── <ProjectName>.pbip                # fichier pointeur, à ouvrir dans Power BI Desktop
├── <ProjectName>.Report/             # définition du rapport (pages, visuels)
│   ├── .platform
│   ├── StaticResources/              # images importées directement dans le rapport
│   ├── definition/                   # PBIR : report.json, pages/, visuals/ en JSON
│   └── .pbi/                         # cache & paramètres locaux — gitignored
└── <ProjectName>.SemanticModel/      # modèle sémantique (ex-Dataset)
    ├── .platform
    ├── definition/                    # TMDL : tables/, relationships.tmdl, expressions.tmdl (Power Query M)
    ├── diagramLayout.json
    └── .pbi/                          # cache & paramètres locaux — gitignored
```

## Points d'attention

- **Le code Power Query M** vit désormais dans `<ProjectName>.SemanticModel/definition/**.tmdl` (blocs `source = ...`), pas dans des fichiers `.pq`/`.m` séparés.
- **Les dossiers `.pbi/`** (cache, `localSettings.json`) sont spécifiques à la machine/l'utilisateur et sont exclus du versioning (voir [.gitignore](../../.gitignore)).
- **Un seul projet PBIP ouvert à la fois par personne** évite les conflits de merge sur `definition/` — traiter les branches comme pour du code.
- Le dossier `../assets/` reste indépendant : il contient les sources éditables (`.pptx`, `.fig`) et exports PNG *avant* import dans le rapport, alors que `StaticResources/` ci-dessus contient les images déjà importées par Power BI Desktop.
