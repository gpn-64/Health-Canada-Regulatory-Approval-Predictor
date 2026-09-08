# tableau

Classeur Tableau du dashboard.

```
tableau/
├── project.twbx     # classeur packagé (données + mise en forme) — binaire
└── project.twb      # classeur non packagé (référence des données en externe) — XML texte
```

- `.twbx` : à privilégier pour le partage/publication (auto-suffisant, données incluses).
- `.twb` : préférable si versionné dans git — c'est du XML, donc diffable, contrairement au `.twbx`.

> Dossier non utilisé si le dashboard est fait sous Power BI (voir [../powerbi/](../powerbi/)) ou en app Streamlit/Dash — à supprimer dans ce cas.
