# King_County_House_Sale_Price
Claude: 
Q: How typical data science project should be structured?
A: 
project/
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and transformed data
│   └── external/     # Third-party data sources
├── notebooks/        # Jupyter notebooks for exploration
├── src/              # Source code (importable modules)
│   ├── data/         # Data ingestion & processing scripts
│   ├── features/     # Feature engineering
│   ├── models/       # Model training & evaluation
│   └── visualization/
├── models/           # Saved model artifacts
├── reports/          # Generated reports, figures
├── tests/            # Unit and integration tests
├── requirements.txt  # or pyproject.toml / environment.yml
├── .env              # Secrets (never commit this!)
├── .gitignore
└── README.md
