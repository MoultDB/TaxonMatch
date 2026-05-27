# 🌿 TaxonMatch: Integrating Taxonomic Data from GBIF, NCBI, iNaturalist, PaleoDB, IUCN and other sources

<!-- [![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-red)](https://www.biorxiv.org/content/10.64898/2026.03.18.712418v1)

## 📄 Associated preprint
https://www.biorxiv.org/content/10.64898/2026.03.18.712418v1 -->

## 📌 Introduction
**TaxonMatch** is a machine learning–based framework for **large-scale taxonomic reconciliation**, designed to integrate and standardize heterogeneous biodiversity data from **GBIF**, **NCBI**, **iNaturalist**, **PaleoDB**, and **IUCN**. It enhances taxonomic consistency across biodiversity datasets, simplifies taxonomic name matching, and enables the generation of **phylogenetic trees** based on consolidated data.

## 🔬 Scope

TaxonMatch is designed as a core component for:
- biodiversity data integration
- trait extraction pipelines
- species distribution and ecological modelling workflows

### 🔍 Main Features
- 📥 **Download and clean taxonomic datasets** from GBIF, NCBI, iNaturalist, PaleoDB, and IUCN.
- 🔗 **Taxonomic name matching** to identify synonyms and discrepancies.
- 🌳 **Generate phylogenetic trees** from consolidated taxonomic data.
- 🦴 **Analyze fossil taxa** and identify their closest living relatives.
- 🌍 **Assign conservation status** to species using IUCN data.

---

## ⚙ Installation & Setup

### 🐍 Create a clean Conda environment (recommended)

```bash
conda create -n taxonmatch-env python=3.10 -y
conda activate taxonmatch-env

```

### 📦 Install TaxonMatch

```bash
# Install the latest version directly from GitHub:
pip install git+https://github.com/MicheleRoar/TaxonMatch.git
```

### 📓 (Optional) Run example notebooks
```bash
# Install Jupyter and optional plotting libraries:
pip install notebook ipython ipywidgets
jupyter notebook
```

### ⚠️ macOS + XGBoost fix (Apple Silicon only)
```bash
# If you're on macOS with M1/M2/M3 and XGBoost fails to load due to libomp.dylib
arch -arm64 brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
# To make this permanent, add to your shell config (~/.zshrc or ~/.bash_profile):
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
# Then run:
source ~/.zshrc  # or source ~/.bash_profile
```

## 🚀 Usage & Workflow

📝 Note: The code examples below are extracted from the notebooks available in the notebooks/ folder. For full examples, outputs, and extended explanations, please refer to those notebooks.

### 1️⃣ Download Taxonomic Datasets
```bash
import taxonmatch as txm

# Download GBIF and NCBI datasets
gbif_dataset = txm.download_gbif_taxonomy()
ncbi_dataset = txm.download_ncbi_taxonomy()
```

### 2️⃣ Taxonomic Name Matching (GBIF vs NCBI)
```bash
# Select a specific clade (example: Apidae)
gbif_apidae, ncbi_apidae = txm.select_taxonomic_clade("Apidae", gbif_dataset, ncbi_dataset)

# Load a pre-trained model
model = txm.load_xgb_model()

# Perform taxonomic matching
matched_df, unmatched_df, possible_typos_df = txm.match_dataset(gbif_apidae, ncbi_apidae, model, tree_generation = True)
```

### 3️⃣ Generate a Phylogenetic Tree
```bash
tree = txm.generate_taxonomic_tree(matched_df, unmatched_df)
txm.print_tree(tree, root_name="Apidae")
txm.save_tree(tree, "taxon_tree.txt")
```

### 4️⃣ Identify Closest Living Relatives for Fossil Species
```bash
# Retrieve the dataset ID associated with a fossil species
dataset_id = txm.get_dataset_from_species("Ristoria pliocaenica")

# Download taxonomic datasets
paleodb_dataset = txm.download_gbif_taxonomy(source=dataset_id)
ncbi_dataset = txm.download_ncbi_taxonomy(source="ncbi")

# Find the closest matching clades for the species
col_parents, ncbi_parents = txm.select_closest_common_clade(
    "Ristoria pliocaenica",
    paleodb_dataset,
    ncbi_dataset
)
```

### 5️⃣ Add Conservation Status Using IUCN Data
```bash
df_with_iucn_status = txm.add_iucn_status_column(matched_df)
df_with_iucn_status[df_with_iucn_status.iucnRedListCategory.isin(['ENDANGERED', 'CRITICALLY_ENDANGERED', 'VULNERABLE'])]
```

## 📈 Visualization
### Distribution of Conservation Status
```bash
txm.plot_conservation_statuses(df_with_iucn_status)
```

## 🤝 Contributing & Maintenance

If you want to contribute to **TaxonMatch**:

- **Open an issue** to report bugs or suggest improvements.
- **Create a pull request** with a clear description of your changes.
- **Follow the coding guidelines** and ensure all modifications are tested.

<!--For questions or support, contact michele.leone@unil.ch.-->


## License

This project is licensed under the MIT License.




