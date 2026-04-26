DNA methylation-based classification of central nervous system tumours
================

Collection of R-scripts used to perform DNA-methylation data analysis presented in [DNA methylation-based classification of central nervous system tumours](https://www.nature.com/articles/nature26000). The raw data is publicly available at NCBI GEO under Accession number [GSE90496](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90496).

#### Preprocessing and Normalization

``` r
preprocessing.R
```

Reads raw data, performs normalization, basic filtering and batch effect adjustment between Frozen and FFPE samples. Normalized and batch adjusted as well as unadjusted data is stored in `./results`

#### Unsupervised tSNE analysis

``` r
tsne.R
```

Performs non-linear dimension reduction on preprocessed DNA-methylation data.

#### Classifier training and cross-validation

``` r
training.R
```

Trains the Random Forest classifier on the complete data set and stores the final classifier in `./results`

``` r
cross_validation.R
```

Performs nested cross-validation and stores the results in `./CV`

``` r
calibration.R
```

Evaluates the results of the cross-validation and fits a calibration model that is stored in `./results` and compiles a final report showing classifier performance metrics `CVresults.html`.

#### Tumor purity estimation

``` r
purity.Rmd
purity.html
```

Example how TCGA 450k methylation data and ABSOLUTE tumor purity estimates can be used to train a Random Forest to predict tumor purity.

#### Copy number analysis

``` r
cnv_analysis.R
```

Example how the [conumee](http://bioconductor.org/packages/release/bioc/html/conumee.html) Bioconductor is applied to perform copy number variation analysis. Note, to get the reference objects stored in `./CNV_data`, [Git large file storage](https://git-lfs.github.com/) needs to be installed before cloning the repository.
