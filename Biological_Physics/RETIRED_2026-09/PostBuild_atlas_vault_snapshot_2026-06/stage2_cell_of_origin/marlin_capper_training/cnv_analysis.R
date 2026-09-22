#--------------------------------------------------------------------
# #CNV analysis
#
#
# Martin Sill
# m.sill@dkfz.de                                                                  
# 
# 2019-02-11 UTC
#--------------------------------------------------------------------                   
#options(max.print = 1000)
#options(stringsAsFactors = FALSE)
#options(scipen = 10)
#rm(list=ls())
library(minfi)
library(conumee)
library(GEOquery)

# load conumee annoatation object
load(file.path("CNV_data","CNanalysis4_conumee_ANNO.vh20150715.RData"))
# load conumee reference male
load(file.path("CNV_data","CNanalysis4_conumee_REF-M.vh20150715.RData"))
# load conumee reference female
load(file.path("CNV_data","CNanalysis4_conumee_REF-F.vh20150715.RData"))

# get sample annotation from GEO
gse <- getGEO("GSE90496", GSEMatrix=TRUE, getGPL=FALSE)
anno <- pData(gse$GSE90496_series_matrix.txt.gz)

# read raw data downloaded from GEO and extracted in GSE90496_RAW
fname <- paste0(gsub("_Grn.*","",gsub(".*suppl/","",anno$supplementary_file)))
filepath <- file.path("GSE90496_RAW",fname)

# read just the first sample
RGset <- read.metharray(filepath[1],verbose=TRUE)

# we perform no normalization before CNV analysis
Mset <- preprocessRaw(RGset)

# get CN data and perform conumee analysis
cndata <- CNV.load(Mset)
x <- CNV.fit(cndata, refM.data, annoXY)
x <- CNV.bin(x)
x <- CNV.detail(x)
x <- CNV.segment(x)

# plot 
CNV.genomeplot(x, chrY=TRUE)
