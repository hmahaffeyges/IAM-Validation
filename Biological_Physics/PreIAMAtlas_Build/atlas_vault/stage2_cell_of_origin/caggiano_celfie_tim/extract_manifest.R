# Load the Locations.rda object (S4 class GRanges from Bioconductor)
load("IlluminaHumanMethylation450kanno.ilmn12.hg19/data/Locations.rda")
cat("Objects loaded:", ls(), "\n")

# Locations is a DataFrame with rows = CpG IDs, columns = chr, pos, strand
print(class(Locations))
print(head(Locations, 3))
print(dim(Locations))

# Convert to data.frame and write CSV
df <- as.data.frame(Locations)
df$cpg_id <- rownames(Locations)
df <- df[, c("cpg_id", "chr", "pos", "strand")]
write.csv(df, "hm450_hg19_manifest.csv", row.names = FALSE)
cat("Wrote", nrow(df), "rows to hm450_hg19_manifest.csv\n")
