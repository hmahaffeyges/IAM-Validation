#!/usr/bin/env Rscript
# DEVELOPMENT - not commissioned (DEV-EPIC-V2-01, 2026-10-04). One EPIC v2 IDAT pair -> probe,beta CSV with SeSAMe.
# usage: Rscript dev_epicv2_sesame.R <idat prefix (without _Grn.idat[.gz])> <out.csv> [prep, default QCDPB]
# prep QCDPB = quality mask, infer type I channel, non-linear dye bias, pOOBAH (masked probes -> NA), noob. Needs sesameDataCache() once.
# If the non-linear dye-bias step fails (preprocessCore threads: "pthread_create() is 22" on the box, 2026-10-04), the same steps run with the
# linear dye-bias correction (dyeBiasL) instead; the steps used are written to <out.csv>.prep.
suppressPackageStartupMessages(library(sesame))
a <- commandArgs(trailingOnly = TRUE); prep <- if (length(a) >= 3) a[3] else "QCDPB"
b <- tryCatch({ x <- openSesame(a[1], prep = prep, func = getBetas); writeLines(prep, paste0(a[2], ".prep")); x },
  error = function(e) {
    s <- readIDATpair(a[1]); s <- qualityMask(s); s <- inferInfiniumIChannel(s); s <- dyeBiasL(s); s <- pOOBAH(s); s <- noob(s)
    writeLines(paste0("qualityMask, inferInfiniumIChannel, dyeBiasL, pOOBAH, noob (non-linear dye bias failed: ", conditionMessage(e), ")"), paste0(a[2], ".prep"))
    getBetas(s) })
write.csv(data.frame(probe = names(b), beta = as.numeric(b)), a[2], row.names = FALSE)
