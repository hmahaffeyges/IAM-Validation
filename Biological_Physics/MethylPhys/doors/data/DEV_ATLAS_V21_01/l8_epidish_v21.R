# Atlas v2.1 L8, D1 (ii) and D10: EpiDISH 2.26.0 RPC fractions on the exported sites. Committed 2026-10-14 BEFORE it runs; rules and cuts in
# l8_conditions_v21.md (D1 eosinophil < 0.10 per array; D10 fibroblast + immune < 0.20 per array, at least 2 arrays per cell remain).
# Usage: Rscript l8_epidish_v21.R l8_epidish_betas.csv
suppressMessages(library(EpiDISH)); data(centBloodSub.m); data(centEpiFibIC.m)
args <- commandArgs(TRUE); B <- as.matrix(read.csv(args[1], row.names = 1, check.names = FALSE))
grp <- sapply(strsplit(colnames(B), "|", fixed = TRUE), `[`, 1); cel <- sapply(strsplit(colnames(B), "|", fixed = TRUE), `[`, 2)
run <- function(cols, ref) { X <- B[intersect(rownames(B), rownames(ref)), cols, drop = FALSE]; X <- X[complete.cases(X), , drop = FALSE]
  list(n = nrow(X), est = epidish(beta.m = X, ref.m = ref[rownames(X), ], method = "RPC")$estF) }
n1 <- which(cel == "neutrophils")
if (length(n1) > 0) { a <- run(n1, centBloodSub.m)
d1 <- data.frame(array = colnames(B)[n1], source = grp[n1], eosino = round(a$est[, "Eosino"], 4), neutro = round(a$est[, "Neutro"], 4), sites = a$n)
d1$passes <- d1$eosino < 0.10
write.csv(d1, "l8_d1_epidish.csv", row.names = FALSE); print(d1[order(d1$source), ]) }
o <- which(grp %in% c("GSE146552", "GSE137716"))   # decision 10 covers both (addendum, 2026-10-14); b <- run(o, centEpiFibIC.m)
d10 <- data.frame(array = colnames(B)[o], cell = cel[o], epi = round(b$est[, "Epi"], 4), fib = round(b$est[, "Fib"], 4), ic = round(b$est[, "IC"], 4), sites = b$n)
d10$non_epithelial <- d10$fib + d10$ic; d10$passes <- d10$non_epithelial < 0.20
write.csv(d10, "l8_d10_epidish.csv", row.names = FALSE); print(d10)
