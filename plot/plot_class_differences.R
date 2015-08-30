differences = c()

for (i in seq(1800, 1980, 20)) {
  sf = mean(l$logistic[l$realclass == 1 & l$dateused > i & l$dateused < (i + 20)])
  det = mean(l$logistic[l$realclass == 0 & l$dateused > i & l$dateused < (i + 20)])
  differences = c(differences, (sf - det))
}