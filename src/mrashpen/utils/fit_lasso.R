#script for calling mr.ash.alpha

library(optparse)

options = list(
  make_option(c("-o", "--outfile"), type="character", default = NULL, 
              help="name of output RDS file", metavar="character"),
  make_option(c("-i", "--infile"), type="character", default = NULL,
              help="name of input RDS file", metavar="character"),
  make_option("--nfolds", type="integer", default = 10,
              help="number of CV folds", metavar="int"),
  make_option("--alpha", type="double", default = 1.0,
              help="elastic net mixing parameter, 0 <= alpha <= 1", metavar="float")
)

opt_parser = OptionParser(option_list=options);
opt = parse_args(opt_parser);
data = readRDS(opt$infile);

fit_lasso <- function (X, y, nfolds = 10, alpha = 1, cvlambda = "min") {
  out.cv <- glmnet::cv.glmnet(X, y, alpha = alpha, nfolds = nfolds)
  fit    <- glmnet::glmnet(X, y, alpha = alpha, standardize = FALSE)
  cvs    <- if (cvlambda == "1se") out.cv$lambda.1se else out.cv$lambda.min
  b      <- as.vector(coef(fit, s = cvs))
  return(list(fit = fit, cv = out.cv, mu = b[1], beta = b[-1]))
}

res = fit_lasso(as.matrix(data$X), as.vector(data$y), nfolds = opt$nfolds, alpha = opt$alpha)
saveRDS(res, file = opt$outfile)
