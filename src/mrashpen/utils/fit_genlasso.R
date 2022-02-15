#script for calling mr.ash.alpha

library(optparse)

options = list(
  make_option(c("-o", "--outfile"), type="character", default = NULL, 
              help="name of output RDS file", metavar="character"),
  make_option(c("-i", "--infile"), type="character", default = NULL,
              help="name of input RDS file", metavar="character"),
  make_option("--nfolds", type="integer", default = 10,
              help="number of CV folds", metavar="int"),
  make_option("--order", type="integer", default = 1,
              help="trendfiltering order", metavar="int"),
  make_option("--trendfilter", type="logical", action="store_true", default=FALSE,
              help="fit using trendfilter")
)

opt_parser = OptionParser(option_list=options);
opt = parse_args(opt_parser);
data = readRDS(opt$infile);

fit_trendfilter <- function (y, order = 1, nfolds = 5, cvlambda = "1se") {
    #pos   <- 1:length(y)
    #out   <- genlasso::trendfilter(y, pos, X, ord = order)
    out   <- genlasso::trendfilter(y, ord = order)
    cvout <- genlasso::cv.trendfilter(out, k = nfolds)
    cvlam <- if (cvlambda == "1se") cvout$lambda.1se else cvout$lambda.min
    cvidx <- if (cvlambda == "1se") cvout$i.1se else cvout$i.min
    b     <- coef(out, lambda = cvlam)
    #b    <- out$beta[,cvidx]
    ypred <- out$fit[, cvidx]
    return (list(fit = out, cv = cvout, mu = 0, 
                 beta = as.vector(b$beta), ypred = ypred, df = b$df))
}

gcv_genlasso <- function(object) {
    lams <- object$lambda
    df   <- object$df
    n    <- length(object$y)
    ymat <- matrix(object$y, n, length(lams))
    pred <- object$fit
    err  <- colMeans((ymat-pred)^2)/(1-df/n)^2
    names(err) <- round(lams,3)
    lam_min <- lams[which.min(err)]
    out <- list(err = err, lambda = lams, lambda.min = lam_min, i.min = which(lams == lam_min))
    class(out) <- c("gcv.genlasso", "list")
    return (out)
}


fit_genlasso <- function (y, order = 1) {
    n     <- length(y)
    D     <- genlasso::getDtf(n, order)
    out   <- genlasso::genlasso(y = y, X = diag(n), D = D)
    cvout <- gcv_genlasso(out)
    cvidx <- cvout$i.min
    cvlam <- cvout$lambda.min
    b     <- coef(out, lambda = cvlam)
    ypred <- out$fit[, cvidx]
    return (list(fit = out, cv = cvout, mu = 0, 
                 beta = as.vector(b$beta), ypred = ypred, df = b$df))
}

if (opt$trendfilter) {
    res = fit_trendfilter(as.vector(data$y), order = opt$order, nfolds = opt$nfolds)
} else {
    res = fit_genlasso(as.matrix(data$X), as.vector(data$y), order = opt$order)
}
saveRDS(res, file = opt$outfile)
