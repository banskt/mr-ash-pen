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
              help="trendfiltering order", metavar="int")
)

opt_parser = OptionParser(option_list=options);
opt = parse_args(opt_parser);
data = readRDS(opt$infile);

fit_trendfilter <- function (X, y, order = 1, nfolds = 5, cvlambda = "1se") {
    #pos   <- 1:length(y)
    #out   <- genlasso::trendfilter(y, pos, X, ord = order)
    out   <- genlasso::trendfilter(y, ord = order)
    cvout <- genlasso::cv.trendfilter(out, k = nfolds)
    cvs   <- if (cvlambda == "1se") cvout$lambda.1se else cvout$lambda.min
    b     <- as.vector(coef(out, lambda = cvs))
    idx   <- which(out$lambda == cvs)
    ypred <- out$fit[, idx]
    return (list(fit = out, cv = cvout, mu = 0, beta = b$beta, ypred = ypred))
}

gcv_genlasso <- function(object) {
    lams <- object$lambda
    df <- object$df
    n <- length(object$y)
    ymat <- matrix(object$y,n,length(lams))
    pred <- object$fit
    err <- colMeans((ymat-pred)^2)/(1-df/n)^2
    names(err) <- round(lams,3)
    lam_min <- lams[which.min(err)]
    out <- list(err = err, lambda = lams, lambda.min = lam_min)
    class(out) <- c("gcv.genlasso", "list")
    return (out)
}


fit_genlasso <- function (X, y, order = 1) {
    D     <- genlasso::getDtf(length(y), order)
    out   <- genlasso::genlasso(y = y, X = X, D = D)
    cvout <- gcv_genlasso(out)
    #cvs   <- cvout$lambda.min
    tfout <- fit_trendfilter(X, y, order = order)
    cvs   <- tfout$cv$lambda.1se
    b     <- as.vector(coef(out, lambda = cvs))
    idx   <- which(out$lambda == cvs)
    ypred <- out$fit[, idx]
    return (list(fit = out, cv = cvout, mu = 0, beta = b$beta, ypred = ypred))
}

#res = fit_trendfilter(as.matrix(data$X), as.vector(data$y), order = opt$order, nfolds = opt$nfolds)
res = fit_genlasso(as.matrix(data$X), as.vector(data$y), order = opt$order)
saveRDS(res, file = opt$outfile)
