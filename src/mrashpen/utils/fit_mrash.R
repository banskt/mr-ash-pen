# script for calling mr.ash.alpha

library(optparse)

options = list(
  make_option(c("-o", "--outfile"), type="character", default = NULL, 
              help="name of output RDS file", metavar="character"),
  make_option(c("-i", "--infile"), type="character", default = NULL,
              help="name of input RDS file", metavar="character"),
  make_option("--maxiter", type="integer", default = 2000,
              help="maximum number of outer loop iterations allowed", metavar="int"),
  make_option("--epstol", type="double", default = 1e-12,
              help="epstol", metavar="float"),
  make_option("--convtol", type="double", default = 1e-8,
              help="epstol", metavar="float"),
  make_option("--fix_pi", action = "store_true", default = FALSE,
              help="run without updating Pi", metavar="bool"),
  make_option("--fix_sigma2", action = "store_true", default = FALSE,
              help="run without updating sigma2", metavar="bool")
)

opt_parser = OptionParser(option_list=options);
opt = parse_args(opt_parser);
data = readRDS(opt$infile);
update_pi = TRUE
if (opt$fix_pi) {
    update_pi = FALSE
    }
update_sigma2 = TRUE
if (opt$fix_sigma2) {
    update_sigma2 = FALSE;
    }

start_time <- Sys.time()
res = mr.ash.alpha::mr.ash(as.matrix(data$X), as.vector(data$y),
                           max.iter = opt$maxiter,
                           sa2 = as.vector(data$sk2),
                           beta.init = as.vector(data$binit),
                           pi = as.vector(data$winit),
                           sigma2 = data$s2init,
                           update.pi = update_pi,
                           update.sigma2 = update_sigma2,
                           tol = list(epstol = opt$epstol, convtol = opt$convtol))
end_time <- Sys.time()
run_time <- difftime(end_time, start_time, units = "secs")
res["run_time"] <- run_time

saveRDS(res, file = opt$outfile)
