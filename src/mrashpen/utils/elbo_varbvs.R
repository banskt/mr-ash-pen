# ----------------------------------------------------------------------
# Return the dot product of vectors x and y.
dot <- function (x,y) {
  sum(x*y)
}

# ----------------------------------------------------------------------
# Return the quadratic norm (2-norm) of vector x.
norm2 <- function (x) {
  sqrt(dot(x,x))
}

# ----------------------------------------------------------------------
# betavarmix(p,mu,s) returns variances of variables drawn from mixtures of
# normals. Each of the inputs is a n x k matrix, where n is the number of
# variables and k is the number of mixture components. Specifically,
# variable i is drawn from a mixture in which the jth mixture component is
# the univariate normal with mean mu[i,j] and variance s[i,j].
#
# Note that the following two lines should return the same result when k=2
# and the first component is the "spike" density with zero mean and
# variance.
#
#   y1 <- betavar(p,mu,s)
#   y2 <- betavarmix(c(1-p,p),cbind(0,mu),cbind(0,s))
#
betavarmix <- function (p, mu, s) {
  rowSums(p*(s + mu^2)) - rowSums(p*mu)^2
}

# ----------------------------------------------------------------------
# Compute the lower bound to the marginal log-likelihood.
computevarlbmix <- function (Z, Xr, d, y, sigma, sa, w, alpha, mu, s) {
  
  # Get the number of samples (n), variables (p) and mixture
  # components (K).
  n <- length(y)
  p <- length(d)
  K <- length(w)
  eps <- 1e-8
  
  detZ = determinant(crossprod(Z),logarithm = TRUE)$modulus[1]/2
  
  # Compute the variational lower bound.
  out <- (-n/2*log(2*pi*sigma)
          - detZ
          - (norm2(y - Xr)^2 + dot(d,betavarmix(alpha,mu,s)))/(2*sigma))
  for (i in 1:K)
    out <- (out + sum(alpha[,i]*log(w[i] + eps)) 
            - sum(alpha[,i]*log(alpha[,i] + eps)))
  for (i in 2:K)
    out <- (out + (sum(alpha[,i]) + sum(alpha[,i]*log(s[,i]/(sigma*sa[i]))))/2
            - sum(alpha[,i]*(s[,i] + mu[,i]^2))/(sigma*sa[i])/2)
  return(out)
}

computevarlbmix_debug <- function (Z, Xr, d, y, sigma, sa, w, alpha, mu, s) {
  
  # Get the number of samples (n), variables (p) and mixture
  # components (K).
  n <- length(y)
  p <- length(d)
  K <- length(w)
  eps <- 1e-8
  
  detZ = determinant(crossprod(Z),logarithm = TRUE)$modulus[1]/2
  
  # Compute the variational lower bound.
  out1 <- (-n/2*log(2*pi*sigma)
           - detZ
           - (norm2(y - Xr)^2 + dot(d,betavarmix(alpha,mu,s)))/(2*sigma))
  out2 <- 0
  for (i in 1:K)
    out2 <- (out2 + sum(alpha[,i]*log(w[i] + eps)) 
            - sum(alpha[,i]*log(alpha[,i] + eps)))
  out3 <- 0
  for (i in 2:K)
    out3 <- (out3 + (sum(alpha[,i]) + sum(alpha[,i]*log(s[,i]/(sigma*sa[i]))))/2
            - sum(alpha[,i]*(s[,i] + mu[,i]^2))/(sigma*sa[i])/2)
  return(out1 + out2 + out3)
}

# ----------------------------------------------------------------------
# Get ELBO from unscaled X and Y.
ash_elbo <- function(X, y, sa2, w, sigma2, alpha, mu, s) {
  n <- length(y)
  Xscale  <- scale(X,center = TRUE,scale = FALSE)
  ycenter <- y - mean(y)
  d <- diag(t(Xscale) %*% Xscale)
  Z <- matrix(1,n,1)
  Xr = drop(Xscale %*% rowSums(alpha * mu))
  logZ <- computevarlbmix(Z, Xr, d, ycenter, sigma2, sa2, w, alpha, mu, s)
  return (logZ)
}
