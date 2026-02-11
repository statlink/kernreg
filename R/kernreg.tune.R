kernreg.tune <- function(y, x, h = seq(0.1, 1, length = 10), type = "gauss",
                         nfolds = 10, folds = NULL, seed = NULL, graph = FALSE, ncores = 1) {
  if ( is.matrix(y) )  {
    n <- dim(y)[1]
    D <- dim(y)[2]
  } else {
    n <- length(y)
    D <- 1
  }
  runtime <- proc.time()
  if ( !is.matrix(x) )  x <- as.matrix(x)
  ina <- 1:n
  if ( is.null(folds) )  folds <- .makefolds(ina, nfolds = nfolds, stratified = FALSE, seed = seed)
  nfolds <- length(folds)
  msp <- matrix( nrow = length(folds), ncol = length(h) )
  for ( i in 1:nfolds ) {
    nu <- folds[[ i ]]
    if ( D > 1 )  {
      ytrain <- y[-nu, , drop = FALSE]
      ytest <- y[nu, , drop = FALSE]
    } else {
      ytest <- y[nu]
      ytrain <- y[-nu]
    }
    xtest <- x[nu, , drop = FALSE]
    xtrain <- x[-nu, , drop = FALSE]
    est <- kernreg::kern_reg(xtest, ytrain, xtrain, h, type = type, ncores = ncores)
    if ( D > 1 ) {
      for ( j in 1:length(h) )  msp[i, j] <- mean( est[[ j ]] - ytest)^2
    } else  msp[i, ] <- Rfast::colmeans( (ytest - est)^2 )
  }
 
  runtime <- proc.time() - runtime
  mspe <- Rfast::colmeans(msp)
  names(mspe) <- paste("h=", h, sep = "")
  if (graph) {
    plot(h, mspe, xlab = "Bandwidth parameter (h)", ylab = "MSPE", cex.lab = 1.2, cex.axis = 1.2)
    abline(v = h, col = "lightgrey", lty = 2)
    abline(h = seq(min(mspe), max(mspe), length = 10), col = "lightgrey", lty = 2)
    points(h, mspe, pch = 19)
    lines(h, mspe, lwd = 2)
  }
  list(mspe = mspe, h = h[ which.min(mspe) ], performance = min(mspe), runtime = runtime)
}



.makefolds <- function(ina, nfolds = 10, stratified = TRUE, seed = NULL) {
  names <- paste("Fold", 1:nfolds)
  runs <- sapply(names, function(x) NULL)
  if ( !is.null(seed) )  set.seed(seed)
  
  if ( !stratified ) {
    rat <- length(ina) %% nfolds
    suppressWarnings({
      mat <- matrix( Rfast2::Sample.int( length(ina) ), ncol = nfolds )
    })
    mat[-c( 1:length(ina) )] <- NA
    for ( i in 1:c(nfolds - 1) )  runs[[ i ]] <- mat[, i]
    a <- prod(dim(mat)) - length(ina)
    runs[[ nfolds ]] <- mat[1:c(nrow(mat) - a), nfolds]
  } else {
    labs <- unique(ina)
    run <- list()
    for (i in 1:length(labs)) {
      names <- which( ina == labs[i] )
      run[[ i ]] <- sample(names)
    }
    run <- unlist(run)
    for ( i in 1:length(ina) ) {
      k <- i %% nfolds
      if ( k == 0 )  k <- nfolds
      runs[[ k ]] <- c( runs[[ k ]], run[i] )
    }
  }
  
  for (i in 1:nfolds)  {
    if ( any( is.na(runs[[ i ]]) ) )  runs[[ i ]] <- runs[[ i ]][ !is.na(runs[[ i ]]) ]
  }
  
  if ( length(runs[[ nfolds ]]) == 0 ) runs[[ nfolds ]] <- NULL
  runs
}