\name{Kernel regression with a numerical response vector or matrix}
\alias{kern_reg}

\title{
Kernel regression with a numerical response vector or matrix
}

\description{
Kernel regression (Nadaraya-Watson estimator) with a numerical response vector or matrix.
}

\usage{
kern_reg(xnew, y, x, h = as.numeric( c(0.1, 0.2, 0.3, 0.4, 0.5, 
0.6, 0.7, 0.8, 0.9, 1.0 )), type = "gauss", ncores = 1L)
}

\arguments{
\item{xnew}{
A matrix with the new predictor variables whose compositions are to be predicted.
}
\item{y}{
A numerical vector or a matrix with the response value.
}
\item{x}{
A matrix with the available predictor variables.
}
\item{h}{
The bandwidth value(s) to consider.
}
\item{type}{
The type of kernel to use, "gauss" or "laplace".
}
\item{ncores}{
The number of cores to use. If greater than 1, parallel computing will take
place. It is advisable to use it if you have many observations and or many
variables, otherwise it will slow down the process. The default is 1, meaning
that code is executed serially.
}
}

\details{
The Nadaraya-Watson estimator regression is applied.
}

\value{
The fitted values. If a single bandwidth is considered then this is a vector or a matrix, depeding on the nature of the response. If multiple bandwidth values are considered then this is a matrix, if the response is a vector, or a list, if the response is a matrix.
}

\references{
Wand M. P. and Jones M. C. (1994). Kernel smoothing. CRC press.
}

\author{
Michail Tsagris and Christos Adam.
}

%\note{
  %%  ~~further notes~~
    %}

\seealso{
\code{\link{kernreg.tune}}
}

\examples{
y <- iris[, 1]
x <- iris[, 2:4]
est <- kern_reg(x, y, x, h = c(0.1, 0.2) )
}

