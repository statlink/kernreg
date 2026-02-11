// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

#ifdef RCPP_PARALLEL_USE_TBB
#include <tbb/global_control.h>  // For controlling the number of threads
#endif

// ---------------- Standardization helper ----------------
inline void standardize_x(NumericMatrix& x,
                          NumericMatrix& xnew,
                          NumericVector& mean_x,
                          NumericVector& sd_x) {
  int n  = x.nrow();
  int p  = x.ncol();
  int nu = xnew.nrow();
  
  for (int j = 0; j < p; j++) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x(i,j);
    double m = sum / n;
    mean_x[j] = m;
    
    double ssq = 0.0;
    for (int i = 0; i < n; i++) {
      double diff = x(i,j) - m;
      ssq += diff * diff;
    }
    double sd = std::sqrt(ssq / (n - 1));
    sd_x[j] = sd;
    
    for (int i = 0; i < n; i++)  x(i,j)    = (x(i,j)    - m) / sd_x[j];
    for (int i = 0; i < nu; i++) xnew(i,j) = (xnew(i,j) - m) / sd_x[j];
  }
}

NumericMatrix to_matrix_copy(SEXP obj) {
  
  // Case 1: already a REAL matrix
  if (Rf_isMatrix(obj) && TYPEOF(obj) == REALSXP) {
    return clone(obj);
  }
  
  // Case 2: numeric vector -> treat as column matrix
  if (TYPEOF(obj) == REALSXP && Rf_isVectorAtomic(obj)) {
    NumericVector v(obj);
    v.attr("dim") = IntegerVector::create(v.size(), 1);
    return NumericMatrix(v);
  }
  
  // Case 3: DataFrame -> convert column-by-column
  if (Rf_inherits(obj, "data.frame")) {
    DataFrame df(obj);
    int n = df.nrows();
    int p = df.size();
    
    NumericMatrix out(n, p);
    
    for (int j = 0; j < p; j++) {
      SEXP col = df[j];
      
      // numeric
      if (TYPEOF(col) == REALSXP) {
        NumericVector v(col);
        for (int i = 0; i < n; i++) out(i,j) = v[i];
      }
      // integer or logical -> promote to double
      else if (TYPEOF(col) == INTSXP || TYPEOF(col) == LGLSXP) {
        IntegerVector v(col);
        for (int i = 0; i < n; i++) out(i,j) = (double)v[i];
      }
      // factor -> convert to underlying integer codes
      else if (Rf_isFactor(col)) {
        IntegerVector v(col);
        for (int i = 0; i < n; i++) out(i,j) = (double)v[i];
      }
      else {
        stop("Non-numeric column in data.frame: column %d", j + 1);
      }
    }
    
    return out;
  }
  
  stop("Input must be a numeric matrix, numeric vector, or data.frame.");
}



// ---------------- Worker: single h, d > 1 ----------------
struct KernRegWorkerMultiH1 : public Worker {
  const RMatrix<double> x;
  const RMatrix<double> xnew;
  const RMatrix<double> y;
  const double h;
  const double h2;
  const std::string type;
  RMatrix<double> out;
  
  KernRegWorkerMultiH1(const NumericMatrix& x_,
                       const NumericMatrix& xnew_,
                       const NumericMatrix& y_,
                       double h_,
                       const std::string& type_,
                       NumericMatrix& out_)
    : x(x_), xnew(xnew_), y(y_), h(h_), h2(h_*h_), type(type_), out(out_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    int n  = x.nrow();
    int p  = x.ncol();
    int d  = y.ncol();
    
    for (std::size_t i = begin; i < end; i++) {
      double rowsum = 0.0;
      std::vector<double> acc(d, 0.0);
      
      for (int j = 0; j < n; j++) {
        double dist = 0.0;
        
        if (type == "gauss") {
          for (int k = 0; k < p; k++) {
            double diff = xnew(i,k) - x(j,k);
            dist += diff * diff;
          }
          dist = std::exp(-0.5 * dist / h2);
        } else {
          for (int k = 0; k < p; k++) {
            dist += std::fabs(xnew(i,k) - x(j,k));
          }
          dist = std::exp(-dist / h);
        }
        
        rowsum += dist;
        for (int k = 0; k < d; k++) acc[k] += dist * y(j,k);
      }
      
      if (rowsum == 0.0) {
        for (int k = 0; k < d; k++) out(i,k) = 0.0;
      } else {
        for (int k = 0; k < d; k++) out(i,k) = acc[k] / rowsum;
      }
    }
  }
};

// ---------------- Worker: single h, d = 1 ----------------
struct KernRegWorkerSingleH1 : public Worker {
  const RMatrix<double> x;
  const RMatrix<double> xnew;
  const RMatrix<double> y;
  const double h;
  const double h2;
  const std::string type;
  RVector<double> out;
  
  KernRegWorkerSingleH1(const NumericMatrix& x_,
                        const NumericMatrix& xnew_,
                        const NumericMatrix& y_,
                        double h_,
                        const std::string& type_,
                        NumericVector& out_)
    : x(x_), xnew(xnew_), y(y_), h(h_), h2(h_*h_), type(type_), out(out_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    int n  = x.nrow();
    int p  = x.ncol();
    
    for (std::size_t i = begin; i < end; i++) {
      double rowsum = 0.0;
      double acc = 0.0;
      
      for (int j = 0; j < n; j++) {
        double dist = 0.0;
        
        if (type == "gauss") {
          for (int k = 0; k < p; k++) {
            double diff = xnew(i,k) - x(j,k);
            dist += diff * diff;
          }
          dist = std::exp(-0.5 * dist / h2);
        } else {
          for (int k = 0; k < p; k++) {
            dist += std::fabs(xnew(i,k) - x(j,k));
          }
          dist = std::exp(-dist / h);
        }
        
        rowsum += dist;
        acc    += dist * y(j,0);
      }
      
      out[i] = (rowsum == 0.0 ? 0.0 : acc / rowsum);
    }
  }
};

// ---------------- Worker: multi h, d = 1 ----------------
struct KernRegWorkerMultiH1Mat : public Worker {
  const RMatrix<double> x;
  const RMatrix<double> xnew;
  const RMatrix<double> y;
  const RVector<double> hvec;
  const std::string type;
  RMatrix<double> out;
  
  KernRegWorkerMultiH1Mat(const NumericMatrix& x_,
                          const NumericMatrix& xnew_,
                          const NumericMatrix& y_,
                          const NumericVector& hvec_,
                          const std::string& type_,
                          NumericMatrix& out_)
    : x(x_), xnew(xnew_), y(y_), hvec(hvec_), type(type_), out(out_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    int n  = x.nrow();
    int p  = x.ncol();
    int H  = hvec.length();
    
    for (std::size_t i = begin; i < end; i++) {
      for (int hidx = 0; hidx < H; hidx++) {
        double h  = hvec[hidx];
        double h2 = h * h;
        double rowsum = 0.0;
        double acc = 0.0;
        
        for (int j = 0; j < n; j++) {
          double dist = 0.0;
          
          if (type == "gauss") {
            for (int k = 0; k < p; k++) {
              double diff = xnew(i,k) - x(j,k);
              dist += diff * diff;
            }
            dist = std::exp(-0.5 * dist / h2);
          } else {
            for (int k = 0; k < p; k++) {
              dist += std::fabs(xnew(i,k) - x(j,k));
            }
            dist = std::exp(-dist / h);
          }
          
          rowsum += dist;
          acc    += dist * y(j,0);
        }
        
        out(i,hidx) = (rowsum == 0.0 ? 0.0 : acc / rowsum);
      }
    }
  }
};

// ---------------- Worker: multi h, d > 1 ----------------
struct KernRegWorkerMultiHList : public Worker {
  const RMatrix<double> x;
  const RMatrix<double> xnew;
  const RMatrix<double> y;
  const RVector<double> hvec;
  const std::string type;
  std::vector< RMatrix<double> > mats; // one matrix per h
  
  KernRegWorkerMultiHList(const NumericMatrix& x_,
                          const NumericMatrix& xnew_,
                          const NumericMatrix& y_,
                          const NumericVector& hvec_,
                          const std::string& type_,
                          List& out_list)
    : x(x_), xnew(xnew_), y(y_), hvec(hvec_), type(type_) {
    int H  = hvec.length();
    int nu = xnew_.nrow();
    int d  = y_.ncol();
    mats.reserve(H);
    for (int k = 0; k < H; k++) {
      NumericMatrix tmp(nu, d);
      out_list[k] = tmp;
      mats.emplace_back(tmp);
    }
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    int n  = x.nrow();
    int p  = x.ncol();
    int d  = y.ncol();
    int H  = hvec.length();
    
    for (std::size_t i = begin; i < end; i++) {
      for (int hidx = 0; hidx < H; hidx++) {
        double h  = hvec[hidx];
        double h2 = h * h;
        double rowsum = 0.0;
        std::vector<double> acc(d, 0.0);
        
        for (int j = 0; j < n; j++) {
          double dist = 0.0;
          
          if (type == "gauss") {
            for (int k = 0; k < p; k++) {
              double diff = xnew(i,k) - x(j,k);
              dist += diff * diff;
            }
            dist = std::exp(-0.5 * dist / h2);
          } else {
            for (int k = 0; k < p; k++) {
              dist += std::fabs(xnew(i,k) - x(j,k));
            }
            dist = std::exp(-dist / h);
          }
          
          rowsum += dist;
          for (int k = 0; k < d; k++) acc[k] += dist * y(j,k);
        }
        
        if (rowsum == 0.0) {
          for (int k = 0; k < d; k++) mats[hidx](i,k) = 0.0;
        } else {
          for (int k = 0; k < d; k++) mats[hidx](i,k) = acc[k] / rowsum;
        }
      }
    }
  }
};


SEXP kern_reg_parallel(const SEXP xnew_,
                       const SEXP y_,
                       const SEXP x_,
                       const NumericVector h_,
                       std::string type = "gauss",
                       int ncores = 1) {
  
  // NumericMatrix x    = clone(x_);
  // NumericMatrix xnew = clone(xnew_);
  // const NumericMatrix& y = y_;
  
  NumericMatrix x    = to_matrix_copy(x_);
  NumericMatrix xnew = to_matrix_copy(xnew_);
  NumericMatrix y    = to_matrix_copy(y_);
  
  // int n  = x_.nrow();
  int p  = x.ncol();
  int d  = y.ncol();
  int nu = xnew.nrow();
  int len_h = h_.size();
  
  NumericVector mean_x(p), sd_x(p);
  standardize_x(x, xnew, mean_x, sd_x);
  
  // column names
  CharacterVector y_colnames;
  bool has_colnames = false;
  if (y.hasAttribute("dimnames")) {
    List dn = y.attr("dimnames");
    if (dn.size() == 2 && !Rf_isNull(dn[1])) {
      y_colnames = dn[1];
      has_colnames = true;
    }
  }
  
  // rownames
  CharacterVector rn(nu);
  for (int i = 0; i < nu; i++) rn[i] = std::to_string(i + 1);
  
  // ---- CASE 1: single bandwidth ----
  if (len_h == 1) {
    double h = h_[0];
    
    if (d == 1) {
      NumericVector out_vec(nu);
      KernRegWorkerSingleH1 worker(x, xnew, y, h, type, out_vec);
      tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
      parallelFor(0, nu, worker);
      return out_vec;
    } else {
      NumericMatrix out(nu, d);
      KernRegWorkerMultiH1 worker(x, xnew, y, h, type, out);
      tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
      parallelFor(0, nu, worker);
      
      if (has_colnames)
        out.attr("dimnames") = List::create(rn, y_colnames);
      else
        out.attr("dimnames") = List::create(rn, R_NilValue);
      
      return out;
    }
  }
  
  // ---- CASE 2: multiple bandwidths, d = 1 ----
  if (d == 1) {
    NumericMatrix out(nu, len_h);
    KernRegWorkerMultiH1Mat worker(x, xnew, y, h_, type, out);
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
    parallelFor(0, nu, worker);
    
    CharacterVector cn(len_h);
    for (int k = 0; k < len_h; k++) {
      std::ostringstream oss;
      oss << "h=" << h_[k];
      cn[k] = oss.str();
    }
    out.attr("dimnames") = List::create(R_NilValue, cn);
    
    return out;
  }
  
  // ---- CASE 3: multiple bandwidths, d > 1 ----
  List out(len_h);
  KernRegWorkerMultiHList worker(x, xnew, y, h_, type, out);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelFor(0, nu, worker);
  
  // add dimnames and list names
  for (int hidx = 0; hidx < len_h; hidx++) {
    NumericMatrix mhx = out[hidx];
    if (has_colnames)
      mhx.attr("dimnames") = List::create(rn, y_colnames);
    else
      mhx.attr("dimnames") = List::create(rn, R_NilValue);
    out[hidx] = mhx;
  }
  
  CharacterVector ln(len_h);
  for (int k = 0; k < len_h; k++) {
    std::ostringstream oss;
    oss << "h=" << h_[k];
    ln[k] = oss.str();
  }
  out.attr("names") = ln;
  
  return out;
}



SEXP kern_reg_cpp(const SEXP xnew_,
                  const SEXP y_,
                  const SEXP x_,
                  const NumericVector h_,
                  std::string type = "gauss") {
  
  // NumericMatrix x    = clone(x_);
  // NumericMatrix xnew = clone(xnew_);
  // const NumericMatrix& y = y_;
  
  NumericMatrix x    = to_matrix_copy(x_);
  NumericMatrix xnew = to_matrix_copy(xnew_);
  NumericMatrix y    = to_matrix_copy(y_);
  
  int n  = x.nrow();
  int p  = x.ncol();
  int d  = y.ncol();
  int nu = xnew.nrow();
  int len_h = h_.size();
  
  // ------------------------------------------------------------
  // Standardize x and xnew
  // ------------------------------------------------------------
  NumericVector mean_x(p), sd_x(p);
  
  for (int j = 0; j < p; j++) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x(i,j);
    double m = sum / n;
    mean_x[j] = m;
    
    double ssq = 0.0;
    for (int i = 0; i < n; i++) {
      double diff = x(i,j) - m;
      ssq += diff * diff;
    }
    sd_x[j] = std::sqrt(ssq / (n - 1));
    
    for (int i = 0; i < n; i++)  x(i,j)    = (x(i,j)    - m) / sd_x[j];
    for (int i = 0; i < nu; i++) xnew(i,j) = (xnew(i,j) - m) / sd_x[j];
  }
  
  // Extract column names
  CharacterVector y_colnames;
  bool has_colnames = false;
  
  if (y.hasAttribute("dimnames")) {
    List dn = y.attr("dimnames");
    if (dn.size() == 2 && !Rf_isNull(dn[1])) {
      y_colnames = dn[1];
      has_colnames = true;
    }
  }
  
  // Precompute rownames once
  CharacterVector rn(nu);
  for (int i = 0; i < nu; i++) rn[i] = std::to_string(i + 1);
  
  // ------------------------------------------------------------
  // CASE 1: single bandwidth
  // ------------------------------------------------------------
  if (len_h == 1) {
    double h  = h_[0];
    double h2 = h * h;
    
    NumericMatrix out(nu, d);
    
    for (int i = 0; i < nu; i++) {
      double rowsum = 0.0;
      NumericVector acc(d);
      for (int k = 0; k < d; k++) acc[k] = 0.0;
      
      for (int j = 0; j < n; j++) {
        double dist = 0.0;
        
        if (type == "gauss") {
          for (int k = 0; k < p; k++) {
            double diff = xnew(i,k) - x(j,k);
            dist += diff * diff;
          }
          dist = std::exp(-0.5 * dist / h2);
        } else {
          for (int k = 0; k < p; k++) {
            dist += std::fabs(xnew(i,k) - x(j,k));
          }
          dist = std::exp(-dist / h);
        }
        
        rowsum += dist;
        for (int k = 0; k < d; k++) acc[k] += dist * y(j,k);
      }
      
      if (rowsum == 0) {
        for (int k = 0; k < d; k++) out(i,k) = 0.0;
      } else {
        for (int k = 0; k < d; k++) out(i,k) = acc[k] / rowsum;
      }
    }
    
    // d = 1 -> return vector
    if (d == 1) {
      NumericVector v(nu);
      for (int i = 0; i < nu; i++) v[i] = out(i,0);
      return v;
    }
    
    if (has_colnames)
      out.attr("dimnames") = List::create(rn, y_colnames);
    else
      out.attr("dimnames") = List::create(rn, R_NilValue);
    
    return out;
  }
  
  // ------------------------------------------------------------
  // CASE 2: multiple bandwidths, d = 1 -> matrix
  // ------------------------------------------------------------
  if (d == 1) {
    NumericMatrix out(nu, len_h);
    
    for (int hidx = 0; hidx < len_h; hidx++) {
      double h  = h_[hidx];
      double h2 = h * h;
      
      for (int i = 0; i < nu; i++) {
        double rowsum = 0.0;
        double acc = 0.0;
        
        for (int j = 0; j < n; j++) {
          double dist = 0.0;
          
          if (type == "gauss") {
            for (int k = 0; k < p; k++) {
              double diff = xnew(i,k) - x(j,k);
              dist += diff * diff;
            }
            dist = std::exp(-0.5 * dist / h2);
          } else {
            for (int k = 0; k < p; k++) {
              dist += std::fabs(xnew(i,k) - x(j,k));
            }
            dist = std::exp(-dist / h);
          }
          
          rowsum += dist;
          acc    += dist * y(j,0);
        }
        
        out(i,hidx) = (rowsum == 0 ? 0.0 : acc / rowsum);
      }
    }
    
    CharacterVector cn(len_h);
    for (int k = 0; k < len_h; k++) {
      std::ostringstream oss;
      oss << "h=" << h_[k];
      cn[k] = oss.str();
    }
    out.attr("dimnames") = List::create(R_NilValue, cn);
    
    return out;
  }
  
  // ------------------------------------------------------------
  // CASE 3: multiple bandwidths, d > 1 -> list of matrices
  // ------------------------------------------------------------
  List out(len_h);
  
  for (int hidx = 0; hidx < len_h; hidx++) {
    double h  = h_[hidx];
    double h2 = h * h;
    
    NumericMatrix mhx(nu, d);
    
    for (int i = 0; i < nu; i++) {
      double rowsum = 0.0;
      NumericVector acc(d);
      for (int k = 0; k < d; k++) acc[k] = 0.0;
      
      for (int j = 0; j < n; j++) {
        double dist = 0.0;
        
        if (type == "gauss") {
          for (int k = 0; k < p; k++) {
            double diff = xnew(i,k) - x(j,k);
            dist += diff * diff;
          }
          dist = std::exp(-0.5 * dist / h2);
        } else {
          for (int k = 0; k < p; k++) {
            dist += std::fabs(xnew(i,k) - x(j,k));
          }
          dist = std::exp(-dist / h);
        }
        
        rowsum += dist;
        for (int k = 0; k < d; k++) acc[k] += dist * y(j,k);
      }
      
      if (rowsum == 0) {
        for (int k = 0; k < d; k++) mhx(i,k) = 0.0;
      } else {
        for (int k = 0; k < d; k++) mhx(i,k) = acc[k] / rowsum;
      }
    }
    
    if (has_colnames)
      mhx.attr("dimnames") = List::create(rn, y_colnames);
    else
      mhx.attr("dimnames") = List::create(rn, R_NilValue);
    
    out[hidx] = mhx;
  }
  
  CharacterVector ln(len_h);
  for (int k = 0; k < len_h; k++) {
    std::ostringstream oss;
    oss << "h=" << h_[k];
    ln[k] = oss.str();
  }
  out.attr("names") = ln;
  
  return out;
}


// [[Rcpp::export]]
SEXP kern_reg(const SEXP xnew,
              const SEXP y,
              const SEXP x,
              const NumericVector h = NumericVector::create(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ),
              std::string type = "gauss",
              int ncores = 1) {
  
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1) {
    return kern_reg_parallel(xnew, y, x, h, type, ncores);
  } else {
    return kern_reg_cpp(xnew, y, x, h, type);
  }
#else
  // No TBB available -> always use serial version
  return kern_reg_cpp(xnew, y, x, h, type);
#endif
}
