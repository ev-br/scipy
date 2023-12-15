// forward declarations for CDFLIB

#pragma once

inline double algdiv(double a, double b);
inline double alngam(double x);
inline double alnrel(double a);
inline double apser(double a, double b, double x, double eps);
inline double basym(double a, double b, double lmbda, double eps);
inline double bcorr(double a0, double b0);
inline double betaln(double a0, double b0);
inline double bfrac(double a, double b, double x, double y, double lmbda, double eps);
inline std::tuple<double, int> bgrat(double a, double b, double x , double y, double w, double eps);
inline std::tuple<double, double, int> bratio(double a, double b, double x, double y);
inline double bpser(double a, double b, double x, double eps);
inline double brcmp1(int mu, double a, double b, double x, double y);
inline double brcomp(double a, double b, double x, double y);
inline double bup(double a, double b, double x, double y, int n, double eps);

inline std::tuple<double, double, int, double> cdfbet_which1(double p, double q, double x, double y, double b);
inline std::tuple<double, double, int, double> cdfbet_which2(double p, double q, double x, double y, double b);
inline std::tuple<double, int, double> cdfbet_which3(double p, double q, double x, double y, double b);
inline std::tuple<double, int, double> cdfbet_which4(double p, double q, double x, double y, double b);

inline std::tuple<double, double, int, double> cdfbin_which1(double s, double xn, double pr, double ompr);
inline std::tuple<double, int, double> cdfbin_which2(double p, double q, double xn, double pr, double ompr);
inline std::tuple<double, int, double> cdfbin_which3(double p, double q, double s, double pr, double ompr);
inline std::tuple<double, double, int, double> cdfbin_which4(double p, double q, double s, double xn);

inline std::tuple<double, double, int, double> cdfchi_which1(double x, double df);
inline std::tuple<double, int, double> cdfchi_which2(double p, double q, double df);
inline std::tuple<double, int, double> cdfchi_which3(double p, double q, double x);

inline std::tuple<double, double, int, double> cdfchn_which1(double x, double df, double pnonc);
inline std::tuple<double, int, double> cdfchn_which2(double p, double df, double pnonc);
inline std::tuple<double, int, double> cdfchn_which3(double p, double x, double pnonc);
inline std::tuple<double, int, double> cdfchn_which4(double p, double x, double df);

inline std::tuple<double, double, int, double> cdff_which1(double f, double dfn, double dfd);
inline std::tuple<double, int, double> cdff_which2(double p, double q, double dfn, double dfd);
inline std::tuple<double, int, double> cdff_which3(double p, double q, double f, double dfd);
inline std::tuple<double, int, double> cdff_which4(double p, double q, double f, double dfn);

inline std::tuple<double, double, int, double> cdffnc_which1(double f, double dfn, double dfd, double phonc);
inline std::tuple<double, int, double> cdffnc_which2(double p, double q, double dfn, double dfd, double phonc);
inline std::tuple<double, int, double> cdffnc_which3(double p, double q, double f, double dfd, double phonc);
inline std::tuple<double, int, double> cdffnc_which4(double p, double q, double f, double dfn, double phonc);
inline std::tuple<double, int, double> cdffnc_which5(double p, double q, double f, double dfn, double dfd);

inline std::tuple<double, double, int, double> cdfgam_which1(double x, double shape, double scale);
inline std::tuple<double, int, double> cdfgam_which2(double p, double q, double shape, double scale);
inline std::tuple<double, int, double> cdfgam_which3(double p, double q, double x, double scale);
inline std::tuple<double, int, double> cdfgam_which4(double p, double q, double x, double shape);

inline std::tuple<double, double, int, double> cdfnbn_which1(double s, double xn, double pr, double ompr);
inline std::tuple<double, int, double> cdfnbn_which2(double p, double q, double xn, double pr, double ompr);
inline std::tuple<double, int, double> cdfnbn_which3(double p, double q, double s, double pr, double ompr);
inline std::tuple<double, double, int, double> cdfnbn_which4(double p, double q, double s, double xn);

inline std::tuple<double, double, int, double> cdfnor_which1(double x, double mean, double sd);
inline std::tuple<double, int, double> cdfnor_which2(double p, double q, double mean, double sd);
inline std::tuple<double, int, double> cdfnor_which3(double p, double q, double x, double sd);
inline std::tuple<double, int, double> cdfnor_which4(double p, double q, double x, double mean);

inline std::tuple<double, double, int, double> cdfpoi_which1(double s, double xlam);
inline std::tuple<double, int, double> cdfpoi_which2(double p, double q, double xlam);
inline std::tuple<double, int, double> cdfpoi_which3(double p, double q, double s);

inline std::tuple<double, double, int, double> cdft_which1(double t, double df);
inline std::tuple<double, int, double> cdft_which2(double p, double q, double df);
inline std::tuple<double, int, double> cdft_which3(double p, double q, double t);

inline std::tuple<double, double, int, double> cdftnc_which1(double t, double df, double pnonc);
inline std::tuple<double, int, double> cdftnc_which2(double p, double q, double df, double pnonc);
inline std::tuple<double, int, double> cdftnc_which3(double p, double q, double t, double pnonc);
inline std::tuple<double, int, double> cdftnc_which4(double p, double q, double t, double df);

inline std::tuple<double, double> cumbet(double x, double y, double a, double b);
inline std::tuple<double, double> cumbin(double s, double xn, double pr, double ompr);
inline std::tuple<double, double> cumchi(double x, double df);
inline std::tuple<double, double> cumchn(double x, double df, double pnonc);
inline std::tuple<double, double> cumf(double f, double dfn, double dfd);
inline std::tuple<double, double, int> cumfnc(double f, double dfn, double dfd, double pnonc);
inline std::tuple<double, double> cumgam(double x, double a);
inline std::tuple<double, double> cumnbn(double s, double xn, double pr, double ompr);
inline std::tuple<double, double> cumnor(double x);
inline std::tuple<double, double> cumpoi(double s, double xlam);
inline std::tuple<double, double> cumt(double t, double df);
inline std::tuple<double, double> cumtnc(double t, double df, double pnonc);

inline double erf(double x);
inline double erfc1(int ind, double x);
inline double esum(int mu, double x);
inline double fpser(double a, double b, double x, double eps);
inline double gam1(double a);
inline std::tuple<double, int> gaminv(double a, double p, double q, double x0);
inline double gaminv_helper_30(double a, double s, double y, double z);
inline double gamln(double a);
inline double gamln1(double a);
inline double gamma(double a);
inline std::tuple<double, double> grat1(double a, double x, double r, double eps);
inline std::tuple<double, double> gratio(double a, double x, int ind);
inline double gsumln(double a, double b);
inline double psi(double xx);
inline double rcomp(double a, double x);
inline double rexp(double x);
inline double rlog(double x);
inline double rlog1(double x);

inline double devlpl(double *a, int n, double x);
inline double dinvnr(double p, double q);
inline double dt1(double p, double q, double df);
inline double stvaln(double p);



double spmpar[3] = { std::numeric_limits<double>::epsilon(), //np.finfo(np.float64).eps,
                     -101.,    // np.finfo(np.float64).tiny,
                     -101.    //np.finfo(np.float64).max     # FIXME
};
