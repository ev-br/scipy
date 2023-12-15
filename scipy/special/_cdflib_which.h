#include "_cdflib_forward.h"

/*
# %% ---------------------------------------- cdfbet_whichX
#
#                Cumulative Distribution Function
#                          BETa Distribution
#
#
#                               Function
#
#
#      Calculates any one parameter of the beta distribution given
#      values for the others.
#
#
#                               Arguments
#
#
#      WHICH --> Integer indicating which of the next four argument
#                values is to be calculated from the others.
#                Legal range: 1..4
#                iwhich = 1 : Calculate P and Q from X,Y,A and B
#                iwhich = 2 : Calculate X and Y from P,Q,A and B
#                iwhich = 3 : Calculate A from P,Q,X,Y and B
#                iwhich = 4 : Calculate B from P,Q,X,Y and A
#
#                     INTEGER WHICH
#
#      P <--> The integral from 0 to X of the chi-square
#             distribution.
#             Input range: [0, 1].
#                     DOUBLE PRECISION P
#
#      Q <--> 1-P.
#             Input range: [0, 1].
#             P + Q = 1.0.
#                     DOUBLE PRECISION Q
#
#      X <--> Upper limit of integration of beta density.
#             Input range: [0,1].
#             Search range: [0,1]
#                     DOUBLE PRECISION X
#
#      Y <--> 1-X.
#             Input range: [0,1].
#             Search range: [0,1]
#             X + Y = 1.0.
#                     DOUBLE PRECISION Y
#
#      A <--> The first parameter of the beta density.
#             Input range: (0, +infinity).
#             Search range: [1D-100,1D100]
#                     DOUBLE PRECISION A
#
#      B <--> The second parameter of the beta density.
#             Input range: (0, +infinity).
#             Search range: [1D-100,1D100]
#                     DOUBLE PRECISION B
#
#      STATUS <-- 0 if calculation completed correctly
#                -I if input parameter number I is out of range
#                 1 if answer appears to be lower than lowest
#                   search bound
#                 2 if answer appears to be higher than greatest
#                   search bound
#                 3 if P + Q .ne. 1
#                 4 if X + Y .ne. 1
#                     INTEGER STATUS
#
#      BOUND <-- Undefined if STATUS is 0
#
#                Bound exceeded by parameter number I if STATUS
#                is negative.
#
#                Lower search bound if STATUS is 1.
#
#                Upper search bound if STATUS is 2.
#
#
#                               Method
#
#
#      Cumulative distribution function  (P)  is calculated directly by
#      code associated with the following reference.
#
#      DiDinato, A. R. and Morris,  A.   H.  Algorithm 708: Significant
#      Digit Computation of the Incomplete  Beta  Function Ratios.  ACM
#      Trans. Math.  Softw. 18 (1993), 360-373.
#
#      Computation of other parameters involve a search for a value that
#      produces  the desired  value  of P.   The search relies  on  the
#      monotinicity of P with the other parameter.
#
#
#                               Note
#
#      The beta density is proportional to
#                t^(A-1) * (1-t)^(B-1)
*/

inline std::tuple<double, double, int, double> cdfbet_which1(
    double x, double y,double a, double b
    ){
    double p, q;

    if (!((0 <= x) &&(x <= 1.))){
        return std::make_tuple(0., 0., -1, !(x > 0) ? 0. : 1.);
    }
    if (! ((0 <= y) && (y <= 1.))){
        return std::make_tuple(0., 0., -2, !(y > 0) ? 0. : 1.);
    }
    if (! (a > 0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }
    if (! (b > 0.)){
        return std::make_tuple(0., 0., -4, 0.);
    }
    if (((abs(x+y)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 0., 4, x + y < 0 ? 0. : 1.);
    }

    std::tie(p, q) = cumbet(x, y, a, b);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, double, int, double> cdfbet_which2(
    double p, double q, double a, double b){

    double ccum, cum, xx, yy;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq;
    // Cython doesn't allow for default values in structs
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0};
    DZ.xlo = 0.;
    DZ.xhi = 1.;
    DZ.atol = atol;
    DZ.rtol = tol;
    DZ.x = 0.;
    DZ.b = 0.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., 0., -1, !(p > 0) ? 0. : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., 0., -2, !(q > 0) ? 0. : 1);
    }
    if (! (a > 0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }
    if (! (b > 0.)){
        return std::make_tuple(0., 0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 0., 3, (p+q) < 0 ? 0. : 1.);
    }

    qporq = p <= q;
    if (qporq){
        dzror(&DZ);
        yy = 1. - DZ.x;
        while (DZ.status == 1){
            std::tie(cum, ccum) = cumbet(DZ.x, yy, a, b);
            DZ.fx = cum - p;
            dzror(&DZ);
            yy = 1. - DZ.x;
        }
        xx = DZ.x;
    }
    else{
        dzror(&DZ);
        xx = 1. - DZ.x;
        while (DZ.status == 1){
            std::tie(cum, ccum) = cumbet(xx, DZ.x, a, b);
            DZ.fx = ccum - q;
            dzror(&DZ);
            xx = 1. - DZ.x;
        }
        yy = DZ.x;
    }
    if (DZ.status == -1){
        return std::make_tuple(xx, yy, DZ.qleft ? 1 : 2, DZ.qleft ? 0. : 1.);
    }
    else{
        return std::make_tuple(xx, yy, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfbet_which3(
    double p, double q, double x, double y, double b){
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    double cum, ccum;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0) ? 0. : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0) ? 0. : 1); 
    }
    if (! ((0 <= x) &&(x <= 1.))){
        return std::make_tuple(0., -3, !(x > 0) ? 0. : 1.);
    }
    if (! ((0 <= y) && (y <= 1.))){
        return std::make_tuple(0., -4, !(y > 0) ? 0. : 1.);
    }
    if (! (b > 0.)){
        return std::make_tuple(0., -5, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, (p+q) < 0 ? 0. : 1.);
    }
    if (((abs(x+y)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 4, x + y < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumbet(x, y, DS.x, b);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft  ? 0. : INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfbet_which4(
    double p, double q, double x, double y, double a){
    double tol = 1e-8;
    double atol = 1e-50;
    double cum, ccum;
    bool qporq = p <= q;

    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0) ? 0. : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0) ? 0. : 1);
    }
    if (! ((0 <= x) &&(x <= 1.))){
        return std::make_tuple(0., -3, !(x > 0) ? 0. : 1.);
    }
    if (! ((0 <= y) && (y <= 1.))){
        return std::make_tuple(0., -4, !(y > 0) ? 0. : 1.);
    }
    if (! (a > 0.)){
        return std::make_tuple(0., -5, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, (p+q) < 0 ? 0. : 1.);
    }
    if (((abs(x+y)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 4, x + y < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumbet(x, y, a, DS.x);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0. : INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdfbin_whichX
#      SUBROUTINE CDFBIN ( WHICH, P, Q, S, XN, PR, OMPR, STATUS, BOUND )
#               Cumulative Distribution Function
#                         BINomial distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the binomial
#     distribution given values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which of the next four argument
#               values is to be calculated from the others.
#               Legal range: 1..4
#               iwhich = 1 : Calculate P and Q from S,XN,PR and OMPR
#               iwhich = 2 : Calculate S from P,Q,XN,PR and OMPR
#               iwhich = 3 : Calculate XN from P,Q,S,PR and OMPR
#               iwhich = 4 : Calculate PR and OMPR from P,Q,S and XN
#                    INTEGER WHICH
#
#     P <--> The cumulation from 0 to S of the binomial distribution.
#            (Probablility of S or fewer successes in XN trials each
#            with probability of success PR.)
#            Input range: [0,1].
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Input range: [0, 1].
#            P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#     S <--> The number of successes observed.
#            Input range: [0, XN]
#            Search range: [0, XN]
#                    DOUBLE PRECISION S
#
#     XN  <--> The number of binomial trials.
#              Input range: (0, +infinity).
#              Search range: [1E-100, 1E100]
#                    DOUBLE PRECISION XN
#
#     PR  <--> The probability of success in each binomial trial.
#              Input range: [0,1].
#              Search range: [0,1]
#                    DOUBLE PRECISION PR
#
#     OMPR  <--> 1-PR
#              Input range: [0,1].
#              Search range: [0,1]
#              PR + OMPR = 1.0
#                    DOUBLE PRECISION OMPR
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                4 if PR + OMPR .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula  26.5.24    of   Abramowitz  and    Stegun,  Handbook   of
#     Mathematical   Functions (1966) is   used  to reduce the  binomial
#     distribution  to  the  cumulative incomplete    beta distribution.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
*/

inline std::tuple<double, double, int, double> cdfbin_which1(
    double s, double xn, double pr, double ompr){
    double p, q;

    if (! ((0 <= s) && (s <= xn))){
        return std::make_tuple(0., 0., -1, !(s > 0.) ? 0. : xn);
    }
    if (! (xn > 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    if (! ((0 <= pr) && (pr <= 1.))){
        return std::make_tuple(0., 0., -3, !(pr > 0.) ? 0. : 1.);
    }
    if (! ((0 <= ompr) && (ompr <= 1.))){
        return std::make_tuple(0., 0., -4, !(ompr > 0.) ? 0. : 1.);
    }
    if (((abs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 0., 4, (pr+ompr) < 0 ? 0. : 1.);
    }

    std::tie(p, q) = cumbin(s, xn, pr, ompr);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, int, double> cdfbin_which2(
    double p, double q, double xn, double pr, double ompr){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 0.;
    DS.big = xn;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = xn/2.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0) ? 0. : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0) ? 0. : 1);
    }
    if (! (xn > 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! ((0 <= pr) && (pr <= 1.))){
        return std::make_tuple(0., -4, !(pr > 0.) ? 0. : 1.);
    }
    if (! ((0 <= ompr) && (ompr <= 1.))){
        return std::make_tuple(0., -5, !(ompr > 0.) ? 0. : 1.);
    }
    if (((abs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 4, (pr+ompr) < 0 ? 0. : 1.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, (p+q) < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumbin(DS.x, xn, pr, ompr);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0. : xn);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfbin_which3(
    double p, double q, double s, double pr, double ompr){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 1e-100;
    DS.big = 1.e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0) ? 0. : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0) ? 0. : 1);
    }
    if (! (s >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! ((0 <= pr) && (pr <= 1.))){
        return std::make_tuple(0., -4, !(pr > 0.) ? 0. : 1.);
    }
    if (! ((0 <= ompr) && (ompr <= 1.))){
        return std::make_tuple(0., -5, !(ompr > 0.) ? 0. : 1.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, (p+q) < 0 ? 0. : 1.);
    }
    if (((abs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 4, (pr+ompr) < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumbin(s, DS.x, pr, ompr);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft  ? 0. : INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, double, int, double> cdfbin_which4(
    double p, double q, double s, double xn){
    double ccum, cum, pr, ompr;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    // Cython doesn't allow for default values in structs
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0};
    DZ.xlo = 0.;
    DZ.xhi = 1.;
    DZ.atol = atol;
    DZ.rtol = tol;
    DZ.x = 0.;
    DZ.b = 0.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., 0., -1, !(p > 0) ? 0. : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., 0., -2, !(q > 0) ? 0. : 1);
    }
    if (! ((0 <= s) && (s <= xn))){
        return std::make_tuple(0., 0., -3, !(s > 0.) ? 0. : xn);
    }
    if (! (xn > 0.)){
        return std::make_tuple(0., 0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 0., 3, (p+q) < 0 ? 0. : 1.);
    }

    if (qporq){
        dzror(&DZ);
        ompr = 1. - DZ.x;
        while (DZ.status == 1){
            std::tie(cum, ccum) = cumbin(s, xn, DZ.x, ompr);
            DZ.fx = cum - p;
            dzror(&DZ);
            ompr = 1. - DZ.x;
        }
        pr = DZ.x;
    }
    else{
        dzror(&DZ);
        pr = 1. - DZ.x;
        while (DZ.status == 1){
            std::tie(cum, ccum) = cumbin(s, xn, pr, DZ.x);
            DZ.fx = ccum - q;
            dzror(&DZ);
            pr = 1. - DZ.x;
        }
        ompr = DZ.x;
    }

    if (DZ.status == -1){
        return std::make_tuple(pr, ompr, DZ.qleft ? 1 : 2, DZ.qleft ? 0. : 1.);
    }
    else{
        return std::make_tuple(pr, ompr, 0, 0.);
    }
}



/*
# %% ---------------------------------------- cdfchi_whichX
#
#               Cumulative Distribution Function
#               CHI-Square distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the chi-square
#     distribution given values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which of the next three argument
#               values is to be calculated from the others.
#               Legal range: 1..3
#               iwhich = 1 : Calculate P and Q from X and DF
#               iwhich = 2 : Calculate X from P,Q and DF
#               iwhich = 3 : Calculate DF from P,Q and X
#                    INTEGER WHICH
#
#     P <--> The integral from 0 to X of the chi-square
#            distribution.
#            Input range: [0, 1].
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Input range: (0, 1].
#            P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#     X <--> Upper limit of integration of the non-central
#            chi-square distribution.
#            Input range: [0, +infinity).
#            Search range: [0,1E100]
#                    DOUBLE PRECISION X
#
#     DF <--> Degrees of freedom of the
#             chi-square distribution.
#             Input range: (0, +infinity).
#             Search range: [ 1E-100, 1E100]
#                    DOUBLE PRECISION DF
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#               10 indicates error returned from cumgam.  See
#                  references in cdfgam
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula    26.4.19   of Abramowitz  and     Stegun, Handbook  of
#     Mathematical Functions   (1966) is used   to reduce the chisqure
#     distribution to the incomplete distribution.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
*/

inline std::tuple<double, double, int, double> cdfchi_which1(
    double x, double df){
    double p, q;

    if (! (x >= 0.)){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (df >= 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    std::tie(p, q) = cumchi(x, df);

    return std::make_tuple(p, q, 0, 0);
}

inline std::tuple<double, int, double> cdfchi_which2(
    double p, double q, double df){
    bool qporq = p <= q;
    double porq = qporq ? p : q;
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (df >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    DS.small = 0.;
    DS.big = 1.e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumchi(DS.x, df);
        DS.fx = qporq ? cum - p : ccum - q;
        if (DS.fx + porq <= 1.5){
            return std::make_tuple(DS.x, 10, 0.);
        }
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfchi_which3(
    double p, double q, double x){
    bool qporq = p <= q;
    double porq = qporq ? p : q;
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (x >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    DS.small = 0.;
    DS.big = 1.e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumchi(x, DS.x);
        DS.fx = qporq ? cum - p : ccum - q;
        if (DS.fx + porq > 1.5){
            return std::make_tuple(DS.x, 10, 0.);
        }
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdfchn_whichX
#
#      SUBROUTINE CDFCHN( WHICH, P, Q, X, DF, PNONC, STATUS, BOUND )
#               Cumulative Distribution Function
#               Non-central Chi-Square
#
#
#                              Function
#
#
#     Calculates any one parameter of the non-central chi-square
#     distribution given values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which of the next three argument
#               values is to be calculated from the others.
#               Input range: 1..4
#               iwhich = 1 : Calculate P and Q from X and DF
#               iwhich = 2 : Calculate X from P,DF and PNONC
#               iwhich = 3 : Calculate DF from P,X and PNONC
#               iwhich = 3 : Calculate PNONC from P,X and DF
#                    INTEGER WHICH
#
#     P <--> The integral from 0 to X of the non-central chi-square
#            distribution.
#            Input range: [0, 1-1E-16).
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Q is not used by this subroutine and is only included
#            for similarity with other cdf* routines.
#                    DOUBLE PRECISION Q
#
#     X <--> Upper limit of integration of the non-central
#            chi-square distribution.
#            Input range: [0, +infinity).
#            Search range: [0,1E300]
#                    DOUBLE PRECISION X
#
#     DF <--> Degrees of freedom of the non-central
#             chi-square distribution.
#             Input range: (0, +infinity).
#             Search range: [ 1E-300, 1E300]
#                    DOUBLE PRECISION DF
#
#     PNONC <--> Non-centrality parameter of the non-central
#                chi-square distribution.
#                Input range: [0, +infinity).
#                Search range: [0,1E4]
#                    DOUBLE PRECISION PNONC
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula  26.4.25   of   Abramowitz   and   Stegun,  Handbook  of
#     Mathematical  Functions (1966) is used to compute the cumulative
#     distribution function.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
#
#
#                            WARNING
#
#     The computation time  required for this  routine is proportional
#     to the noncentrality  parameter  (PNONC).  Very large  values of
#     this parameter can consume immense  computer resources.  This is
#     why the search range is bounded by 1e9.
*/

inline std::tuple<double, double, int, double> cdfchn_which1(
    double x, double df, double pnonc){
    double p, q;

    x = std::min(x, spmpar[2]);
    df = std::min(df, spmpar[2]);
    pnonc = std::min(pnonc, 1.e9);
    if (! (x >= 0.)){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (df >= 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    if (! (pnonc >= 0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }
    std::tie(p, q) = cumchn(x, df, pnonc);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, int, double> cdfchn_which2(
    double p, double df, double pnonc){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0.;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    df = std::min(df, spmpar[2]);
    pnonc = std::min(pnonc, 1.e9);

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (df >= 0.)){
        return std::make_tuple(0., -2, 0.);
    }
    if (! (pnonc >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused) = cumchn(DS.x, df, pnonc);
        DS.fx = cum - p;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfchn_which3(
    double p, double x, double pnonc){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0.;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    x = std::min(x, spmpar[2]);
    pnonc = std::min(pnonc, 1.e9);

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (x >= 0.)){
        return std::make_tuple(0., -2, 0.);
    }
    if (! (pnonc >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused) = cumchn(x, DS.x, pnonc);
        DS.fx = cum - p;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfchn_which4(
    double p, double x, double df){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0.;
    DS.big = 1.e9;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    x = std::min(x, spmpar[2]);
    df = std::min(df, spmpar[2]);

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (x >= 0.)){
        return std::make_tuple(0., -2, 0.);
    }
    if (! (df >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused) = cumchn(x, df, DS.x);
        DS.fx = cum - p;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdff_whichX
#
#               Cumulative Distribution Function
#               F distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the F distribution
#     given values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which of the next four argument
#               values is to be calculated from the others.
#               Legal range: 1..4
#               iwhich = 1 : Calculate P and Q from F,DFN and DFD
#               iwhich = 2 : Calculate F from P,Q,DFN and DFD
#               iwhich = 3 : Calculate DFN from P,Q,F and DFD
#               iwhich = 4 : Calculate DFD from P,Q,F and DFN
#                    INTEGER WHICH
#
#       P <--> The integral from 0 to F of the f-density.
#              Input range: [0,1].
#                    DOUBLE PRECISION P
#
#       Q <--> 1-P.
#              Input range: (0, 1].
#              P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#       F <--> Upper limit of integration of the f-density.
#              Input range: [0, +infinity).
#              Search range: [0,1E100]
#                    DOUBLE PRECISION F
#
#     DFN < --> Degrees of freedom of the numerator sum of squares.
#               Input range: (0, +infinity).
#               Search range: [ 1E-100, 1E100]
#                    DOUBLE PRECISION DFN
#
#     DFD < --> Degrees of freedom of the denominator sum of squares.
#               Input range: (0, +infinity).
#               Search range: [ 1E-100, 1E100]
#                    DOUBLE PRECISION DFD
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula   26.6.2   of   Abramowitz   and   Stegun,  Handbook  of
#     Mathematical  Functions (1966) is used to reduce the computation
#     of the  cumulative  distribution function for the  F  variate to
#     that of an incomplete beta.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
#
#                              WARNING
#
#     The value of the  cumulative  F distribution is  not necessarily
#     monotone in  either degrees of freedom.  There  thus may  be two
#     values  that  provide a given CDF  value.   This routine assumes
#     monotonicity and will find an arbitrary one of the two values.
*/

inline std::tuple<double, double, int, double> cdff_which1(
    double f, double dfn, double dfd){
    double p, q;

    if (! (f >= 0.)){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (dfn >  0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    if (! (dfd >  0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }

    std::tie(p, q) = cumf(f, dfn, dfd);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, int, double> cdff_which2(
    double p, double q, double dfn, double dfd){
    double cum, ccum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0.;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (dfn >  0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (dfd >  0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumf(DS.x, dfd, dfd);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdff_which3(
    double p, double q, double f, double dfd){
    double cum, ccum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (f >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (dfd >  0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumf(f, DS.x, dfd);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdff_which4(
    double p, double q, double f, double dfn){
    double cum, ccum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (f >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (dfn >  0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumf(f, dfn, DS.x);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdffnc_whichX
#
#               Cumulative Distribution Function
#               Non-central F distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the Non-central F
#     distribution given values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which of the next five argument
#               values is to be calculated from the others.
#               Legal range: 1..5
#               iwhich = 1 : Calculate P and Q from F,DFN,DFD and PNONC
#               iwhich = 2 : Calculate F from P,Q,DFN,DFD and PNONC
#               iwhich = 3 : Calculate DFN from P,Q,F,DFD and PNONC
#               iwhich = 4 : Calculate DFD from P,Q,F,DFN and PNONC
#               iwhich = 5 : Calculate PNONC from P,Q,F,DFN and DFD
#                    INTEGER WHICH
#
#       P <--> The integral from 0 to F of the non-central f-density.
#              Input range: [0,1-1E-16).
#                    DOUBLE PRECISION P
#
#       Q <--> 1-P.
#            Q is not used by this subroutine and is only included
#            for similarity with other cdf* routines.
#                    DOUBLE PRECISION Q
#
#       F <--> Upper limit of integration of the non-central f-density.
#              Input range: [0, +infinity).
#              Search range: [0,1E100]
#                    DOUBLE PRECISION F
#
#     DFN < --> Degrees of freedom of the numerator sum of squares.
#               Input range: (0, +infinity).
#               Search range: [ 1E-100, 1E100]
#                    DOUBLE PRECISION DFN
#
#     DFD < --> Degrees of freedom of the denominator sum of squares.
#               Must be in range: (0, +infinity).
#               Input range: (0, +infinity).
#               Search range: [ 1E-100, 1E100]
#                    DOUBLE PRECISION DFD
#
#     PNONC <-> The non-centrality parameter
#               Input range: [0,infinity)
#               Search range: [0,1E4]
#                    DOUBLE PRECISION PHONC
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula  26.6.20   of   Abramowitz   and   Stegun,  Handbook  of
#     Mathematical  Functions (1966) is used to compute the cumulative
#     distribution function.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
#
#                            WARNING
#
#     The computation time  required for this  routine is proportional
#     to the noncentrality  parameter  (PNONC).  Very large  values of
#     this parameter can consume immense  computer resources.  This is
#     why the search range is bounded by 10,000.
#
#                              WARNING
#
#     The  value  of the  cumulative  noncentral F distribution is not
#     necessarily monotone in either degrees  of freedom.  There  thus
#     may be two values that provide a given  CDF value.  This routine
#     assumes monotonicity  and will find  an arbitrary one of the two
#     values.
#
*/

std::tuple<double, double, int, double> cdffnc_which1(
    double f, double dfn, double dfd, double phonc){
    double p, q;
    int ierr;

    if (! (f >= 0.)){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (dfn >  0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    if (! (dfd >  0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }
    if (! (phonc >= 0.)){
        return std::make_tuple(0., 0., -4, 0.);
    }

    std::tie(p, q, ierr) = cumfnc(f, dfn, dfd, phonc);
    if (ierr != 0){
        return std::make_tuple(p, q, 10, 0.);
    }
    else{
        return std::make_tuple(p, q, 0, 0.);
    }
}

std::tuple<double, int, double> cdffnc_which2(
    double p, double q, double dfn, double dfd, double phonc){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    int ierr;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0.;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if  (! ((0 <= p) && (p <= 1. - 1e-16))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (dfn >  0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (dfd >  0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (! (phonc >= 0.)){
        return std::make_tuple(0., -5, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused, ierr) = cumfnc(DS.x, dfn, dfd, phonc);
        DS.fx = cum - p;
        if (ierr != 0){
            return std::make_tuple(DS.x, 10, 0.);
        }
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

std::tuple<double, int, double> cdffnc_which3(
    double p, double q, double f, double dfd, double phonc){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    int ierr;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if  (! ((0 <= p) && (p <= 1. - 1e-16))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (f >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (dfd >  0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (! (phonc >= 0.)){
        return std::make_tuple(0., -5, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused, ierr) = cumfnc(f, DS.x, dfd, phonc);
        DS.fx = cum - p;
        if (ierr != 0){
            return std::make_tuple(DS.x, 10, 0.);
        }
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

std::tuple<double, int, double> cdffnc_which4(
    double p, double q, double f, double dfn, double phonc){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    int ierr;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 1e-100;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if  (! ((0 <= p) && (p <= 1. - 1e-16))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (f >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (dfn >  0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (! (phonc >= 0.)){
        return std::make_tuple(0., -5, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused, ierr) = cumfnc(f, dfn, DS.x, phonc);
        DS.fx = cum - p;
        if (ierr != 0){
            return std::make_tuple(DS.x, 10, 0.);
        }
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

std::tuple<double, int, double> cdffnc_which5(
    double p, double q, double f, double dfn, double dfd){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    int ierr;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0.;
    DS.big = 1e4;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if  (! ((0 <= p) && (p <= 1. - 1e-16))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (f >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (dfd >  0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (! (dfn >  0.)){
        return std::make_tuple(0., -5, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused, ierr) = cumfnc(f, dfn, dfd, DS.x);
        DS.fx = cum - p;
        if (ierr != 0){
            return std::make_tuple(DS.x, 10, 0.);
        }
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  1e4);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdfgam_whichX
#
#               Cumulative Distribution Function
#                         GAMma Distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the gamma
#     distribution given values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which of the next four argument
#               values is to be calculated from the others.
#               Legal range: 1..4
#               iwhich = 1 : Calculate P and Q from X,SHAPE and SCALE
#               iwhich = 2 : Calculate X from P,Q,SHAPE and SCALE
#               iwhich = 3 : Calculate SHAPE from P,Q,X and SCALE
#               iwhich = 4 : Calculate SCALE from P,Q,X and SHAPE
#                    INTEGER WHICH
#
#     P <--> The integral from 0 to X of the gamma density.
#            Input range: [0,1].
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Input range: (0, 1].
#            P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#
#     X <--> The upper limit of integration of the gamma density.
#            Input range: [0, +infinity).
#            Search range: [0,1E100]
#                    DOUBLE PRECISION X
#
#     SHAPE <--> The shape parameter of the gamma density.
#                Input range: (0, +infinity).
#                Search range: [1E-100,1E100]
#                  DOUBLE PRECISION SHAPE
#
#
#     SCALE <--> The scale parameter of the gamma density.
#                Input range: (0, +infinity).
#                Search range: (1E-100,1E100]
#                   DOUBLE PRECISION SCALE
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                10 if the gamma or inverse gamma routine cannot
#                   compute the answer.  Usually happens only for
#                   X and SHAPE very large (gt 1E10 or more)
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Cumulative distribution function (P) is calculated directly by
#     the code associated with:
#
#     DiDinato, A. R. and Morris, A. H. Computation of the  incomplete
#     gamma function  ratios  and their  inverse.   ACM  Trans.  Math.
#     Softw. 12 (1986), 377-393.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
#
#
#                              Note
#
#
#
#     The gamma density is proportional to
#       T**(SHAPE - 1) * EXP(- SCALE * T)
*/

std::tuple<double, double, int, double> cdfgam_which1(
    double x, double shape, double scale){
    double p, q;

    if (! (x >= 0.)){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (shape > 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    if (! (scale > 0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }

    std::tie(p, q) = cumgam(x*scale, shape);
    if (p >= 1.5){
        return std::make_tuple(p, q, 10, 0.);
    }
    else{
        return std::make_tuple(p, q, 0, 0.);
    }
}

std::tuple<double, int, double> cdfgam_which2(
    double p, double q, double shape, double scale){
    double xx;
    int ierr;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (shape > 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (scale > 0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    std::tie(xx, ierr) = gaminv(shape, p, q, -1);
    if (ierr < 0){
        return std::make_tuple(0., 10, 0.);
    }
    else{
        return std::make_tuple(xx/scale, 0, 0.);
    }
}

std::tuple<double, int, double> cdfgam_which3(
    double p, double q, double x, double scale){
    double cum, ccum;
    double xscale = x*scale;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;

    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0.;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (x >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (scale > 0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumgam(xscale, DS.x);
        DS.fx = qporq ? cum - p : ccum - q;
        if(qporq && (cum > 1.5) || ((!qporq) && (ccum > 1.5))){
            return std::make_tuple(DS.x, 10, 0.);
        }
        dinvr(&DS, &DZ); 
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

std::tuple<double, int, double> cdfgam_which4(
    double p, double q, double x, double shape){
    double xx;
    int ierr;

    if (! ((0 <= p) && (p <= 1.))){;
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (x >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (shape > 0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    std::tie(xx, ierr) = gaminv(shape, p, q, -1);
    if (ierr < 0){
        return std::make_tuple(0., 10, 0.);
    }
    else{
        return std::make_tuple(xx/x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdfnbn_whichX
#
#               Cumulative Distribution Function
#               Negative BiNomial distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the negative binomial
#     distribution given values for the others.
#
#     The  cumulative  negative   binomial  distribution  returns  the
#     probability that there  will be  F or fewer failures before  the
#     XNth success in binomial trials each of which has probability of
#     success PR.
#
#     The individual term of the negative binomial is the probability of
#     S failures before XN successes and is
#          Choose( S, XN+S-1 ) * PR^(XN) * (1-PR)^S
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which of the next four argument
#               values is to be calculated from the others.
#               Legal range: 1..4
#               iwhich = 1 : Calculate P and Q from S,XN,PR and OMPR
#               iwhich = 2 : Calculate S from P,Q,XN,PR and OMPR
#               iwhich = 3 : Calculate XN from P,Q,S,PR and OMPR
#               iwhich = 4 : Calculate PR and OMPR from P,Q,S and XN
#                    INTEGER WHICH
#
#     P <--> The cumulation from 0 to S of the  negative
#            binomial distribution.
#            Input range: [0,1].
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Input range: (0, 1].
#            P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#     S <--> The upper limit of cumulation of the binomial distribution.
#            There are F or fewer failures before the XNth success.
#            Input range: [0, +infinity).
#            Search range: [0, 1E100]
#                    DOUBLE PRECISION S
#
#     XN  <--> The number of successes.
#              Input range: [0, +infinity).
#              Search range: [0, 1E100]
#                    DOUBLE PRECISION XN
#
#     PR  <--> The probability of success in each binomial trial.
#              Input range: [0,1].
#              Search range: [0,1].
#                    DOUBLE PRECISION PR
#
#     OMPR  <--> 1-PR
#              Input range: [0,1].
#              Search range: [0,1]
#              PR + OMPR = 1.0
#                    DOUBLE PRECISION OMPR
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                4 if PR + OMPR .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula   26.5.26   of   Abramowitz  and  Stegun,  Handbook   of
#     Mathematical Functions (1966) is used  to  reduce calculation of
#     the cumulative distribution  function to that of  an  incomplete
#     beta.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
*/

inline std::tuple<double, double, int, double> cdfnbn_which1(
    double s, double xn, double pr, double ompr){
    double p, q;

    if (! ((0 <= s) && (s <= xn))){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (xn >= 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    if (! ((0 <= pr) && (pr < 1.))){
        return std::make_tuple(0., 0., -3, !(pr > 0.) ? 0. : 1.);
    }
    if (! ((0 <= ompr) && (ompr < 1.))){
        return std::make_tuple(0., 0., -4, !(ompr > 0.) ? 0. : 1.);
    }
    if (((abs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 0., 4, (pr+ompr) < 0 ? 0. : 1.);
    }

    std::tie(p, q) = cumnbn(s, xn, pr, ompr);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, int, double> cdfnbn_which2(
    double p, double q, double xn, double pr, double ompr){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 0.;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (xn >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! ((0 <= pr) && (pr <= 1.))){
        return std::make_tuple(0., -4, !(pr > 0.) ? 0. : 1.);
    }
    if (! ((0 <= ompr) && (ompr <= 1.))){
        return std::make_tuple(0., -5, !(ompr > 0.) ? 0. : 1.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }
    if (((abs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 4, (pr+ompr) < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumnbn(DS.x, xn, pr, ompr);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  xn);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfnbn_which3(
    double p, double q, double s, double pr, double ompr){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 0.;
    DS.big = 1e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (s >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! ((0 <= pr) && (pr <= 1.))){
        return std::make_tuple(0., -4, !(pr > 0.) ? 0. : 1.);
    }
    if (! ((0 <= ompr) && (ompr <= 1.))){
        return std::make_tuple(0., -5, !(ompr > 0.) ? 0. : 1.);
    }

    if (((abs(pr+ompr)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 4, (pr+ompr) < 0 ? 0. : 1.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumnbn(s, DS.x, pr, ompr);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, double, int, double> cdfnbn_which4(
    double p, double q, double s, double xn){
    double ccum, cum, pr, ompr;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    // Cython doesn't allow for default values in structs
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0};
    DZ.xlo = 0.;
    DZ.xhi = 1.;
    DZ.atol = atol;
    DZ.rtol = tol;
    DZ.x = 0.;
    DZ.b = 0.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., 0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., 0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (s >= 0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }
    if (! (xn >= 0.)){
        return std::make_tuple(0., 0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 0., 4, p + q < 0 ? 0. : 1.);
    }

    if (qporq){
        dzror(&DZ);
        ompr = 1. - DZ.x;
        while (DZ.status == 1){
            std::tie(cum, ccum) = cumnbn(s, xn, DZ.x, ompr);
            DZ.fx = cum - p;
            dzror(&DZ);
            ompr = 1. - DZ.x;
        }
        pr = DZ.x;
    }
    else{
        dzror(&DZ);
        pr = 1. - DZ.x;
        while (DZ.status == 1){
            std::tie(cum, ccum) = cumnbn(s, xn, pr, DZ.x);
            DZ.fx = ccum - q;
            dzror(&DZ);
            pr = 1. - DZ.x;
        }
        ompr = DZ.x;
    }

    if (DZ.status == -1){
        return std::make_tuple(pr, ompr, DZ.qleft ? 1 : 2, DZ.qleft ? 0. : 1.);
    }
    else{
        return std::make_tuple(pr, ompr, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdfnor_whichX
#
#               Cumulative Distribution Function
#               NORmal distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the normal
#     distribution given values for the others.
#
#
#                              Arguments
#
#
#     WHICH  --> Integer indicating  which of the  next  parameter
#     values is to be calculated using values  of the others.
#     Legal range: 1..4
#               iwhich = 1 : Calculate P and Q from X,MEAN and SD
#               iwhich = 2 : Calculate X from P,Q,MEAN and SD
#               iwhich = 3 : Calculate MEAN from P,Q,X and SD
#               iwhich = 4 : Calculate SD from P,Q,X and MEAN
#                    INTEGER WHICH
#
#     P <--> The integral from -infinity to X of the normal density.
#            Input range: (0,1].
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Input range: (0, 1].
#            P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#     X < --> Upper limit of integration of the normal-density.
#             Input range: ( -infinity, +infinity)
#                    DOUBLE PRECISION X
#
#     MEAN <--> The mean of the normal density.
#               Input range: (-infinity, +infinity)
#                    DOUBLE PRECISION MEAN
#
#     SD <--> Standard Deviation of the normal density.
#             Input range: (0, +infinity).
#                    DOUBLE PRECISION SD
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#
#
#     A slightly modified version of ANORM from
#
#     Cody, W.D. (1993). "ALGORITHM 715: SPECFUN - A Portabel FORTRAN
#     Package of Special Function Routines and Test Drivers"
#     acm Transactions on Mathematical Software. 19, 22-32.
#
#     is used to calculate the cumulative standard normal distribution.
#
#     The rational functions from pages  90-95  of Kennedy and Gentle,
#     Statistical  Computing,  Marcel  Dekker, NY,  1980 are  used  as
#     starting values to Newton's Iterations which compute the inverse
#     standard normal.  Therefore no  searches  are necessary for  any
#     parameter.
#
#     For X < -15, the asymptotic expansion for the normal is used  as
#     the starting value in finding the inverse standard normal.
#     This is formula 26.2.12 of Abramowitz and Stegun.
#
#
#                              Note
#
#
#      The normal density is proportional to
#      exp( - 0.5 * (( X - MEAN)/SD)**2)
#
*/

inline std::tuple<double, double, int, double> cdfnor_which1(
    double x, double mean, double sd){
    double z, p, q;
    if (! (sd > 0.)){
        return std::make_tuple(0., 0., -3, 0.);
    }
    z = (x-mean)/sd;
    std::tie(p, q) = cumnor(z);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, int, double> cdfnor_which2(
    double p, double q, double mean, double sd){
    double z;
    if (! (sd > 0.)){
        return std::make_tuple(0., -4, 0.);
    }
    z = dinvnr(p, q);
    return std::make_tuple(sd*z + mean, 0, 0.);
}

inline std::tuple<double, int, double> cdfnor_which3(
    double p, double q, double x, double sd){
    double z;
    if (! (sd > 0.)){
        return std::make_tuple(0., -4, 0.);
    }
    z = dinvnr(p, q);
    return std::make_tuple(x - sd*z, 0, 0.);
}

inline std::tuple<double, int, double> cdfnor_which4(
    double p, double q, double x, double mean){
    double z;
    z = dinvnr(p, q);
    return std::make_tuple((x-mean)/z, 0, 0.);
}

/*
# %% ---------------------------------------- cdfpoi_whichX
#
#               Cumulative Distribution Function
#               POIsson distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the Poisson
#     distribution given values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which  argument
#               value is to be calculated from the others.
#               Legal range: 1..3
#               iwhich = 1 : Calculate P and Q from S and XLAM
#               iwhich = 2 : Calculate S from P,Q and XLAM
#               iwhich = 3 : Calculate XLAM from P,Q and S
#                    INTEGER WHICH
#
#        P <--> The cumulation from 0 to S of the poisson density.
#               Input range: [0,1].
#                    DOUBLE PRECISION P
#
#        Q <--> 1-P.
#               Input range: (0, 1].
#               P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#        S <--> Upper limit of cumulation of the Poisson.
#               Input range: [0, +infinity).
#               Search range: [0,1E100]
#                    DOUBLE PRECISION S
#
#     XLAM <--> Mean of the Poisson distribution.
#               Input range: [0, +infinity).
#               Search range: [0,1E100]
#                    DOUBLE PRECISION XLAM
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula   26.4.21  of   Abramowitz  and   Stegun,   Handbook  of
#     Mathematical Functions (1966) is used  to reduce the computation
#     of  the cumulative distribution function to that  of computing a
#     chi-square, hence an incomplete gamma function.
#
#     Cumulative  distribution function  (P) is  calculated  directly.
#     Computation of other parameters involve a search for a value that
#     produces  the desired value of  P.   The  search relies  on  the
#     monotinicity of P with the other parameter.
#
*/

inline std::tuple<double, double, int, double> cdfpoi_which1(
    double s, double xlam){
    double p, q;
    if (! (s >= 0.)){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (xlam >= 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    std::tie(p, q) = cumpoi(s, xlam);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, int, double> cdfpoi_which2(
    double p, double q, double xlam){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 0.;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (xlam >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    if ((xlam < 0.01) && (p < 0.975)){
        return std::make_tuple(0., 0, 0.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumpoi(DS.x, xlam);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdfpoi_which3(
    double p, double q, double s){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 0.;
    DS.big = 1e300;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (s >= 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumpoi(s, DS.x);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdft_whichX
#
#               Cumulative Distribution Function
#                         T distribution
#
#
#                              Function
#
#
#     Calculates any one parameter of the t distribution given
#     values for the others.
#
#
#                              Arguments
#
#
#     WHICH --> Integer indicating which  argument
#               values is to be calculated from the others.
#               Legal range: 1..3
#               iwhich = 1 : Calculate P and Q from T and DF
#               iwhich = 2 : Calculate T from P,Q and DF
#               iwhich = 3 : Calculate DF from P,Q and T
#                    INTEGER WHICH
#
#        P <--> The integral from -infinity to t of the t-density.
#              Input range: (0,1].
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Input range: (0, 1].
#            P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#        T <--> Upper limit of integration of the t-density.
#               Input range: ( -infinity, +infinity).
#               Search range: [ -1E100, 1E100 ]
#                    DOUBLE PRECISION T
#
#        DF <--> Degrees of freedom of the t-distribution.
#                Input range: (0 , +infinity).
#                Search range: [1e-100, 1E10]
#                    DOUBLE PRECISION DF
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#
#                              Method
#
#
#     Formula  26.5.27  of   Abramowitz   and  Stegun,   Handbook   of
#     Mathematical Functions  (1966) is used to reduce the computation
#     of the cumulative distribution function to that of an incomplete
#     beta.
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
*/

inline std::tuple<double, double, int, double> cdft_which1(
    double t, double df){
    double p, q;
    if (! (df > 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }
    std::tie(p, q) = cumt(t, df);
    return std::make_tuple(p, q, 0, 0.);
}


inline std::tuple<double, int, double> cdft_which2(
    double p, double q, double df){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = -1.e100;
    DS.big = 1.e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = dt1(p, q, df);

    if (! ((0 < p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (! (df > 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumt(DS.x, df);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft? -1.e100 : 1.e100);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdft_which3(
    double p, double q, double t){
    double ccum, cum;
    double tol = 1e-8;
    double atol = 1e-50;
    bool qporq = p <= q;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};
    DS.small = 1e-100;
    DS.big = 1e10;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 < p) && (p <= 1.)) ){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! ((0 <= q) && (q <= 1.))){
        return std::make_tuple(0., -2, !(q > 0.) ? 0 : 1);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, ccum) = cumt(t, DS.x);
        DS.fx = qporq ? cum - p : ccum - q;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 1e-100 : 1e10);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

/*
# %% ---------------------------------------- cdftnc_whichX
#
#               Cumulative Distribution Function
#                  Non-Central T distribution
#
#                               Function
#
#     Calculates any one parameter of the noncentral t distribution give
#     values for the others.
#
#                               Arguments
#
#     WHICH --> Integer indicating which  argument
#               values is to be calculated from the others.
#               Legal range: 1..3
#               iwhich = 1 : Calculate P and Q from T,DF,PNONC
#               iwhich = 2 : Calculate T from P,Q,DF,PNONC
#               iwhich = 3 : Calculate DF from P,Q,T
#               iwhich = 4 : Calculate PNONC from P,Q,DF,T
#                    INTEGER WHICH
#
#        P <--> The integral from -infinity to t of the noncentral t-den
#              Input range: (0,1].
#                    DOUBLE PRECISION P
#
#     Q <--> 1-P.
#            Input range: (0, 1].
#            P + Q = 1.0.
#                    DOUBLE PRECISION Q
#
#        T <--> Upper limit of integration of the noncentral t-density.
#               Input range: ( -infinity, +infinity).
#               Search range: [ -1E100, 1E100 ]
#                    DOUBLE PRECISION T
#
#        DF <--> Degrees of freedom of the noncentral t-distribution.
#                Input range: (0 , +infinity).
#                Search range: [1e-100, 1E10]
#                    DOUBLE PRECISION DF
#
#     PNONC <--> Noncentrality parameter of the noncentral t-distributio
#                Input range: [-1e6, 1E6].
#
#     STATUS <-- 0 if calculation completed correctly
#               -I if input parameter number I is out of range
#                1 if answer appears to be lower than lowest
#                  search bound
#                2 if answer appears to be higher than greatest
#                  search bound
#                3 if P + Q .ne. 1
#                    INTEGER STATUS
#
#     BOUND <-- Undefined if STATUS is 0
#
#               Bound exceeded by parameter number I if STATUS
#               is negative.
#
#               Lower search bound if STATUS is 1.
#
#               Upper search bound if STATUS is 2.
#
#                                Method
#
#     Upper tail    of  the  cumulative  noncentral t is calculated usin
#     formulae  from page 532  of Johnson, Kotz,  Balakrishnan, Coninuou
#     Univariate Distributions, Vol 2, 2nd Edition.  Wiley (1995)
#
#     Computation of other parameters involve a search for a value that
#     produces  the desired  value  of P.   The search relies  on  the
#     monotinicity of P with the other parameter.
*/

inline std::tuple<double, double, int, double> cdftnc_which1(
    double t, double df, double pnonc){
    double p, q;
    if (! (t == t)){
        return std::make_tuple(0., 0., -1, 0.);
    }
    if (! (df > 0.)){
        return std::make_tuple(0., 0., -2, 0.);
    }

    df = std::min(df, 1.e10);
    t = std::max(std::min(t, spmpar[2]), -spmpar[2]);

    if (! ((-1.e-6 <= pnonc) && ( pnonc <= 1.e6))){
        return std::make_tuple(0., 0., -3, pnonc > -1e6 ? 1.e6 : -1e6);
    }

    std::tie(p, q) = cumtnc(t, df, pnonc);
    return std::make_tuple(p, q, 0, 0.);
}

inline std::tuple<double, int, double> cdftnc_which2(
    double p, double q, double df, double pnonc){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = -1.e100;
    DS.big = 1.e100;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (df > 0.)){
        return std::make_tuple(0., -3, 0.);
    }
    df = std::min(df, 1.e10);

    if (! ((-1.e-6 <= pnonc) && ( pnonc <= 1.e6))){
        return std::make_tuple(0., -4, pnonc > -1e6 ? 1.e6 : -1e6);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused) = cumf(DS.x, df, pnonc);
        DS.fx = cum - p;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

inline std::tuple<double, int, double> cdftnc_which3(
    double p, double q, double t, double pnonc){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = 0;
    DS.big = 1.e6;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    t = std::max(std::min(t, spmpar[2]), -spmpar[2]);
    if (! (t == t)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! ((-1.e-6 <= pnonc) && ( pnonc <= 1.e6))){
        return std::make_tuple(0., -4, pnonc > -1e6 ? 1.e6 : -1e6);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused) = cumf(t, DS.x, pnonc);
        DS.fx = cum - p;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}


inline std::tuple<double, int, double> cdftnc_which4(
    double p, double q, double t, double df){
    double cum, unused;
    double tol = 1e-8;
    double atol = 1e-50;
    DinvrState DS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DzrorState DZ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0};

    DS.small = -1.e6;
    DS.big = 1.e6;
    DS.absstp = 0.5;
    DS.relstp = 0.5;
    DS.stpmul = 5.;
    DS.abstol = atol;
    DS.reltol = tol;
    DS.x = 5.;

    if (! ((0 <= p) && (p <= 1.))){
        return std::make_tuple(0., -1, !(p > 0.) ? 0 : 1);
    }
    if (! (t == t)){
        return std::make_tuple(0., -3, 0.);
    }
    if (! (df > 0.)){
        return std::make_tuple(0., -4, 0.);
    }
    if (((abs(p+q)-0.5)-0.5) > 3*spmpar[0]){
        return std::make_tuple(0., 3, p + q < 0 ? 0. : 1.);
    }

    t = std::max(std::min(t, spmpar[2]), -spmpar[2]);
    df = std::min(df, 1.e10);

    dinvr(&DS, &DZ);
    while (DS.status == 1){
        std::tie(cum, unused) = cumf(t, df, DS.x);
        DS.fx = cum - p;
        dinvr(&DS, &DZ);
    }

    if (DS.status == -1){
        return std::make_tuple(DS.x, DS.qleft ? 1 : 2, DS.qleft ? 0 :  INFINITY);
    }
    else{
        return std::make_tuple(DS.x, 0, 0.);
    }
}

