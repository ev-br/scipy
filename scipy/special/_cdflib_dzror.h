// forward declarations
struct DinvrState;
struct DzrorState;

void dinvr(DinvrState *S, DzrorState *DZ);
void dzror(DzrorState *S);

/*
# %% ------------------------------------- dinvr
# 
# Double Precision - SeT INverse finder - Reverse Communication
# 
# 
#                         Function
# 
# 
# Concise Description - Given a monotone function F finds X
# such that F(X) = Y.  Uses Reverse communication -- see invr.
# This routine sets quantities needed by INVR.
# 
#     More Precise Description of INVR -
# 
# F must be a monotone function, the results of QMFINV are
# otherwise undefined.  QINCR must be .TRUE. if F is non-
# decreasing and .FALSE. if F is non-increasing.
# 
# QMFINV will return .TRUE. if and only if F(SMALL) and
# F(BIG) bracket Y, i. e.,
#     QINCR is .TRUE. and F(SMALL)<=Y<=F(BIG) or
#     QINCR is .FALSE. and F(BIG)<=Y<=F(SMALL)
# 
# if QMFINV returns .TRUE., then the X returned satisfies
# the following condition.  let
#         TOL(X) = MAX(ABSTOL,RELTOL*ABS(X))
# then if QINCR is .TRUE.,
#     F(X-TOL(X)) <= Y <= F(X+TOL(X))
# and if QINCR is .FALSE.
#     F(X-TOL(X)) >= Y >= F(X+TOL(X))
# 
# 
#                         Arguments
# 
# 
# SMALL --> The left endpoint of the interval to be
#     searched for a solution.
#             SMALL is DOUBLE PRECISION
# 
# BIG --> The right endpoint of the interval to be
#     searched for a solution.
#             BIG is DOUBLE PRECISION
# 
# ABSSTP, RELSTP --> The initial step size in the search
#     is MAX(ABSSTP,RELSTP*ABS(X)). See algorithm.
#             ABSSTP is DOUBLE PRECISION
#             RELSTP is DOUBLE PRECISION
# 
# STPMUL --> When a step doesn't bound the zero, the step
#         size is multiplied by STPMUL and another step
#         taken.  A popular value is 2.0
#             DOUBLE PRECISION STPMUL
# 
# ABSTOL, RELTOL --> Two numbers that determine the accuracy
#     of the solution.  See function for a precise definition.
#             ABSTOL is DOUBLE PRECISION
#             RELTOL is DOUBLE PRECISION
# 
# 
#                         Method
# 
# 
# Compares F(X) with Y for the input value of X then uses QINCR
# to determine whether to step left or right to bound the
# desired x.  the initial step size is
#     MAX(ABSSTP,RELSTP*ABS(S)) for the input value of X.
# Iteratively steps right or left until it bounds X.
# At each step which doesn't bound X, the step size is doubled.
# The routine is careful never to step beyond SMALL or BIG.  If
# it hasn't bounded X at SMALL or BIG, QMFINV returns .FALSE.
# after setting QLEFT and QHI.
# 
# If X is successfully bounded then Algorithm R of the paper
# 'Two Efficient Algorithms with Guaranteed Convergence for
# Finding a Zero of a Function' by J. C. P. Bus and
# T. J. Dekker in ACM Transactions on Mathematical
# Software, Volume 1, No. 4 page 330 (DEC. '75) is employed
# to find the zero of the function F(X)-Y. This is routine
# QRZERO.
*/

struct DinvrState{
    double absstp;
    double abstol;
    double big;
    double fbig;
    double fx;
    double fsmall;
    double relstp;
    double reltol;
    double small;
    int status;
    double step;
    double stpmul;
    double x;
    double xhi;
    double xlb;
    double xlo;
    double xsave;
    double xub;
    double yy;
    double zx;
    double zy;
    double zz;
    int next_state;
    bool qbdd;
    bool qcond;
    bool qdum1;
    bool qdum2;
    bool qhi;
    bool qleft;
    bool qincr;
    bool qlim;
    bool qok;
    bool qup;
};


struct DzrorState{
    double a;
    double atol;
    double b;
    double c;
    double d;
    double fa;
    double fb;
    double fc;
    double fd;
    double fda;
    double fdb;
    double fx;
    double m;
    double mb;
    double p;
    double q;
    double tol;
    double rtol;
    double w;
    double xhi;
    double xlo;
    double x;
    int ext;
    int status;
    int next_state;
    bool first;
    bool qrzero;
    bool qleft;
    bool qhi;
};


void dinvr(DinvrState *S, DzrorState *DZ){
    /*
        Double precision
        bounds the zero of the function and invokes zror
                Reverse Communication


                            Function


    Bounds the    function  and  invokes  ZROR   to perform the   zero
    finding.  STINVR  must  have   been  called  before this   routine
    in order to set its parameters.


                            Arguments


    STATUS <--> At the beginning of a zero finding problem, STATUS
                should be set to 0 and INVR invoked.  (The value
                of parameters other than X will be ignored on this cal

                When INVR needs the function evaluated, it will set
                STATUS to 1 and return.  The value of the function
                should be set in FX and INVR again called without
                changing any of its other parameters.

                When INVR has finished without error, it will return
                with STATUS 0.  In that case X is approximately a root
                of F(X).

                If INVR cannot bound the function, it returns status
                -1 and sets QLEFT and QHI.
                        INTEGER STATUS

    X <-- The value of X at which F(X) is to be evaluated.
                        DOUBLE PRECISION X

    FX --> The value of F(X) calculated when INVR returns with
        STATUS = 1.
                        DOUBLE PRECISION FX

    QLEFT <-- Defined only if QMFINV returns .FALSE.  In that
        case it is .TRUE. If the stepping search terminated
        unsuccessfully at SMALL.  If it is .FALSE. the search
        terminated unsuccessfully at BIG.
                QLEFT is LOGICAL

    QHI <-- Defined only if QMFINV returns .FALSE.  In that
        case it is .TRUE. if F(X) > Y at the termination
        of the search and .FALSE. if F(X) < Y at the
        termination of the search.
                QHI is LOGICAL
    */
    while(true){
        if (S->next_state == 0){
            // See that SMALL and BIG bound the zero and set QINCR
            S->qcond = (S->small <= S->x) &&( S->x <= S->big);
            if (!S->qcond){
                S->status = -2;
                return;
            }
            S->xsave = S->x;
            S->x = S->small;
            S->next_state = 10;
            S->status = 1;
            return;
        }

        else if (S->next_state == 10){
            S->fsmall = S->fx;
            S->x = S->big;
            S->next_state = 20;
            S->status = 1;
            return;
        }
        else if (S->next_state == 20){
            S->fbig = S->fx;
            S->qincr = S->fbig > S->fsmall;
            S->status = -1;
            if (!S->qincr){
                // 50
                if (S->fsmall >= 0.){
                    // 60
                    if (S->fbig <= 0.){
                        // 70
                        /* pass */
                    }
                    else{
                        S->qleft = false;
                        S->qhi = true;
                        return;
                    }
                }
                else {
                    S->qleft = true;
                    S->qhi = false;
                    return;
                }
            }
            else {
                if (S->fsmall <= 0.){
                    // 30
                    if (S->fbig >= 0.){
                        // 40
                        /* pass  */
                    }
                    else{
                        S->qleft = false;
                        S->qhi = false;
                        return;
                    }
                }
                else {
                    S->qleft = true;
                    S->qhi = true;
                    return;
                }
            }

            S->status = 1;
            S->x = S->xsave;
            S->step = std::max(S->absstp, S->relstp*abs(S->x));
            S->next_state = 90;
            return;
        }
        else if (S->next_state == 90){
            S->yy = S->fx;
            if (S->yy == 0.){
                S->status = 0;
                S->qok = true;
                return;
            }
            S->next_state = 100;
        }

    //  Handle case in which we must step higher
        else if (S->next_state == 100){
            S->qup = ((S->qincr) && (S->yy < 0.)) || ((!S->qincr) && (S->yy > 0.));
            if (S->qup){
                S->xlb = S->xsave;
                S->xub = std::min(S->xlb + S->step, S->big);
                S->next_state = 120;
            }
            else{
                // 170
                S->xub = S->xsave;
                S->xlb = std::max(S->xub - S->step, S->small);
                S->next_state = 190;
            }
        }
        else if (S->next_state == 120){
            S->x = S->xub;
            S->status = 1;
            S->next_state = 130;
            return;
        }
        else if (S->next_state == 130){
            S->yy = S->fx;
            S->qbdd = (S->qincr && (S->yy >= 0.)) || ((!S->qincr) && (S->yy <= 0.));
            S->qlim = (S->xub >= S->big);
            S->qcond = S->qbdd || S->qlim;
            if (S->qcond){
                S->next_state = 150;
            }
            else{
                S->step *= S->stpmul;
                S->xlb = S->xub;
                S->xub = std::min(S->xlb + S->step, S->big);
                S->next_state = 120;
            }
        }
        else if (S->next_state == 150){
            if (S->qlim && (!S->qbdd)){
                S->status = -1;
                S->qleft = false;
                S->qhi = !S->qincr;
                S->x = S->big;
                return;
            }
            else{
                S->next_state = 240;
            }
        }
        else if (S->next_state == 190){
            S->x = S->xlb;
            S->status = 1;
            S->next_state = 200;
            return;
        }
    // Handle case in which we must step lower
        else if (S->next_state == 200){
            S->yy = S->fx;
            S->qbdd = (S->qincr && (S->yy <= 0.)) || ((!S->qincr) && (S->yy >= 0.));
            S->qlim = (S->xlb <= S->small);
            S->qcond = S->qbdd || S->qlim;
            if (S->qcond){
                S->next_state = 220;
            }
            else{
                S->step *= S->stpmul;
                S->xub = S->xlb;
                S->xlb = std::max(S->xub - S->step, S->small);
                S->next_state = 190;
            }
        }
        else if (S->next_state == 220){
            if (S->qlim && (!S->qbdd)){
                S->status = -1;
                S->qleft = true;
                S->qhi = S->qincr;
                S->x = S->small;
                return;
            }
            else{
                S->next_state = 240;
            }
        }

    // If we reach here, xlb and xub bound the zero of f.
        else if (S->next_state == 240){
            // Overwrite supplied DZ with the problem
            DZ->xhi = S->xub;
            DZ->xlo = S->xlb;
            DZ->atol = S->abstol;
            DZ->rtol = S->reltol;
            DZ->x = S->xlb;
            DZ->b = S->xlb;
            S->next_state = 250;
        }
        else if (S->next_state == 250){
            dzror(DZ);
            if (DZ->status == 1){
                S->next_state = 260;
                S->status = 1;
                S->x = DZ->x;
                return;
            }
            else{
                S->x = DZ->xlo;
                S->status = 0;
                return;
            }
        }
        else if (S->next_state == 260){
            DZ->fx = S->fx;
            S->next_state = 250;
        }

        else{
            S->status = -9999;  // Bad state, should not be possible to get here
            return;
        }
    }

}



/*
# %% ------------------------------------- dzror
# 
# Double precision SeT ZeRo finder - Reverse communication version
# 
# 
#                         Function
# 
# 
# 
# Sets quantities needed by ZROR.  The function of ZROR
# and the quantities set is given here.
# 
# Concise Description - Given a function F
# find XLO such that F(XLO) = 0.
# 
#     More Precise Description -
# 
# Input condition. F is a double precision function of a single
# double precision argument and XLO and XHI are such that
#     F(XLO)*F(XHI)  <=  0.0
# 
# If the input condition is met, QRZERO returns .TRUE.
# and output values of XLO and XHI satisfy the following
#     F(XLO)*F(XHI)  <= 0.
#     ABS(F(XLO)  <= ABS(F(XHI)
#     ABS(XLO-XHI)  <= TOL(X)
# where
#     TOL(X) = MAX(ABSTOL,RELTOL*ABS(X))
# 
# If this algorithm does not find XLO and XHI satisfying
# these conditions then QRZERO returns .FALSE.  This
# implies that the input condition was not met.
# 
# 
#                         Arguments
# 
# 
# XLO --> The left endpoint of the interval to be
#     searched for a solution.
#             XLO is DOUBLE PRECISION
# 
# XHI --> The right endpoint of the interval to be
#     for a solution.
#             XHI is DOUBLE PRECISION
# 
# ABSTOL, RELTOL --> Two numbers that determine the accuracy
#                 of the solution.  See function for a
#                 precise definition.
#             ABSTOL is DOUBLE PRECISION
#             RELTOL is DOUBLE PRECISION
# 
# 
#                         Method
# 
# 
# Algorithm R of the paper 'Two Efficient Algorithms with
# Guaranteed Convergence for Finding a Zero of a Function'
# by J. C. P. Bus and T. J. Dekker in ACM Transactions on
# Mathematical Software, Volume 1, no. 4 page 330
# (Dec. '75) is employed to find the zero of F(X)-Y.
#
*/


void dzror(DzrorState *S){
    /*
    Double precision ZeRo of a function -- Reverse Communication


                            Function


    Performs the zero finding.  STZROR must have been called before
    this routine in order to set its parameters.


                            Arguments


    STATUS <--> At the beginning of a zero finding problem, STATUS
                should be set to 0 and ZROR invoked.  (The value
                of other parameters will be ignored on this call.)

                When ZROR needs the function evaluated, it will set
                STATUS to 1 and return.  The value of the function
                should be set in FX and ZROR again called without
                changing any of its other parameters.

                When ZROR has finished without error, it will return
                with STATUS 0.  In that case (XLO,XHI) bound the answe

                If ZROR finds an error (which implies that F(XLO)-Y an
                F(XHI)-Y have the same sign, it returns STATUS -1.  In
                this case, XLO and XHI are undefined.
                        INTEGER STATUS

    X <-- The value of X at which F(X) is to be evaluated.
                        DOUBLE PRECISION X

    FX --> The value of F(X) calculated when ZROR returns with
        STATUS = 1.
                        DOUBLE PRECISION FX

    XLO <-- When ZROR returns with STATUS = 0, XLO bounds the
            inverval in X containing the solution below.
                        DOUBLE PRECISION XLO

    XHI <-- When ZROR returns with STATUS = 0, XHI bounds the
            inverval in X containing the solution above.
                        DOUBLE PRECISION XHI

    QLEFT <-- .TRUE. if the stepping search terminated unsuccessfully
            at XLO.  If it is .FALSE. the search terminated
            unsuccessfully at XHI.
                QLEFT is LOGICAL

    QHI <-- .TRUE. if F(X) > Y at the termination of the
            search and .FALSE. if F(X) < Y at the
            termination of the search.
                QHI is LOGICAL
    */
    while(true){
        if (S->next_state == 0){
            S->next_state = 10;
            S->status = 1;
            return;
        }

        else if (S->next_state == 10){
            S->fb = S->fx;
            S->xlo = S->xhi;
            S->a = S->xlo;
            S->x = S->xlo;
            S->next_state = 20;
            S->status = 1;
            return;
        }

        else if (S->next_state == 20){
            if ((!(S->fb > 0.)) || (!(S->fx > 0.)) ){
                if ((!(S->fb < 0.)) || (!(S->fx < 0.))){
        // 60
                    S->fa = S->fx;
                    S->first = true;
        // 70
                    S->c = S->a;
                    S->fc = S->fa;
                    S->ext = 0;
                    S->status = 1;
                    S->next_state = 80;
                }
                else {
        // 40
                    S->status = -1;
                    S->qleft = S->fx > S->fb;
                    S->qhi = true;
                    return;
                }
            }
            else{
                S->status = -1;
                S->qleft = S->fx < S->fb;
                S->qhi = false;
                return;
            }
        }
        else if (S->next_state == 80){
            if (abs(S->fc) < abs(S->fb)){
                if (S->c != S->a){
                    S->d = S->a;
                    S->fd = S->fa;
                }
            // 90
                S->a = S->b;
                S->fa = S->fb;
                S->xlo = S->c;
                S->b = S->xlo;
                S->fb = S->fc;
                S->c = S->a;
                S->fc = S->fa;
            }
            // 100
            S->tol = 0.5 * std::max(S->atol, S->rtol * abs(S->xlo));
            S->m = (S->c + S->b) * 0.5;
            S->mb = S->m - S->b;

            if (! (abs(S->mb) > S->tol)){
                S->next_state = 240;
            }
            else{
                if (!(S->ext > 3)){
                    S->tol = S->mb < 0 ? -abs(S->tol) : abs(S->tol);
                    S->p = (S->b - S->a)*S->fb;
                    if (S->first){
                        S->q = S->fa - S->fb;
                        S->first = false;
                    }
                    else {
            // 120
                        S->fdb = (S->fd - S->fb)/(S->d - S->b);
                        S->fda = (S->fd - S->fa)/(S->d - S->a);
                        S->p = S->fda*S->p;
                        S->q = S->fdb*S->fa - S->fda*S->fb;
                    }
            // 130
                    if (S->p < 0.){
                        S->p = -S->p;
                        S->q = -S->q;
                    }
            // 140
                    if (S->ext == 3){
                        S->p *= 2.;
                    }

                    if (!((S->p*1. == 0.) || (S->p <= S->q*S->tol)) ){
            // 150
                        if (! (S->p < (S->mb*S->q))){
            // 160
                            S->w = S->mb;
                        }
                        else{
                            S->w = S->p / S->q;
                        }
                    }
                    else {
                        S->w = S->tol;
                    }
                }
                else{
                    S->w = S->mb;
                }
            // 190
                S->d = S->a;
                S->fd = S->fa;
                S->a = S->b;
                S->fa = S->fb;
                S->b = S->b + S->w;
                S->xlo = S->b;
                S->x = S->xlo;
                S->next_state = 200;
                S->status = 1;
                return;
            }
        }
        else if (S->next_state == 200){

            S->fb = S->fx;
            if (!(S->fc*S->fb >= 0.)){
            // 210
                if (!(S->w == S->mb)){
            // 220
                    S->ext += 1;
                }
                else {
                    S->ext = 0;
                }
            }
            else{
                // 70
                S->c = S->a;
                S->fc = S->fa;
                S->ext = 0;
            }
            // 230
            S->next_state = 80;
        }
        else if (S->next_state == 240){
            S->xhi = S->c;
            S->qrzero = (((S->fc >= 0.) && (S->fb <= 0.)) ||
                        ((S->fc < 0.) && (S->fb >= 0.)));

            if (S->qrzero){
                S->status = 0;
            }
            else{
            // 250
                S->status = -1;
            }
            return;
        }
        else{
            S->status = -9999;  // Bad state, should not be possible to get here
            return;
        }
    } // while(true)
}

