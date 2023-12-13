#include<cmath>
#include<limits>

// forward declarations
inline double algdiv(double a, double b);
inline double alngam(double x);
inline double alnrel(double a);
inline double apser(double a, double b, double x, double eps);
inline double basym(double a, double b, double lmbda, double eps);
inline double bcorr(double a0, double b0);
inline double betaln(double a0, double b0);
inline double bfrac(double a, double b, double x, double y, double lmbda, double eps);
inline double brcmp1(int mu, double a, double b, double x, double y);
inline double brcomp(double a, double b, double x, double y);
inline double bup(double a, double b, double x, double y, int n, double eps);

inline double erf(double x);
inline double erfc1(int ind, double x);
inline double esum(int mu, double x);
inline double fpser(double a, double b, double x, double eps);
inline double gam1(double a);
inline double gaminv_helper_30(double a, double s, double y, double z);
inline double gamln(double a);
inline double gamln1(double a);
inline double gamma(double a);
inline double gsumln(double a, double b);
inline double psi(double xx);
inline double rcomp(double a, double x);
inline double rexp(double x);
inline double rlog(double x);
inline double rlog1(double x);

inline double devlpl(double *a, int n, double x);


double spmpar[3] = { std::numeric_limits<double>::epsilon(), //np.finfo(np.float64).eps,
                     -101.,    // np.finfo(np.float64).tiny,
                     -101.    //np.finfo(np.float64).max     # FIXME
};


inline double algdiv(double a, double b){
    /*
         Computation of ln(gamma(b)/gamma(a+b)) when b >= 8

                             --------

         In this algorithm, del(x) is the function defined by
         Ln(gamma(x)) = (x - 0.5)*ln(x) - x + 0.5*ln(2*pi) + del(x).

    */
    double c, d, h, s11, s3, s5, s7, s9, t, u, v, w, x, x2;
    double carr[6] = {0.833333333333333e-01, -0.277777777760991e-02,
                           0.793650666825390e-03, -0.595202931351870e-03,
                           0.837308034031215e-03, -0.165322962780713e-02};

    if (a > b){
        h = b / a;
        c = 1./(1. + h);
        x = h/(1. + h);
        d = a + (b - 0.5);
    }
    else {
        h = a / b;
        c = h/(1. + h);
        x = 1./(1. + h);
        d = b + (a - 0.5);
    }

    // Set sn = (1 - x**n)/(1 - x)
    x2 = x*x;
    s3 = 1. + (x + x2);
    s5 = 1. + (x + x2*s3);
    s7 = 1. + (x + x2*s5);
    s9 = 1. + (x + x2*s7);
    s11 = 1. + (x + x2*s9);

    // Set w = del(b) - del(a + b)
    t = (1. / b / b);
    w = (((((carr[5]*s11
            )*t + carr[4]*s9
           )*t + carr[3]*s7
          )*t + carr[2]*s5
         )*t + carr[1]*s3
        )*t + carr[0];
    w *= c / b ;
    // Combine the results
    u = d * alnrel(a / b) ;
    v = a * (log(b) - 1.) ;
    //return (w - v) - u if (u>v) else (w - u) - v
    return u > v ? (w - v) - u : (w - u) - v;
}


// %%----------------------------------------- alngam
inline double alngam(double x){
    /*
                   Double precision log of the gamma function


                                  Function


         Returns the natural logarithm of gamma(x).


                                  Arguments


         X --> value at which scaled log gamma is to be returned
                        x is double precision


                                  Method


         If x <= 6.0, then use recursion to get x below 3
         then apply rational approximation number 5236 of
         hart et al, computer approximations, john wiley and
         sons, ny, 1968.

         If x > 6.0, then use recursion to get x to at least 12 and
         then use formula 5423 of the same source.
    */
    double prod, xx, result, offset;
    int i, n;
    double scoefn[9] = {0.62003838007127258804e2, 0.36036772530024836321e2,
                             0.20782472531792126786e2, 0.6338067999387272343e1,
                             0.215994312846059073e1, 0.3980671310203570498e0,
                             0.1093115956710439502e0, 0.92381945590275995e-2,
                             0.29737866448101651e-2};
    double scoefd[4] = {0.62003838007126989331e2, 0.9822521104713994894e1,
                             -0.8906016659497461257e1, 0.1000000000000000000e1};
    double coef[5] = {0.83333333333333023564e-1, -0.27777777768818808e-2,
                           0.79365006754279e-3, -0.594997310889e-3, 0.8065880899e-3};

    if (x <= 6.) {
        prod = 1.;
        xx = x;

        if (x > 3.) {
            while(xx > 3.){
                xx -= 1.;
                prod *= xx;
            }
        }

        if (x < 2.){
            while(xx < 2.){
                prod /= xx;
                xx += 1.;
            }
        }

        result = devlpl(scoefn, 9, xx - 2.) / devlpl(scoefd, 4, xx - 2.);
        // Compute rational approximation to gamma(x)
        return log(result * prod);
    }

    offset = 0.5*log(2.*M_PI);
    // if (necessary make x at least 12 and carry correction in offset
    if (x <= 12.){
        n = static_cast<int>(12. - x);
        if (n > 0){
            prod = 1.;
            for(int i=0; i < n; i++){
                prod *= x + i;
            }
            offset -= log(prod);
            xx = x + n;
        }
        else {
            xx = x;
        }
    }
    else {
        xx = x;
    }
    // Compute power series
    result = devlpl(coef, 5, (1./xx / xx)) / xx ;
    result += offset + (xx - 0.5)*log(xx) - xx ;
    return result ;
}




// %%----------------------------------------- alnrel
inline double alnrel(double a){
    /*
    Evaluation of the function ln(1 + a)

    */
    double p[3] = {-0.129418923021993e+01, 0.405303492862024e+00,
                        -0.178874546012214e-01};
    double q[3] = {-0.162752256355323e+01, 0.747811014037616e+00,
                        -0.845104217945565e-01};
    double t, t2, w;

    if (abs(a) > 0.375){
        return log(1. + a);
    }
    else {
        t = a / (a + 2.);
        t2 = t*t;
        w = ((p[2]*t2 + p[1])*t2 + p[0])*t2 + 1.;
        w /= ((q[2]*t2 + q[1])*t2 + q[0])*t2 + 1.;
        return 2*t*w;
    }
}




// %%----------------------------------------- apser
inline double apser(double a, double b, double x, double eps){
    /*
    Apser yields the incomplete beta ratio I_(1-x))(b,a) for
    a <= Min(eps,eps*b), b*x <= 1, And x <= 0.5. Used when
    a is very small. Use only if above inequalities are satisfied.

    */
    double aj, bx, c, j, s, t, tol;
    double g = 0.577215664901532860606512090082;

    bx = b*x;
    t = x - bx;
    if (b*eps > 0.02){
        c = log(bx) + g + t;
    }
    else {
        c = log(x) + psi(b) + g + t;
    }

    tol = 5.*eps*abs(c);
    j = 1.;
    s = 0.;

    while(true) {
        j += 1;
        t *= (x - bx/j);
        aj = t / j;
        s += aj;
        if(abs(aj) <= tol){
            break;
        }
    }

    return -a * (c + s);
}


// %%-------------------------------------- basym
inline double basym(double a, double b, double lmbda, double eps){
    /*
    Asymptotic expansion for ix(a,b) for large a and b.
    Lambda = (a + b)*y - b  and eps is the tolerance used.
    It is assumed that lambda is nonnegative and that
    a and b are greater than or equal to 15.
    */
    double a0[21];
    double b0[21];
    double c[21];
    double d[21];
    double bsum, dsum, f, h, h2, hn, j0, j1, r, r0, r1, s, ssum;
    double t, t0, t1, u, w, w0, z, z0, z2, zn, znm1;
    double e0 = 2. / sqrt(M_PI);
    double e1 = pow(2., -3./2.);
    int i, imj, j, m, mmj, n, num;
    // ****** Num is the maximum value that n can take in the do loop
    //        ending at statement 50. It is required that num be even.
    //        The arrays a0, b0, c, d have dimension num + 1.
    num = 20;

    if (a < b) {
        h = a / b;
        r0 = 1./(1.+h);
        r1 = (b - a) / b;
        w0 = 1. / sqrt(a * (1. + h));
    }
    else {
        h = b / a;
        r0 = 1./(1.+h);
        r1 = (b - a) / a;
        w0 = 1. / sqrt(b * (1. + h));
    }

    f = a*rlog1(-lmbda/a) + b*rlog1(lmbda/b);
    t = exp(-f);
    if(t == 0.) {
        return 0.;
    }
    z0 = sqrt(f);
    z = 0.5*(z0/e1);
    z2 = f + f;

    a0[0] = (2./3.)*r1;
    c[0] = -0.5*a0[0];
    d[0] = -c[0];
    j0 = (0.5/e0)*erfc1(1, z0);
    j1 = e1;
    ssum = j0 + d[0]*w0*j1;

    //s, h2, hn, w, znm1, zn = 1., h*h, 1., w0, z, z2
    s = 1;
    h2 = h*h;
    hn = 1.0;
    w = w0;
    znm1 = z;
    zn = z2;

    //for n in range(2, num+1, 2):
    for(int n=2; i < num; i+=2) {
        hn *= h2;
        a0[n-1] = 2.*r0*(1.+ h*hn)/(n + 2.);
        s += hn;
        a0[n] = 2.*r1*s/(n + 3.);

        //for i in range(n, n+2):
        for (i=n; i < n+2; i++){
            r = -0.5*(i + 1.);
            b0[0] = r*a0[0];

            //for m in range(2, i+1):
            for(m=2; m < i; m++) {
                bsum = 0.;
                //for j in range(1, m):
                for(j=1; j < m; j++) {
                    mmj = m - j;
                    bsum += (j*r - mmj)*a0[j-1]*b0[mmj-1];
                }
                b0[m-1] = r*a0[m-1] + bsum/m;
            }

            c[i-1] = b0[i-1] / (i + 1.);
            dsum = 0.;

            //for j in range(1, i):
            for(j=1; j < i; j++){
                imj = i - j;
                dsum += d[imj-1]*c[j-1];
            }
            d[i-1] = - (dsum+c[i-1]);
        }

        j0 = e1*znm1 + (n-1.)*j0;
        j1 = e1*zn + n*j1;
        znm1 *= z2;
        zn *= z2;
        w *= w0;
        t0 = d[n-1]*w*j0;
        w *= w0;
        t1 = d[n]*w*j1;
        ssum += t0 + t1;
        if ( (abs(t0) + abs(t1)) <= eps*ssum) {
            break;
        }
    }

    u = exp(-bcorr(a, b));

    return e0*t*u*ssum;
}

// %%-------------------------------------- bcorr
inline double bcorr(double a0, double b0){
    /*
    Evaluation of  del(a0) + del(b0) - del(a0 + b0)  where
    ln(gamma(a)) = (a - 0.5)*ln(a) - a + 0.5*ln(2*pi) + del(a).
    It is assumed that a0 >= 8 And b0 >= 8.
    */
    double a,b,c,h,s11,s3,s5,s7,s9,t,w,x,x2;
    double carr[9] = {0.833333333333333e-01, -0.277777777760991e-02,
                           0.793650666825390e-03, -0.595202931351870e-03,
                           0.837308034031215e-03, -0.165322962780713e-02};

    //a, b = min(a0, b0), max(a0, b0);
    a = std::min(a0, b0);
    b = std::max(a0, b0);

    h = a / b;
    c = h/(1. + h);
    x = 1./(1. + h);
    x2 = x*x;
    //  Set sn = (1 - x**n)/(1 - x)
    s3 = 1. + (x + x2);
    s5 = 1. + (x + x2*s3);
    s7 = 1. + (x + x2*s5);
    s9 = 1. + (x + x2*s7);
    s11 = 1. + (x + x2*s9);
    // Set w = del(b) - del(a + b)
    t = (1. / b / b);
    w = (((((carr[5]*s11
            )*t + carr[4]*s9
           )*t + carr[3]*s7
          )*t + carr[2]*s5
         )*t + carr[1]*s3
        )*t + carr[0];
    w *= c / b;
    // Compute  del(a) + w
    t = (1. / a / a);
    return ((((((carr[5])*t + carr[4]
               )*t + carr[3]
              )*t + carr[2]
             )*t + carr[1]
            )*t + carr[0]
           )/a + w;
}

// %% betaln ----------------------------------------- betaln
inline double betaln(double a0, double b0){
    /*
    Evaluation of the logarithm of the beta function
    */
    double a, b, c, h, u, v, w, z;
    double e = .918938533204673;
    int i, n;

    a = std::min(a0, b0) ;
    b = std::max(a0, b0);
    if(a >= 8.) {
        w = bcorr(a, b);
        h = a / b;
        c = h/(1. + h);
        u = -(a - 0.5)*log(c);
        v = b*alnrel(h);
        if (u > v){
            return (((-0.5*log(b)+e)+w)-v) - u;
        }
        else {
            return (((-0.5*log(b)+e)+w)-u) - v;
        }
    }

    if (a < 1){
        if (b > 8) {
            return gamln(a) + algdiv(a,b);
        }
        else {
            return gamln(a) + (gamln(b) - gamln(a+b));
        }
    }


    if (a <= 2) {
        if (b <= 2) {
            return gamln(a) + gamln(b) - gsumln(a, b);
        }

        if (b >= 8) {
            return gamln(a) + algdiv(a, b);
        }
        w = 0.;
    }

    if (a > 2) {
        if (b <= 1000){
            n = static_cast<int>(a - 1.);
            w = 1.;
            //for i in range(n):
            for(i=0; i<n; i++){
                a -= 1.;
                h = a / b;
                w *= h/(1.+h);
            }
            w = log(w);
            if (b >= 8.){
                return w + gamln(a) + algdiv(a, b);
            }
        }
        else {
            n = static_cast<int>(a - 1.);
            w = 1.;
            for(i=0; i<n; i++){
            //for i in range(n):
                a -= 1.;
                w *= a/(1. + (a/b));
            }
            return (log(w) - n*log(b)) + (gamln(a) + algdiv(a, b));
        }
    }

    n = static_cast<int>(b - 1.);
    z = 1.;
    //for i in range(n):
    for(int i=0; i<n; i++){
        b -= 1.;
        z *= b / (a + b);
    }
    return w + log(z) + (gamln(a) + gamln(b) - gsumln(a, b));
}


// %%----------------------------------------- bfrac
inline double bfrac(double a, double b, double x, double y,
                         double lmbda, double eps){
    /*
    Continued fraction expansion for ix(a,b) when a,b > 1.
    It is assumed that  lambda = (a + b)*y - b.
    */
    double alpha, beta, e, r0, t, w, result;
    double c = 1. + lmbda;
    double c0 = b / a;
    double c1 = 1. + (1. / a);
    double yp1 = y + 1.;
    double n = 0.;
    double p = 1.;
    double s = a + 1.;
    double an = 0.;
    double bn = 1.;
    double anp1 = 1.;
    double bnp1 = c / c1;
    double r = c1 / c;

    result = brcomp(a, b, x, y);

    if (result == 0.){
        return 0;
    }

    // Continued fraction calculation
    while (true){
        n += 1.;
        t = n / a;
        w = n * (b - n)*x;
        e = a / s;
        alpha = (p*(p + c0)*e*e) * (w*x);
        e = (1. + t) / (c1 + t + t);
        beta = n + (w / s) + e*(c + n*yp1);
        p = 1. + t;
        s += 2.;
        // Update an, bn, anp1, and bnp1
        t = alpha*an + beta*anp1;
        an = anp1;
        anp1 = t;
        t = alpha*bn + beta*bnp1;
        bn = bnp1;
        bnp1 = t;
        r0 = r;
        r = anp1 / bnp1;
        if (!(abs(r - r0) > eps*r)){
            break;
        }
        // Rescale an, bn, anp1, and bnp1
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.;
    }

    return result*r;
}


// %%----------------------------------------- brcmp1
inline double brcmp1(int mu, double a, double b, double x, double y){
    /*
    Evaluation of  exp(mu) * (x**a*y**b/beta(a,b))
    */
    double a0, apb, b0, c, e, h, lmbda, t, u, v, x0, y0, z;
    double lnx = 1., lny = 1.;
    int i, n;
    double cnst = 1./sqrt(2.*M_PI);

    a0 = std::min(a, b);
    if (a0 >= 8.){
        if (a > b){
            h = b / a;
            x0 = 1. / (1. + h);
            y0 = h / (1. + h);
            lmbda = (a + b)*y - b;
        }
        else {
            h = a / b;
            x0 = h / (1. + h);
            y0 = 1. / (1. + h);
            lmbda = a - (a + b)*x;
        }

        e = -lmbda / a;
        if (abs(e) > 0.6){
            u = e - log(x / x0);
        }
        else{
            u = rlog1(e);
        }

        e = lmbda / b;
        if (abs(e) > 0.6){
            v = e - log(y / y0);
        }
        else{
            v = rlog1(e);
        }

        z = esum(mu, -(a*u + b*v));
        return cnst*sqrt(b*x0)*z*exp(-bcorr(a, b));
    }

    if ((x > 0.375) && (y > 0.375)){
        lnx = log(x);
        lny = log(y);
    }
    else if ((x > 0.375) && (y <= 0.375)){
        lnx = alnrel(-y);
        lny = log(y);
    }
    else if (x <= 0.375){
        lnx = log(x);
        lny = alnrel(-x);
    }

    z = a*lnx + b*lny;

    if (a0 >= 1.){
        z -= betaln(a, b);
        return esum(mu, z);
    }

    b0 = std::max(a, b);
    if (b0 >= 8.){
        u = gamln1(a0) + algdiv(a0, b0);
        return a0*esum(mu, z - u);
    }

    if (b0 > 1.){
        u = gamln1(a0);
        n = static_cast<int>(b0 - 1.);
        if (n >= 1){
            c = 1.;
            for(int i=0; i<n; i++){
                b0 -= 1.;
                c *= b0 / (a0 + b0);
            }
            u += log(c);
        }

        z -= u;
        b0 -= 1.;
        apb = a0 + b0;

        if (apb > 1.){
            u = a0 + b0 - 1.;
            t = (1. + gam1(u)) / apb;
        }
        else{
            t = 1. + gam1(apb);
        }
        return a0*esum(mu, z)*(1. + gam1(b0)) / t;
    }

    if (esum(mu, z) == 0.){
        return 0.;
    }

    apb = a + b;
    t = exp(z);
    if (apb > 1.){
        u = a + b - 1.;
        z = (1. + gam1(u)) / apb;
    }
    else{
        z = 1. + gam1(apb);
    }

    c = (1. + gam1(a)) * (1. + gam1(b)) / z;
    return t*(a0*c) / (1. + a0 / b0);

}

// %%----------------------------------------- brcomp
inline double brcomp(double a, double b, double x, double y){
    /*
    Evaluation of x**a*y**b/beta(a,b)
    */
    double a0, apb, b0, c, e, h, lmbda, lnx, lny, t, u, v, x0, y0, z;
    double cnst = 1. / sqrt(2 * M_PI);
    int i, n;

    if ((x == 0.) || (y == 0.)){
        return 0.;
    }

    a0 = std::min(a, b);

    if (a0 >= 8.){
        if (a > b){
            h = b / a;
            x0 = 1. / (1. + h);
            y0 = h / (1. + h);
            lmbda = (a + b)*y - b;
        }
        else{
            h = a / b;
            x0 = h / (1. + h);
            y0 = 1. / (1. + h);
            lmbda = a - (a + b)*x;
        }

        e = -lmbda / a;
        if (abs(e) > 0.6){
            u = e - log(x / x0);
        }
        else{
            u = rlog1(e);
        }

        e = lmbda / b;
        if (abs(e) > 0.6){
            v = e - log(y / y0);
        }
        else{
            v = rlog1(e);
        }

        z = exp(-(a*u + b*v));
        return cnst*sqrt(b*x0)*z*exp(-bcorr(a, b));
    }

    if (x <= 0.375){
        lnx = log(x);
        lny = alnrel(-x);
    }
    else{
        lnx = y > 3.75 ? log(x) : alnrel(-y);
        lny = log(y);
    }

    z = a*lnx + b*lny;
    if (a0 >= 1.){
        z -= betaln(a, b);
        return exp(z);
    }

    b0 = std::max(a, b);
    if (b0 >= 8.){
        u = gamln1(a0) + algdiv(a0, b0);
        return a0*exp(z - u);
    }

    if (b0 > 1.){
        u = gamln1(a0);
        n = static_cast<int>(b0 - 1.);
        if (n >= 1){
            c = 1.;
            for(int i=0; i<n; i++){
                b0 -= 1.;
                c *= b0 / (a0 + b0);
            }
            u += log(c);
        }

        z -= u;
        b0 -= 1.;
        apb = a0 + b0;

        if (apb > 1.){
            u = a0 + b0 - 1.;
            t = (1. + gam1(u)) / apb;
        }
        else {
            t = 1. + gam1(apb);
        }
        return a0*exp(z)*(1. + gam1(b0)) / t;
    }
    if (exp(z) == 0.){
        return 0.;
    }

    apb = a + b;
    t = exp(z);
    if (apb > 1.){
        u = a + b - 1.;
        z = (1. + gam1(u)) / apb;
    }
    else{
        z = 1. + gam1(apb);
    }

    c = (1. + gam1(a)) * (1. + gam1(b)) / z;
    return t * (a0*c) / (1. + a0 / b0);
}

// %%----------------------------------------- bup
inline double bup(double a, double b, double x, double y,
                       int n, double eps){
    /*
    Evaluation of Ix(a,b) - Ix(a+n,b) where n is a positive integer.
    Eps is the tolerance used.
    */
    double apb = a + b;
    double ap1 = a + 1.;
    double d = 1.;
    double r, t, w, result;
    int i, nm1;
    int k = 0;
    int mu = 0;

    if (!((n == 1) || (a < 1.) || (apb < 1.1*ap1))){
        mu = 708;
        t = mu;
        d = exp(-708);
    }

    result = brcmp1(mu, a, b, x, y) / a;
    if ((n == 1) || (result == 0.)){
        return result;
    }
    nm1 = n - 1;
    w = d;

    k = 0;
    if (b <= 1.){
        // 50
        for(int i=0; i<n-1; i++){
            d *= ((apb + i) / (ap1 + i))*x;
            w += d;
            if (d <= eps*w){
                break;
            }
        }
        return result*w;
    }

    if (y > 1e-4){
        // 20
        r = (b - 1.)*x/y - a;
        if (r < 1.){
            // 50
            for(int i=0; i<n-1; i++){
                d *= ((apb + i) / (ap1 + i))*x;
                w += d;
                if (d <= eps*w){
                    break;
                }
            }
            return result*w;
        }

        k = nm1;
        t = nm1;
        if (r < t){
            k = static_cast<int>(r);
        }
    }
    else{
        k = nm1;
    }

    // 30
    for(int i=0; i<k; i++){
        d *= ((apb + i) / (ap1 + i))*x;
        w += d;
    }

    if (k == nm1){
        return result*w;
    }

    // 50
    for(int i=k; i<n-1; i++){
        d *= ((apb + i) / (ap1 + i))*x;
        w += d;
        if (d <= eps*w){
            break;
        }
    }

    return result*w;
}



// %%-------------------------------------- erf
inline double erf(double x){
    /*
    Evaluation of the real error function
    */
    double ax, bot, t, top;
    double c = .564189583547756;
    double a[5] = {.771058495001320e-04, -.133733772997339e-02,
                        .323076579225834e-01, .479137145607681e-01,
                        .128379167095513e+00};
    double b[3] = {.301048631703895e-02, .538971687740286e-01,
                        .375795757275549e+00};
    double p[8] = {-1.36864857382717e-07, 5.64195517478974e-01,
                        7.21175825088309e+00, 4.31622272220567e+01,
                        1.52989285046940e+02, 3.39320816734344e+02,
                        4.51918953711873e+02, 3.00459261020162e+02};
    double q[8] = {1.00000000000000e+00, 1.27827273196294e+01,
                        7.70001529352295e+01, 2.77585444743988e+02,
                        6.38980264465631e+02, 9.31354094850610e+02,
                        7.90950925327898e+02, 3.00459260956983e+02};
    double r[5] = {2.10144126479064e+00, 2.62370141675169e+01,
                        2.13688200555087e+01, 4.65807828718470e+00,
                        2.82094791773523e-01};
    double s[4] = {9.41537750555460e+01, 1.87114811799590e+02,
                        9.90191814623914e+01, 1.80124575948747e+01};

    ax = abs(x);
    if (ax <= 0.5){
        t = x*x;
        top = ((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4]) + 1.;
        bot = ((b[0]*t+b[1])*t+b[2])*t + 1.;
        return x*(top/bot);
    }

    if (ax <= 4.){
        top = (((((((p[0]
                    )*ax+p[1]
                   )*ax+p[2]
                  )*ax+p[3]
                 )*ax+p[4]
                )*ax+p[5]
               )*ax+p[6]
              )*ax + p[7];
        bot = (((((((q[0]
                    )*ax+q[1]
                   )*ax+q[2]
                  )*ax+q[3]
                 )*ax+q[4]
                )*ax+q[5]
               )*ax+q[6])*ax + q[7];
        t = 0.5 + (0.5 - exp(-x*x)*(top/bot));
        return x < 0 ? -t : t;
    }
    if (ax < 5.8){
        t = 1.0 / x / x;
        top = (((r[0]*t+r[1])*t+r[2])*t+r[3])*t + r[4];
        bot = (((s[0]*t+s[1])*t+s[2])*t+s[3])*t + 1.;
        t = 0.5 + (0.5 - exp(-x*x) * (c - top/(x*x*bot))/ax);
        return x < 0 ? -t : t;
    }

    return x < 0 ? -1 : 1;
}



// %%-------------------------------------- erfc1
inline double erfc1(int ind, double x){
    /*
        Evaluation of the complementary error function

        Erfc1(ind,x) = erfc(x)            if ind = 0
        Erfc1(ind,x) = exp(x*x)*erfc(x)   otherwise

    */
    double ax, bot, t, top, result;
    double c = 0.564189583547756;
    double a[5] = {.771058495001320e-04, -.133733772997339e-02,
                        .323076579225834e-01, .479137145607681e-01,
                        .128379167095513e+00};
    double b[3] = {.301048631703895e-02, .538971687740286e-01,
                        .375795757275549e+00};
    double p[8] = {-1.36864857382717e-07, 5.64195517478974e-01,
                        7.21175825088309e+00, 4.31622272220567e+01,
                        1.52989285046940e+02, 3.39320816734344e+02,
                        4.51918953711873e+02, 3.00459261020162e+02};
    double q[8] = {1.00000000000000e+00, 1.27827273196294e+01,
                        7.70001529352295e+01, 2.77585444743988e+02,
                        6.38980264465631e+02, 9.31354094850610e+02,
                        7.90950925327898e+02, 3.00459260956983e+02};
    double r[5] = {2.10144126479064e+00, 2.62370141675169e+01,
                        2.13688200555087e+01, 4.65807828718470e+00,
                        2.82094791773523e-01};
    double s[4] = {9.41537750555460e+01, 1.87114811799590e+02,
                        9.90191814623914e+01, 1.80124575948747e+01};

    if (x <= -5.6){
        return ind != 0 ? 2*exp(x*x) : 2.0;
    }

    // sqrt(log(np.finfo(np.float64).max)) ~= 26.64
    if ((ind == 0) && (x > 26.64)){
        return 0.;
    }

    ax = abs(x);

    if (ax <= 0.5){
        t = x*x;
        top = (((((a[0])*t+a[1])*t+a[2])*t+a[3])*t+a[4]) + 1.;
        bot = (((b[0])*t+b[1])*t+b[2])*t + 1.;
        result = 0.5 + (0.5 - x*(top/bot));
        return ind ==0 ? result : result*exp(t);
    }

    if ((0.5 < ax) && (ax <= 4.)){
        top = (((((((p[0]
                    )*ax+p[1]
                   )*ax+p[2]
                  )*ax+p[3]
                 )*ax+p[4]
                )*ax+p[5]
               )*ax+p[6]
              )*ax + p[7];
        bot = (((((((q[0]
                  )*ax+q[1]
                 )*ax+q[2]
                )*ax+q[3]
               )*ax+q[4]
              )*ax+q[5]
             )*ax+q[6])*ax + q[7];
        result = top / bot;
    }
    else {
        t = (1 / x / x);
        top = (((r[0]*t+r[1])*t+r[2])*t+r[3])*t + r[4];
        bot = (((s[0]*t+s[1])*t+s[2])*t+s[3])*t + 1.;
        result = (c - t*(top/bot)) / ax;
    }

    if (ind == 0){
        result *= exp(-(x*x));
        return x < 0 ? 2.-result : result;
    }
    else {
        return x<0 ? (2.*exp(x*x) - result) : result;
    }
}



// %%----------------------------------------- esum
inline double esum(int mu, double x){
    /*
    Evaluation of exp(mu + x)
    */
    if (x > 0.){
        if ((mu > 0.) || (mu + x < 0)){
            return exp(mu)*exp(x);
        }
        else{
            return exp(mu + x);
        }
    }
    else {
        if ((mu < 0.) || (mu + x > 0.)){
            return exp(mu)*exp(x);
        }
        else{
            return exp(mu + x);
        }
    }
}

// %%----------------------------------------- fpser
inline double fpser(double a, double b, double x, double eps){
    /*
           Evaluation of i_x(a,b)

    for b < Min(eps,eps*a) and x <= 0.5.
    */
    double an, c, s, t, tol;
    double result = 1.;

    result = 1.;
    if (!(a <= 1e-3*eps)){
        result = 0.;
        t = a*log(x);
        if (t < -708.){
            return result;
        }
        result = exp(t);
    }
    //  Note that 1/Beta(a,b) = b
    result *= (b / a);
    tol = eps / a;
    an = a + 1;
    t = x;

    s = t / an;
    while(true) {
        an += 1;
        t *= x;
        c = t / an;
        s += c;
        if (!(abs(c) > tol)){
            break;
        }
    }

    return result*(1. + a*s);
}

// %%----------------------------------------- gam1
inline double gam1(double a){
    /*
    Computation of 1/gamma(a+1) - 1  for -0.5 <= A <= 1.5
    */
    double bot, d, t, top, w;
    double p[7] = {.577215664901533e+00, -.409078193005776e+00,
                        -.230975380857675e+00, .597275330452234e-01,
                        .766968181649490e-02, -.514889771323592e-02,
                        .589597428611429e-03};
    double q[5] = {.100000000000000e+01, .427569613095214e+00,
                        .158451672430138e+00, .261132021441447e-01,
                        .423244297896961e-02};
    double r[9] = {-.422784335098468e+00, -.771330383816272e+00,
                        -.244757765222226e+00, .118378989872749e+00,
                        .930357293360349e-03, -.118290993445146e-01,
                        .223047661158249e-02, .266505979058923e-03,
                        -.132674909766242e-03};
    double s[2] = {.273076135303957e+00, .559398236957378e-01};

    d = a - 0.5;
    t = d > 0 ? d-0.5 : a;

    if (t == 0.){
        return 0.;
    }

    if (t < 0){
        top = ((((((((r[8]
                     )*t+r[7]
                    )*t+r[6]
                   )*t+r[5]
                  )*t+r[4]
                 )*t+r[3]
                )*t+r[2]
               )*t+r[1]
              )*t + r[0];
        bot = (s[1]*t + s[0])*t + 1.;
        w = top / bot;
        if (d > 0.){
            return t*w/a;
        }
        else{
            return a * ((w + 0.5) + 0.5);
        }
    }
    top = ((((((p[6]
               )*t+p[5]
              )*t+p[4]
             )*t+p[3]
            )*t+p[2]
           )*t+p[1]
          )*t + p[0];
    bot = ((((q[4]
             )*t+q[3]
            )*t+q[2]
           )*t+q[1]
          )*t + 1.;
    w = top / bot;
    if (d > 0.){
        return (t/a) * ((w - 0.5) - 0.5);
    }
    else{
        return a * w;
    }
}


inline double gaminv_helper_30(double a, double s,
                                    double y, double z){
    double c1, c2, c3, c4, c5;
    c1 = -s*z;
    c2 = -s*(1. + c1);
    c3 = s*((0.5*c1+ (2.-a))*c1 + (2.5-1.5*a));
    c4 = -s*(((c1/3. + (2.5-1.5*a))*c1 + ((a-6.)*a+7.))*c1 + ((11.*a-46.)*a+47.)/6.);
    c5 = -s*((((-c1/4.+ (11.*a-17.)/6.
               )*c1+ ((-3.*a+13.)*a-13.)
              )*c1 + 0.5*(((2.*a-25.)*a+72.)*a-61.)
             )*c1+ (((25.*a-195.)*a+477.)*a-379.)/12.);
    return ((((c5/y+c4)/y+c3)/y+c2)/y+c1) + y;
}

// %%----------------------------------------- gamln
inline double gamln(double a){
    /*
    Evaluation of ln(gamma(a)) for positive a
    */
    double t,w;
    double c[6];
    double d = .418938533204673;
    int i,n;

    c[0] = .833333333333333e-01;
    c[1] = -.277777777760991e-02;
    c[2] = .793650666825390e-03;
    c[3] = -.595202931351870e-03;
    c[4] = .837308034031215e-03;
    c[5] = -.165322962780713e-02;

    if (a <= 0.8){
        return gamln1(a) - log(a);
    }

    if (a <= 2.25){
        t = (a-0.5) - 0.5;
        return gamln1(t);
    }

    if (a < 10) {
        n = static_cast<int>(a - 1.25);
        t = a;
        w = 1.;
        for(int i=0; i<n; i++){
            t -= 1.;
            w *= t;
        }
        return gamln1(t-1.) + log(w);
    }

    t = (1./a/a);
    w = (((((c[5]*t+c[4])*t+c[3])*t+c[2])*t+c[1])*t+c[0])/a;
    return (d + w) + (a-0.5)*(log(a) - 1.);
}


// %%----------------------------------------- gamln1
inline double gamln1(double a){
    /*
    Evaluation of ln(gamma(1 + a)) for -0.2 <= A <= 1.25
    */
    double p[7];
    double q[6];
    double r[6];
    double s[5];
    double bot, top, w, x;

    p[0] = .577215664901533e+00;
    p[1] = .844203922187225e+00;
    p[2] = -.168860593646662e+00;
    p[3] = -.780427615533591e+00;
    p[4] = -.402055799310489e+00;
    p[5] = -.673562214325671e-01;
    p[6] = -.271935708322958e-02;
    q[0] = .288743195473681e+01;
    q[1] = .312755088914843e+01;
    q[2] = .156875193295039e+01;
    q[3] = .361951990101499e+00;
    q[4] = .325038868253937e-01;
    q[5] = .667465618796164e-03;
    r[0] = .422784335098467e+00;
    r[1] = .848044614534529e+00;
    r[2] = .565221050691933e+00;
    r[3] = .156513060486551e+00;
    r[4] = .170502484022650e-01;
    r[5] = .497958207639485e-03;
    s[0] = .124313399877507e+01;
    s[1] = .548042109832463e+00;
    s[2] = .101552187439830e+00;
    s[3] = .713309612391000e-02;
    s[4] = .116165475989616e-03;

    if (a < 0.6){
        top = ((((((p[6]
                   )*a+p[5]
                  )*a+p[4]
                 )*a+p[3]
                )*a+p[2]
               )*a+p[1]
              )*a+p[0];
        bot = ((((((q[5]
                   )*a+q[4]
                  )*a+q[3]
                 )*a+q[2]
                )*a+q[1]
               )*a+q[0]
              )*a+1.;
        w = top/bot;
        return -a*w;
    }
    else {
        x = (a - 0.5) - 0.5;
        top = (((((r[5]
                  )*x+r[4]
                 )*x+r[3]
                )*x+r[2]
               )*x+r[1]
              )*x+r[0];
        bot = (((((s[4]
                  )*x+s[3]
                 )*x+s[2]
                )*x+s[1]
               )*x+s[0]
              )*x+1.;
        w = top/bot;
        return x*w;
    }
}


// %%-------------------------------------- gamma
inline double gamma(double a){
    /*
        Evaluation of the gamma function for real arguments

    Gamma(a) is assigned the value 0 when the gamma function cannot
    be computed.
    */
    double bot, g, lnx, t, top, w, z, result;
    int i, j, m, n;
    double s = 0.;
    double d = 0.5*(log(2.*M_PI) - 1);
    double x = a;
    double p[7] = {.539637273585445e-03, .261939260042690e-02,
                        .204493667594920e-01, .730981088720487e-01,
                        .279648642639792e+00, .553413866010467e+00,
                        1.0};
    double q[7] = {-.832979206704073e-03, .470059485860584e-02,
                        .225211131035340e-01, -.170458969313360e+00,
                        -.567902761974940e-01, .113062953091122e+01,
                        1.0};

    double r[5] = {.820756370353826e-03, -.595156336428591e-03,
                        .793650663183693e-03, -.277777777770481e-02,
                        .833333333333333e-01};

    result = 0.;
    if (abs(a) < 15){
        t = 1.;
        m = static_cast<int>(a) - 1;
        if (m > 0){
            for(int j=0; j<m; j++){
                x -= 1.;
                t *= x;
            }
            x -= 1.;
        }
        else if (m == 0){
            x -= 1.;
        }
        else{
            t = a;
            if (a <= 0.){
                m = -m - 1;
                if (m != 0.){
                    for(int j=0; j<m; j++){
                        x += 1.;
                        t *= x;
                    }
                }
                x += 0.5;
                x += 0.5;
                t *= x;

                if (t == 0.){
                    return result;
                }
            }
            if (abs(t) < 1e-30){
                if (abs(t)*spmpar[2] <= 1.0001){
                    return result;
                }
                return 1./t;
            }
        }

        top = p[0];
        bot = q[0];
        for(int i=0; i<7; i++){
            top *= x;
            top += p[i];
            bot *= x;
            bot += q[i];
        }
        result = top / bot;

        return a < 1. ? result/t : result*t;
    }

    if (abs(a) >= 1.e3){
        return result;
    }

    if (a <= 0.){
        x = -a;
        n = static_cast<int>(x);
        t = x - n;
        if (t > 0.9){
            t = 1. - t;
        }
        s = sin(M_PI*t) / M_PI;
        if (n % 2 == 0){
            s = -s;
        }
        if (s == 0.){
            return result;
        }
    }

    t = (1.0 / x / x);
    g = ((((r[0]*t+r[1])*t+r[2])*t+r[3])*t+r[4]) / x;
    lnx = log(x);
    z = x;
    g = (d + g) + (z -0.5)*(lnx - 1.);
    w = g;
    t = g - w;
    if (w > 0.99999*709){
        return result;
    }
    result = exp(w)*(1. + t);
    return a < 0. ? (1. / (result * s)) / x  : result;
}



// %%----------------------------------------- devlpl
inline double devlpl(double *a, int n, double x){
    /*
            Double precision EVALuate a PoLynomial at X


                            Function


    returns
        A(1) + A(2)*X + ... + A(N)*X**(N-1)


                            Arguments


    A --> Array of coefficients of the polynomial.
                                    A is DOUBLE PRECISION(N)

    N --> Length of A, also degree of polynomial - 1.
                                    N is INTEGER

    X --> Point at which the polynomial is to be evaluated.
                                    X is DOUBLE PRECISION
    */
    double temp = a[n-1];
    int i;

    for(int i=n-2; i >= 0; i--){
    //for i in range(n-2,-1,-1):
        temp = a[i] + temp*x;
    }
    return temp;
}


// %%-------------------------------------- gsumln
inline double gsumln(double a, double b){
    /*
    Evaluation of the function ln(gamma(a + b))
    for 1 <= A <= 2  And  1 <= B <= 2
    */
    double x;

    x = a + b - 2;
    if(x <= 0.25) {
        return gamln1(1. + x);
    }

    if (x <= 1.25){
        return gamln1(x) + alnrel(x);
    }

    return gamln1(x - 1.) + log(x*(1. + x));
}


// %%-------------------------------------- psi_fort
inline double psi(double xx){
    /*
                Evaluation of the digamma function

                          -----------

    Psi(xx) is assigned the value 0 when the digamma function cannot
    be computed.

    The main computation involves evaluation of rational chebyshev
    approximations published in math. Comp. 27, 123-127(1973) By
    cody, strecok and thacher.

    ----------------------------------------------------------------
    Psi was written at Argonne National Laboratory for the FUNPACK
    package of special function subroutines. Psi was modified by
    A.H. Morris (nswc).
    */
    double aug, den, dx0, sgn, upper, w, x, xmax1, xmx0, xsmall, z;
    double p1[7];
    double q1[6];
    double p2[4];
    double q2[4];
    int nq, i;

    dx0 = 1.461632144968362341262659542325721325;
    p1[0] = .895385022981970e-02;
    p1[1] = .477762828042627e+01;
    p1[2] = .142441585084029e+03;
    p1[3] = .118645200713425e+04;
    p1[4] = .363351846806499e+04;
    p1[5] = .413810161269013e+04;
    p1[6] = .130560269827897e+04;
    q1[0] = .448452573429826e+02;
    q1[1] = .520752771467162e+03;
    q1[2] = .221000799247830e+04;
    q1[3] = .364127349079381e+04;
    q1[4] = .190831076596300e+04;
    q1[5] = .691091682714533e-05;
    p2[0] = -.212940445131011e+01;
    p2[1] = -.701677227766759e+01;
    p2[2] = -.448616543918019e+01;
    p2[3] = -.648157123766197e+00;
    q2[0] = .322703493791143e+02;
    q2[1] = .892920700481861e+02;
    q2[2] = .546117738103215e+02;
    q2[3] = .777788548522962e+01;

    xmax1 = 4503599627370496.0;
    xsmall = 1e-9;
    x = xx;
    aug = 0.;
    if (x < 0.5){
        if (abs(x) <= xsmall){
            if (x == 0.){
                return 0.;
            }
            aug = -1./x;
        }
        else {
            // 10
            w = -x;
            sgn = M_PI / 4;
            if (w <= 0.){
                w = -w;
                sgn = -sgn;
            }
            // 20
            if (w >= xmax1){
                return 0.;
            }

            w -= static_cast<int>(w);
            nq = static_cast<int>(w*4.);
            w = 4.*(w - 0.25*nq);

            if (nq % 2 == 1) {
                w = 1. - w;
            }
            z = (M_PI / 4.)*w;

            // if (nq // 2) % 2 == 1:   # XXX: check
            if (static_cast<int>(nq / 2) % 2 == 1) {
                sgn = -sgn;
            }

            //if ((nq + 1) // 2) % 2 == 1:   # XXX: check
            if ( static_cast<int>((nq + 1) / 2) % 2 == 1 ) {
                aug = sgn * (tan(z)*4.);
            }
            else {
                if (z == 0.){
                    return 0.;
                }
                aug = sgn * (4./tan(z));
            }
        }
        x = 1 - x;
    }

    if (x <= 3.) {
        // 50
        den = x;
        upper = p1[0]*x;
        for(int i=0; i<5; i++){
            den = (den + q1[i])*x;
            upper = (upper + p1[i+1])*x;
        }
        den = (upper + p1[6]) / (den + q1[5]);
        xmx0 = x - dx0;
        return (den * xmx0) + aug;
    }
    else {
        // 70
        if (x < xmax1){
            w = 1. / (x*x);
            den = w;
            upper = p2[0]*w;

            //for i in range(3):
            for(int i=0; i<3; i++) {
                den = (den + q2[i])*w;
                upper = (upper + p2[i+1])*w;
            }
            aug += upper / (den + q2[3]) - 0.5/x ;
        }
        return aug + log(x);
    }
}


// %%-------------------------------------- rcomp
inline double rcomp(double a, double x){
    /*
    Evaluation of exp(-x)*x**a/gamma(a)
    */
    double t, t1, u;
    double r2pi = sqrt(1. / (2.*M_PI));

    if (a < 20){
        t = a*log(x) - x;
        return a < 1 ? a*exp(t)*(1. + gam1(a)) : exp(t) / gamma(a);
    }
    else {
        u = x / a;
        if (u == 0.){
            return 0.;
        }
        t = (1.0 / a / a);
        t1 = (((0.75*t-1.)*t+3.5)*t - 105.) / (a*1260);
        t1 -= a*rlog(u);
        return r2pi*sqrt(a)*exp(t1);
    }
}

// %%-------------------------------------- rexp
inline double rexp(double x){
    /*
    Evaluation of the function exp(x) - 1
    */
    double p[2] = {.914041914819518e-09, .238082361044469e-01};
    double q[4] = {-.499999999085958e+00, .107141568980644e+00,
                        -.119041179760821e-01, .595130811860248e-03};
    double w;

    if (abs(x) <= 0.15){
        return x*(((p[1]*x+p[0])*x+1.)/((((q[3]*x+q[2])*x+q[1])*x+q[0])*x + 1.));
    }
    else {
        w = exp(x);
        if (x > 0.){
            return w* (0.5+ (0.5 - 1./w));
        }
        else{
            return (w - 0.5) - 0.5;
        }
    }
}

// %%-------------------------------------- rlog
inline double rlog(double x){
    /*
    Computation of  x - 1 - ln(x)
    */
    double r, t, u, w, w1;
    double a = .566749439387324e-01;
    double b = .456512608815524e-01;
    double p[3] = {.333333333333333, -.224696413112536,
                        .620886815375787e-02};
    double q[2] = {-.127408923933623e+01, .354508718369557};

    if ((x < 0.61) || (x > 1.57)){
        return ((x - 0.5) - 0.5) - log(x);
    }

    if (x < 0.82){
        u = (x - 0.7) / 0.7;
        w1 = a - u*0.3;
    }
    else if (x > 1.18){
        u = 0.75*x - 1.;
        w1 = b + u/3.;
    }
    else {
        u = (x - 0.5) - 0.5;
        w1 = 0.;
    }

    r = u / (u + 2.);
    t = r*r;
    w = ((p[2]*t+p[1])*t+p[0])/ ((q[1]*t+q[0])*t+1.);
    return 2.*t*(1. / (1. - r) - r*w) + w1;
}


// %%-------------------------------------- rlog1
inline double rlog1(double x){
    /*
    Evaluation of the function x - ln(1 + x)
    */
    double a = .566749439387324e-01;
    double b = .456512608815524e-01;
    double p0 = .333333333333333e+00;
    double p1 = -.224696413112536e+00;
    double p2 = .620886815375787e-02;
    double q1 = -.127408923933623e+01;
    double q2 = .354508718369557e+00;
    double r;
    double t;
    double w;
    double w1;
    double h;

    if ((-0.39 <= x) && (x <= 0.57)){
        if ((-0.18 <= x)&& (x <= 0.18)){
            h = x;
            w1 = 0.;
        }
        else if (x < -0.18){
            h = (x + 0.3)/0.7;
            w1 = a - h*0.3;
        }
        else {  // 0.57 >= x > 0.18   // XXX: missing abs() ?
            h = 0.75*x - 0.25;
            w1 = b + h/3.;
        }

        r = h / (h + 2);
        t = r*r;
        w = ((p2*t + p1)*t + p0) / ((q2*t + q1)*t + 1.);
        return 2*t*(1./(1.-r) - r*w) + w1;
    }
    else {
        return x - log((x + 0.5) + 0.5);
    }
}

