#include<cmath>

// forward declarations
inline double algdiv(double a, double b);
inline double alngam(double x);
inline double alnrel(double a);
inline double apser(double a, double b, double x, double eps);
inline double basym(double a, double b, double lmbda, double eps);
inline double bcorr(double a0, double b0);
inline double betaln(double a0, double b0);

inline double devlpl(double *a, int n, double x);
inline double rlog1(double x);



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

