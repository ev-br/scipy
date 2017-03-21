#include "mconf.h"
#include <math.h>
#include <float.h>

static double LOG_MAX_VALUE = 709.0;

static int SELECT_METHOD[] = {
    1, 1, 2, 13, 13, 13, 13, 13, 13, 13, 13, 16, 16, 16, 9,
    1, 2, 2, 3, 3, 5, 5, 14, 14, 15, 15, 16, 16, 16, 9,
    2, 2, 3, 3, 3, 5, 5, 15, 15, 15, 15, 16, 16, 16, 10,
    2, 2, 3, 5, 5, 5, 5, 7, 7, 16, 16, 16, 16, 16, 10,
    2, 3, 3, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 11,
    2, 3, 5, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 12,
    2, 3, 4, 4, 6, 6, 8, 8, 17, 17, 17, 17, 17, 12, 12,
    2, 3, 4, 4, 6, 6, 18, 18, 18, 18, 17, 17, 17, 12, 12};

static double HRANGE[] = {0.02, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6,
    1.7, 2.33, 2.4, 3.36, 3.4, 4.8};

static double ARANGE[] = {0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999};

static double ORD[] = {2, 3, 4, 5, 7, 10, 12, 18, 10, 20, 30, 0, 4, 7,
    8, 20, 0, 0};

static int METHODS[] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4,
    5, 6};

static double C[] = {
    0.99999999999999999999999729978162447266851932041876728736094298092917625009873,
    -0.99999999999999999999467056379678391810626533251885323416799874878563998732905968,
    0.99999999999999999824849349313270659391127814689133077036298754586814091034842536,
    -0.9999999999999997703859616213643405880166422891953033591551179153879839440241685,
    0.99999999999998394883415238173334565554173013941245103172035286759201504179038147,
    -0.9999999999993063616095509371081203145247992197457263066869044528823599399470977,
    0.9999999999797336340409464429599229870590160411238245275855903767652432017766116267,
    -0.999999999574958412069046680119051639753412378037565521359444170241346845522403274,
    0.9999999933226234193375324943920160947158239076786103108097456617750134812033362048,
    -0.9999999188923242461073033481053037468263536806742737922476636768006622772762168467,
    0.9999992195143483674402853783549420883055129680082932629160081128947764415749728967,
    -0.999993935137206712830997921913316971472227199741857386575097250553105958772041501,
    0.99996135597690552745362392866517133091672395614263398912807169603795088421057688716,
    -0.99979556366513946026406788969630293820987757758641211293079784585126692672425362469,
    0.999092789629617100153486251423850590051366661947344315423226082520411961968929483,
    -0.996593837411918202119308620432614600338157335862888580671450938858935084316004769854,
    0.98910017138386127038463510314625339359073956513420458166238478926511821146316469589567,
    -0.970078558040693314521331982203762771512160168582494513347846407314584943870399016019,
    0.92911438683263187495758525500033707204091967947532160289872782771388170647150321633673,
    -0.8542058695956156057286980736842905011429254735181323743367879525470479126968822863,
    0.73796526033030091233118357742803709382964420335559408722681794195743240930748630755,
    -0.58523469882837394570128599003785154144164680587615878645171632791404210655891158,
    0.415997776145676306165661663581868460503874205343014196580122174949645271353372263,
    -0.2588210875241943574388730510317252236407805082485246378222935376279663808416534365,
    0.1375535825163892648504646951500265585055789019410617565727090346559210218472356689,
    -0.0607952766325955730493900985022020434830339794955745989150270485056436844239206648,
    0.0216337683299871528059836483840390514275488679530797294557060229266785853764115,
    -0.00593405693455186729876995814181203900550014220428843483927218267309209471516256,
    0.0011743414818332946510474576182739210553333860106811865963485870668929503649964142,
    -1.489155613350368934073453260689881330166342484405529981510694514036264969925132E-4,
    9.072354320794357587710929507988814669454281514268844884841547607134260303118208E-6 };

static double PTS[] = {
    0.35082039676451715489E-02, 0.31279042338030753740E-01,
    0.85266826283219451090E-01, 0.16245071730812277011E+00,
    0.25851196049125434828E+00, 0.36807553840697533536E+00,
    0.48501092905604697475E+00, 0.60277514152618576821E+00,
    0.71477884217753226516E+00, 0.81475510988760098605E+00,
    0.89711029755948965867E+00, 0.95723808085944261843E+00,
    0.99178832974629703586E+00};

static double WTS[] = {
    0.18831438115323502887E-01, 0.18567086243977649478E-01,
    0.18042093461223385584E-01, 0.17263829606398753364E-01,
    0.16243219975989856730E-01, 0.14994592034116704829E-01,
    0.13535474469662088392E-01, 0.11886351605820165233E-01,
    0.10070377242777431897E-01, 0.81130545742299586629E-02,
    0.60419009528470238773E-02, 0.38862217010742057883E-02,
    0.16793031084546090448E-02};

int get_method(double h, double a) {
    int ihint = 14;
    int iaint = 7;
    int i;

    for (i = 0; i < 14; i++) {
        if (h <= HRANGE[i]) {
            ihint = i;
            break;
        }
    }

    for (i = 0; i < 7; i++) {
        if (a <= ARANGE[i]) {
            iaint = i;
            break;
        }
    }

    return SELECT_METHOD[iaint * 15 + ihint];
}

double owens_t_norm1(double x) {
    return erf(x / sqrt(2)) / 2;
}

double owens_t_norm2(double x) {
    return erfc(x / sqrt(2)) / 2;
}

double owensT1(double h, double a, double m) {

    double aa = a * a;
    double nhh = -1 * h * h / 2;
    double h_exp = exp(nhh);

    int i = 0;
    int div = 1;
    double aj = a / (2 * NPY_PI);
    double dj = expm1(nhh);
    double gj = nhh * h_exp;

    double result = atan(a) * (1 / (2 * NPY_PI));

    for (i = 1; i <= m; i++) {
        result += dj * aj / div;
        div += 2;
        aj *= aa;
        dj = gj - dj;
        gj *= nhh / i;
    }

    return result;
}

double owensT2(double h, double a, double ah, double m) {

    double hh = h * h;
    double nah = -0.5 * h * h * a * a;
    double pi_sq = 1 / sqrt(2 * NPY_PI);

    int i = 0;
    double z = (ndtr(ah) - 0.5) / h;

    double result = 0;

    for (i = 0; i <= (2 * m); i++) {
        result += pow(-1, i - 1) * z;
        z = ((2 * i - 1) * z - (pi_sq * pow(a, 2 * i - 1) *
            exp(nah))) / (hh);
    }

    result *= pi_sq * exp(-0.5 * hh);

    return result;
}

double owensT3(double h, double a, double ah) {
    double aa = a * a;
    double hh = h * h;
    double y = 1 / hh;

    int i = 0;
    double vi = a * exp(-ah * ah/ 2) / sqrt(2 * NPY_PI);
    double zi = owens_t_norm1(ah) / h;
    double result = 0;

    for(i = 0; i<= 30; i++) {
        result += zi * C[i];
        zi = y * ((2 * i + 1) * zi - vi);
        vi *= aa;
    }

    result *= exp(-hh / 2) / sqrt(2 * NPY_PI);

    return result;
}

double owensT4(double h, double a, double m) {
    double maxi = 2 * m + 1;
    double hh = h * h;
    double naa = -a * a;

    int i = 1;
    double ai = a * exp(-hh * (1 - naa) / 2) / (2 * NPY_PI);
    double yi = 1;
    double result = 0;

    while (1) {
        result += ai * yi;

        if (maxi <= i) {
            break;
        }

        i += 2;
        yi = (1 - hh * yi) / i;
        ai *= naa;
    }

    return result;
}

double owensT5(double h, double a) {
    double result = 0;

    double aa = a * a;
    double nhh = -0.5 * h * h;
    int i = 0;

    for (i = 1; i < 14; i++) {
        double r = 1 + aa * PTS[i - 1];
        result += WTS[i - 1] * exp(nhh * r) / r;
    }

    result *= a;

    return result;
}

double owensT6(double h, double a) {

    double normh = owens_t_norm2(h);
    double y = 1 - a;
    double r = atan2(y, (1 + a));

    double result = normh * (1 - normh) / 2;

    if (r != 0) {
        result -= r * exp(-y * h * h / (2 * r)) / (2 * NPY_PI);
    }

    return result;
}

double owensT1_accelerated(double h, double a, double target_precision) {
    double hh_half = h * h / 2;
    double a_pow = a;
    double aa = a * a;
    double exp_term = exp(-hh_half);
    double one_minus_dj_sum = exp_term;
    double sum = a_pow * exp_term;
    double dj_pow = exp_term;
    double term = sum;
    double abs_err;
    int j = 1;

    int n = trunc(LOG_MAX_VALUE / 6);

    double d = pow(3 + sqrt(8), n);
    d = (d + 1 / d) / 2;
    double b = -1;
    double c = -d;
    c = b - c;
    sum *= c;
    b = -n * n * b * 2;
    abs_err = ldexp(fabs(sum), -DBL_DIG);

    while (j < n) {
        a_pow *= aa;
        dj_pow *= hh_half / j;
        one_minus_dj_sum += dj_pow;
        term = one_minus_dj_sum * a_pow / (2 * j + 1);
        c = b - c;
        sum += c * term;
        abs_err = ldexp(fmax(fabs(sum), fabs(c * term)), -DBL_DIG);
        b = (j + n) * (j - n) * b / ((j + 0.5) * (j + 1));
        j++;
        if (j > 10 && fabs(sum * DBL_EPSILON) > fabs(c * term)) {
            break;
        }
    }

    abs_err += fabs(c * term);

    if (sum < 0 || (abs_err / sum) > target_precision) {
        return NPY_NAN;
    }

    return (sum / d) / (2 * NPY_PI);
}

double owensT2_accelerated(double h, double a, double ah,
        double target_precision) {
    double hh = h * h;
    double naa = -a * a;
    double y = 1 / hh;

    unsigned short int ii = 1;
    double result = 0;
    double vi = a * exp(-ah * ah / 2) / sqrt(2 * NPY_PI);
    double z = owens_t_norm1(ah / h);
    double last_z = fabs(z);

    int n = trunc(LOG_MAX_VALUE / 6);

    double d = pow(3 + sqrt(8), n);
    d = (d + 1 / d) / 2;
    double b = -1;
    double c = -d;
    int s = 1;

    int k;
    for (k = 0; k < n; k++) {
        if (fabs(z) > last_z || (fabs(result) * DBL_EPSILON > fabs(c * s * z))
            || (s * z < 0)) {
            break;
        }
        c = b - c;
        result += c * s * z;
        b = (k + n) * (k - n) * b / ((k + 0.5) * (k + 1));
        last_z = z;
        s = -s;
        z = y * (vi - ii * z);
        vi *= naa;
        ii += 2;
    } 
    double err = fabs(c * z) / result;

    if (err > target_precision) {
        return NPY_NAN;
    }

    return result * exp(-hh / 2) / (d * sqrt(2 * NPY_PI));
}

double owensT4_mp(double h, double a) {
    double hh = h * h;
    double naa = -a * a;

    unsigned short ii = 1;
    double ai = a * exp(-0.5 * hh * (1 - naa)) / (2 * NPY_PI);
    double yi = 1.0;
    double result = 0.0;

    double lim = DBL_EPSILON;

    while(1) {
        double term = ai * yi;
        result += term;
        if ((yi != 0) && (fabs(result * lim) > fabs(term))) {
            break;
        }
        ii += 2;
        yi = (1.0 - hh * yi) / ii;
        ai *= naa;
        if (ii > 1500) {
            return NPY_NAN;
        }
    }

    return result;
}

double owens_t_dispatch_basic(double h, double a, double ah) {
    int index = get_method(h, a);
    int m = ORD[index];
    int meth_code = METHODS[index - 1];
    double result = 0;

    switch(meth_code) {
        case 1:
            result = owensT1(h, a, m);
            break;
        case 2:
            result = owensT2(h, a, ah, m);
            break;
        case 3:
            result = owensT3(h, a, ah);
            break;
        case 4:
            result = owensT4(h, a, m);
            break;
        case 5:
            result = owensT5(h, a);
            break;
        case 6:
            result = owensT6(h, a);
            break;
    }

    return result;
}

double owens_t_dispatch(double h, double a, double ah) {
    if (a == 0) {
        return 0;
    }

    if (h == 0) {
        return atan(a) / (2 * NPY_PI);
    }

    if (a == 1) {
        return owens_t_norm2(-h) * owens_t_norm2(h) / 2;
    }
    
    double target_precision = DBL_EPSILON * 1000;
    double result_t1, result_t2;

    if (ah < 3) {
        result_t1 = owensT1_accelerated(h, a, target_precision);
        if (!cephes_isnan(result_t1)) {
            return result_t1;
        }
    }
    if (ah > 1) {
        result_t2 = owensT2_accelerated(h, a, ah, target_precision);
        if (!cephes_isnan(result_t2)) {
            return result_t2;
        }
    }

    double result_mp = owensT4_mp(h, a);
    if (!cephes_isnan(result_mp)) {
        return result_mp;
    }

    return owens_t_dispatch_basic(h, a, ah);
}

double owens_t(double h, double a) {
    if (cephes_isnan(h) || cephes_isnan(a)) {
        return NPY_NAN;
    }

    if (h < 0) {
        h = fabs(h);
    }

    double result = 0;
    double fabs_a = fabs(a);
    double fabs_ah = fabs_a * h;

    if (fabs_a <= 1) {
        result = owens_t_dispatch(h, fabs_a, fabs_ah);
    }
    else {
        if (fabs_ah <= 0.67) {
            double normh = owens_t_norm1(h);
            double normah = owens_t_norm1(fabs_ah);
            result = 0.25 - normh * normah -
                owens_t_dispatch(fabs_ah, (1 / fabs_a), h);
        }
        else {
            double normh = owens_t_norm2(h);
            double normah = owens_t_norm2(fabs_ah);
            result = (normh + normah) / 2 - normh * normah -
                owens_t_dispatch(fabs_ah, (1 / fabs_a), h);
        }
    }

    if (a < 0) {
        return -result;
    }

    return result;
}
